# %%
from pathlib import Path

import pandas as pd
import pytest

from transport_flow_model.assignment import _assign_congestion
from transport_flow_model.demand import Demand
from transport_flow_model.network import Network

EXAMPLE_DATA = Path(__file__).resolve().parents[1] / "example" / "processed_data"


def congestion_example_tables() -> tuple[pd.DataFrame, pd.DataFrame, int]:
    network_data = pd.read_csv(EXAMPLE_DATA / "network" / "congestion_network_NI.csv")
    od_data = pd.read_csv(EXAMPLE_DATA / "od" / "congestion_od_NI.csv")

    network = Network(network_data)
    demand = Demand(od_data)

    result = _assign_congestion(network, demand, include_paths=True)
    assigned_paths = (
        result["paths"]
        .select(["origin_id", "destination_id", "flow", "edge_path", "cost"])
        .to_pandas()
    )
    unassigned = (
        result["unassigned"]
        .select(["origin_id", "destination_id", "value"])
        .to_pandas()
        .rename(columns={"value": "unassigned"})
    )
    original_flows = (
        od_data.groupby(["origin_id", "destination_id"], as_index=False)["value"]
        .sum()
        .rename(columns={"value": "original_flow"})
    )
    odpfc = assigned_paths.merge(
        unassigned,
        on=["origin_id", "destination_id"],
        how="outer",
    )
    odpfc = odpfc.merge(
        original_flows,
        on=["origin_id", "destination_id"],
        how="outer",
    )
    odpfc[["flow", "unassigned", "original_flow"]] = odpfc[
        ["flow", "unassigned", "original_flow"]
    ].fillna(0)
    edge_flows = result["link_flows"].select(["edge_id", "flow"]).to_pandas()

    return odpfc, edge_flows, result["iterations"]


def test_assign_congestion_reads_example_data():
    odpfc, edge_flows, iterations = congestion_example_tables()
    network_data = pd.read_csv(EXAMPLE_DATA / "network" / "congestion_network_NI.csv")
    od_data = pd.read_csv(EXAMPLE_DATA / "od" / "congestion_od_NI.csv")

    assert odpfc["flow"].sum() > 0
    assert iterations > 0
    assert "unassigned" in odpfc.columns
    assert "original_flow" in odpfc.columns
    assert list(edge_flows.columns) == ["edge_id", "flow"]
    assert len(edge_flows) == len(network_data)
    od_totals = odpfc.groupby(["origin_id", "destination_id"]).agg(
        assigned=("flow", "sum"),
        unassigned=("unassigned", "first"),
        original_flow=("original_flow", "first"),
    )
    assert (od_totals["assigned"] + od_totals["unassigned"]).to_list() == pytest.approx(
        od_totals["original_flow"].to_list()
    )
    assert od_totals["original_flow"].sum() == pytest.approx(od_data["value"].sum())

    capacities = dict(
        zip(
            network_data["edge_id"],
            network_data["capacity"],
            strict=True,
        )
    )
    for edge_id, flow in zip(
        edge_flows["edge_id"],
        edge_flows["flow"],
        strict=True,
    ):
        assert int(flow) <= int(capacities[edge_id])
