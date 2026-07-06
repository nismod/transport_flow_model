import pandas as pd
import pyarrow as pa
import pytest

from transport_flow_model import Demand, datasets


@pytest.fixture
def od():
    return pd.DataFrame(
        {
            "origin_id": ["A", "A", "B"],
            "destination_id": ["B", "C", "D"],
            "value": [30, 90, 100],
        }
    )


def test_construction_and_totals(od):
    demand = Demand(od)
    assert demand.n_pairs == 3
    assert demand.total == 220.0


def test_value_cast_to_float(od):
    demand = Demand(od)
    assert pa.types.is_float64(demand.to_table()["value"].type)


def test_column_mapping(od):
    demand = Demand(
        od.rename(columns={"value": "tons"}),
        columns={"tons": "value"},
    )
    assert demand.total == 220.0


def test_missing_columns_raise(od):
    with pytest.raises(ValueError, match="required demand columns"):
        Demand(od.drop(columns=["value"]))


def test_extra_columns_preserved(od):
    od["industry_A"] = [1.0, 2.0, 3.0]
    demand = Demand(od)
    assert "industry_A" in demand.to_table().column_names


def test_from_tntp_instance():
    instance = datasets.load_tntp("siouxfalls")
    demand = Demand.from_tntp(instance)
    assert demand.n_pairs == instance.od.to_dataframe(copy=False).shape[0]
    assert demand.total == pytest.approx(
        instance.od.to_dataframe(copy=False)["flow"].sum()
    )
    names = demand.to_table().column_names
    assert "origin_zone" in names and "destination_zone" in names


def test_from_tntp_requires_instance():
    with pytest.raises(TypeError, match="TNTPInstance"):
        Demand.from_tntp("not-an-instance")
