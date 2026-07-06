"""Readers for TNTP-format transport network problems.

TNTP is the de-facto standard exchange format used by the Transportation
Networks for Research repository (https://github.com/bstabler/TransportationNetworks),
which also documents the format. A problem instance is described by:

- ``<name>_net.tntp``: link table with metadata header
- ``<name>_trips.tntp``: origin-destination trip matrix
- ``<name>_flow.tntp``: (optional) best-known equilibrium link flows

Conventions handled here:

- Node ids are 1-based integers; zones (demand centroids) are nodes
  ``1..n_zones``.
- ``<FIRST THRU NODE>`` marks the lowest node id that paths may pass
  *through*. Nodes below it are centroids that may only start or end a trip,
  never appear mid-path. :func:`read_tntp` enforces this by splitting each
  centroid into an origin-only node (keeps the original id) and a
  destination-only node (original id plus ``centroid_offset``), with trip
  destinations remapped to match.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import pandas as pd

from transport_flow_model.model import OD, Network

#: Link attribute columns in TNTP column order.
TNTP_NET_COLUMNS = (
    "init_node",
    "term_node",
    "capacity",
    "length",
    "free_flow_time",
    "b",
    "power",
    "speed",
    "toll",
    "link_type",
)

#: Canonical network schema produced from TNTP link attributes. ``cost`` is
#: the free-flow time; ``alpha``/``beta`` are the BPR volume-delay parameters
#: (TNTP columns ``b`` and ``power``). GIS-sourced networks should be
#: normalized to the same columns so both paths converge on one
#: representation.
NETWORK_COLUMNS = (
    "edge_from",
    "edge_to",
    "edge_id",
    "capacity",
    "cost",
    "alpha",
    "beta",
    "length",
    "speed",
    "toll",
    "link_type",
)

_END_OF_METADATA = "<END OF METADATA>"


@dataclass
class TNTPInstance:
    """A parsed TNTP problem: network, demand and instance metadata."""

    network: Network
    od: OD | None
    n_zones: int
    n_nodes: int
    first_thru_node: int
    n_links: int
    #: Offset added to centroid node ids in destination role, or None if no
    #: centroid was split (``first_thru_node <= 1`` or splitting disabled).
    centroid_offset: int | None = None

    def zone_ids(self, node_ids: pd.Series) -> pd.Series:
        """Map (possibly split) node ids back to original TNTP zone ids."""
        node_ids = pd.Series(node_ids).astype("int64")
        if self.centroid_offset is None:
            return node_ids
        return node_ids.where(
            node_ids <= self.centroid_offset, node_ids - self.centroid_offset
        )


def read_tntp(
    net_path: str | Path,
    trips_path: str | Path | None = None,
    *,
    split_centroids: bool = True,
    keep_zero_flows: bool = False,
) -> TNTPInstance:
    """Read a TNTP network (and optionally its trip matrix).

    Returns a :class:`TNTPInstance` whose ``network`` is ready for
    :meth:`Network.allocate` (``cost`` set to free-flow time) and whose ``od``
    holds the trip matrix as sparse origin/destination/flow rows.

    When ``split_centroids`` is true (the default) and the instance declares
    ``<FIRST THRU NODE>`` greater than 1, centroid nodes are split so that
    no path can route *through* a centroid: links out of a centroid keep the
    original tail id, links into a centroid have their head id offset by
    ``centroid_offset``, and OD destination ids are offset to match. Use
    :meth:`TNTPInstance.zone_ids` to map result node ids back to zones.
    """
    metadata, links = _read_tntp_net(Path(net_path))
    n_zones = int(metadata.get("NUMBER OF ZONES", 0))
    n_nodes = int(metadata.get("NUMBER OF NODES", 0))
    n_links = int(metadata.get("NUMBER OF LINKS", len(links)))
    first_thru_node = int(metadata.get("FIRST THRU NODE", 1))

    od_data = None
    if trips_path is not None:
        od_data = _read_tntp_trips(Path(trips_path), keep_zero_flows=keep_zero_flows)

    centroid_offset = None
    if split_centroids and first_thru_node > 1:
        centroid_offset = int(
            max(
                n_nodes,
                links["init_node"].max(),
                links["term_node"].max(),
            )
        )
        is_centroid_head = links["term_node"] < first_thru_node
        links.loc[is_centroid_head, "term_node"] += centroid_offset
        if od_data is not None:
            remap = od_data["destination_id"] < first_thru_node
            od_data.loc[remap, "destination_id"] += centroid_offset

    network = Network(_network_frame(links))
    od = OD(od_data) if od_data is not None else None
    return TNTPInstance(
        network=network,
        od=od,
        n_zones=n_zones,
        n_nodes=n_nodes,
        first_thru_node=first_thru_node,
        n_links=n_links,
        centroid_offset=centroid_offset,
    )


def read_tntp_flows(path: str | Path) -> pd.DataFrame:
    """Read a TNTP best-known link flows file (``<name>_flow.tntp``).

    Returns a DataFrame with columns ``edge_from``, ``edge_to``, ``flow``
    and ``cost`` (equilibrium link travel cost).
    """
    _, body = _split_metadata(Path(path).read_text())
    data = pd.read_csv(StringIO(body.replace(";", " ")), sep=r"\s+", comment="~")
    data.columns = [str(c).strip().lower() for c in data.columns]
    data = data.rename(columns={"from": "edge_from", "to": "edge_to", "volume": "flow"})
    data["edge_from"] = data["edge_from"].astype("int64")
    data["edge_to"] = data["edge_to"].astype("int64")
    return data.loc[:, ["edge_from", "edge_to", "flow", "cost"]]


def _split_metadata(text: str) -> tuple[dict[str, str], str]:
    """Split a TNTP file into its ``<KEY> value`` metadata header and body."""
    end = text.find(_END_OF_METADATA)
    if end < 0:
        return {}, text
    metadata = {}
    for match in re.finditer(r"<(.+?)>([^\n<]*)", text[:end]):
        metadata[match.group(1).strip().upper()] = match.group(2).strip()
    return metadata, text[end + len(_END_OF_METADATA) :]


def _read_tntp_net(path: Path) -> tuple[dict[str, str], pd.DataFrame]:
    metadata, body = _split_metadata(path.read_text())
    links = pd.read_csv(
        StringIO(body.replace(";", " ")),
        sep=r"\s+",
        comment="~",
        header=None,
    )
    if links.shape[1] < len(TNTP_NET_COLUMNS):
        raise ValueError(
            f"Expected {len(TNTP_NET_COLUMNS)} link columns in {path}, "
            f"found {links.shape[1]}"
        )
    links = links.iloc[:, : len(TNTP_NET_COLUMNS)]
    links.columns = list(TNTP_NET_COLUMNS)
    links["init_node"] = links["init_node"].astype("int64")
    links["term_node"] = links["term_node"].astype("int64")
    for column in TNTP_NET_COLUMNS[2:]:
        links[column] = pd.to_numeric(links[column], errors="coerce").astype("float64")
    return metadata, links


def _network_frame(links: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "edge_from": links["init_node"],
            "edge_to": links["term_node"],
            "edge_id": pd.RangeIndex(len(links)).astype("int64"),
            "capacity": links["capacity"],
            "cost": links["free_flow_time"],
            "alpha": links["b"],
            "beta": links["power"],
            "length": links["length"],
            "speed": links["speed"],
            "toll": links["toll"],
            "link_type": links["link_type"],
        }
    )


_TRIPS_PAIR = re.compile(r"(\d+)\s*:\s*([0-9.eE+\-]+)\s*;")
_TRIPS_ORIGIN = re.compile(r"Origin\s+(\d+)")


def _read_tntp_trips(path: Path, *, keep_zero_flows: bool) -> pd.DataFrame:
    _, body = _split_metadata(path.read_text())
    rows: list[tuple[int, int, float]] = []
    blocks = _TRIPS_ORIGIN.split(body)
    # split() yields [preamble, origin_1, block_1, origin_2, block_2, ...]
    for origin, block in zip(blocks[1::2], blocks[2::2], strict=True):
        origin_id = int(origin)
        for pair in _TRIPS_PAIR.finditer(block):
            rows.append((origin_id, int(pair.group(1)), float(pair.group(2))))
    data = pd.DataFrame(rows, columns=["origin_id", "destination_id", "flow"])
    data["origin_id"] = data["origin_id"].astype("int64")
    data["destination_id"] = data["destination_id"].astype("int64")
    data["flow"] = data["flow"].astype("float64")
    if not keep_zero_flows:
        data = data[data["flow"] > 0].reset_index(drop=True)
    return data
