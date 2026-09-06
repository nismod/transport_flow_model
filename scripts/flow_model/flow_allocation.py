#!/usr/bin/env python
"""Allocate origin-destination flows to a network.

Driver over the transport_flow_model v0 API: load config, run
``assign``, write results.
"""

import argparse
import logging

from transport_flow_model import RunConfig, assign


def main(config: RunConfig):
    results_folder = config.paths.results / "flow_od_paths"
    results_folder.mkdir(parents=True, exist_ok=True)

    logging.info("assign (method=%s)", config.assignment.method)
    network = config.load_network()
    result = assign(
        network,
        config.load_demand(),
        config.assignment.method,
        include_paths=True,
        **config.assignment.options(network),
    )

    logging.info("Writing results to %s", results_folder)
    result.link_flows.to_pandas().to_csv(
        results_folder / "network_edge_total_flows.csv", index=False
    )
    result.unassigned.to_pandas().rename(columns={"value": "flow"}).to_csv(
        results_folder / "unassigned_od_flows.csv", index=False
    )
    od_flows = result.paths.to_pandas()
    od_flows["edge_path"] = od_flows["edge_path"].map(list)
    od_flows.to_csv(results_folder / "od_flows.csv", index=False)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(process)d %(filename)s %(message)s", level=logging.INFO
    )
    logging.info("Start flow_allocation.py")
    parser = argparse.ArgumentParser(
        prog="flow_allocation",
        description="Allocate origin-destination flows to a network",
    )
    parser.add_argument("config", help="Path to config.json")
    args = parser.parse_args()
    main(RunConfig.from_json(args.config))
    logging.info("Done.")
