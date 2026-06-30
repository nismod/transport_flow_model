"""Configuration utilities for transport flow modelling"""

import json


def load_config(config_path=None):
    """Load configuration from JSON file.

    Parameters
    ----------
    config_path : str, optional
        Path to the configuration JSON file. Defaults to "./config.json".

    Returns
    -------
    dict
        Configuration dictionary loaded from the JSON file.
    """
    if config_path is None:
        config_path = "./config.json"

    with open(config_path, "r") as config_fh:
        config = json.load(config_fh)
    return config
