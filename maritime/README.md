# Maritime transport disruption inputs

This workflow prepares maritime network costs, port-to-port trade demand, and edges selected for disruption for the NISMOD transport flow model. Run the preprocessing notebooks to generate three model inputs:

| Input | Stored Location | Purpose |
| --- | --- | --- |
| Network | `processed_data/network/network.csv` | Maritime links and vessel-specific transport costs and travel times |
| OD demand | `processed_data/od/od.csv` | Trade flows between origin and destination ports |
| Failure set | `processed_data/damages/failure_set.csv` | Edge IDs selected for disruption |


## Data sources

### Maritime network and transport costs

The methodological reference is:

Verschuur, J., Koks, E. E. & Hall, J. W. (2022). *Ports’ criticality in international trade and global supply-chains*. Nature Communications **13**, 4321.

Consult the supplementary material for vessel speeds and transport costs (they are summarized also in Network_Maritime.ipynb, one of the pre-processing jupyter notebooks). Supplementary Table 7 supplies transport cost parameters;  Supplementary Table 3 is the source for vessel speeds.

Download the maritime network-edges file from the `Network` section of the [Mendeley dataset, version 1](https://data.mendeley.com/datasets/vzzy3b9gg4/1), and place it at:

```text
raw_data/network/edges_maritime_corrected.gpkg
```

The file is read by the current preprocessing code. 

`Network_Maritime.ipynb` combines the geospatial network with transport parameters. Choose `vessel_type` from `container`, `dry_bulk`, `liquid_bulk`, `roro`, or `general_cargo`. The notebook exports the selected vessel's network to `processed_data/network/network.csv`.

### Port-to-port trade demand

The same [Mendeley dataset](https://data.mendeley.com/datasets/vzzy3b9gg4/1) contains port trade data under `Port_to_port_network`. The file to download is `port_trade_network.csv`.


The notebook infers origin–destination trade flows, writes the intermediate file `raw_data/od/intermediate_data/Worldwide_port_export_vol.parquet`, and produces `processed_data/od/od.csv`. The current demand total sums industries 1–11; it is not filtered by vessel type.

### Chokepoints geometries

Place the following files in `raw_data/failure_set/`:

```text
Chokepoints.gejson
```

The dataset can be found on the PortWatch Website and contains 28 chokepoints:

See: https://portwatch.imf.org/datasets/fa9a5800b0ee4855af8b2944ab1e07af/

It is important to download the .geojson version.



`Failure_Set_maritime.ipynb` imports functions from `Chokepoint_Edges.py` to identify maritime edges intersecting chokepoint buffers. It writes a mapping to `raw_data/failure_set/intermediate_data/chokepoint_edges.csv`. 

Set `selected_chokepoints` in the notebook to select one or more chokepoints. Their unique edge IDs are exported to `processed_data/damages/failure_set.csv`.

**Current model semantics:** each row of the failure set is evaluated as a separate single-edge removal scenario. Selecting all edges belonging to a chokepoint does not, by itself, simulate their simultaneous closure.

## Prepare the inputs

### Python environment

Use Python 3.10 or newer, consistent with the parent model's declared requirement. The maritime preprocessing and inspection notebook dependencies are listed in [requirements.txt](requirements.txt). This includes packages imported by the current notebooks even where those imports are not needed by every cell.

From the parent `transport_flow_model` directory, create an environment and install the notebook dependencies (macOS/Linux commands):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r maritime/requirements.txt
python -m ipykernel install --user --name maritime --display-name "Python (maritime)"
cd maritime
python -m jupyterlab
```

Select **Python (maritime)** as the notebook kernel. If you already have an environment for the model, activate it and install the requirements there instead of creating another environment. If you have only the maritime folder, run `python -m pip install -r requirements.txt` from that folder.

These requirements cover preprocessing and plotting; they do not install the transport flow model itself. Before running allocation/disruption, follow the installation instructions in the [parent README](../README.md) using the same environment. The parent project's `pyproject.toml` remains the source for model dependencies and build configuration.

The requirements list was checked against notebook imports, but a fresh installation and complete preprocessing run have not been verified. Versions are intentionally not pinned; this file is not an exact reproducibility lock. Python standard-library modules (such as `pathlib`, `json`, and `os`) require no separate installation. `Chokepoint_Edges.py` is a local helper and must remain alongside the notebooks.

### Run the notebooks

Run notebook kernels with this `maritime` directory as the working directory. Relative paths in the notebooks are resolved from that directory.

Download or obtain the source files described above, then run:

1. [Network_Maritime.ipynb](Network_Maritime.ipynb): select the vessel and export its network.
2. [OD_maritime.ipynb](OD_maritime.ipynb): infer and format port-to-port demand.
3. [Failure_Set_maritime.ipynb](Failure_Set_maritime.ipynb): select chokepoints and export failure edges.

Restart each kernel and run cells in order to check that the workflow does not depend on previously defined variables.

## Configure and run the model

The configuration file `config.maritime.json` belongs in the parent `transport_flow_model` directory. 


The network notebook Failure_Set_maritime.ipynb updates the existing configuration's results path using the selected vessel. Complete allocation and disruption for that vessel before exporting another vessel's network: `network.csv` is overwritten on each export. Separate results directories preserve runs for different vessels; repeating a run for the same vessel can overwrite its earlier results.

From the parent `transport_flow_model` directory, run:

```bash
python scripts/flow_model/flow_allocation.py config.maritime.json
python scripts/flow_model/flow_disruptions.py config.maritime.json
```

Use the same configuration for both stages. Disruption reads the baseline allocation from the configured results directory.


## Files shared through Git

Keep the notebooks, reusable Python functions, documentation, and example configuration in Git. The local `.gitignore` excludes downloaded data, generated inputs, results, and temporary files. These remain on disk and can be used normally by the notebooks. Ignore rules do not remove files already tracked by Git.
