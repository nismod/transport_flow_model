# Transport Flow Model

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19566285.svg)](https://doi.org/10.5281/zenodo.19566285)

Scripts to run a transport flow model with capacity constraints
The model assumes flows along edges without constrains, till they reach capacity

To run the model:

```bash
pip install -e .
python scripts/flow_model/flow_allocation.py ./config.example.json
```
