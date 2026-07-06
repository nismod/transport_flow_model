"""Registry of published benchmark datasets: fetch, verify checksums, cache.

Small instances (SiouxFalls) are vendored with the package so tests run
offline; larger instances are downloaded on first use into a local cache
directory (``$TFM_CACHE_DIR`` or ``~/.cache/transport-flow-model``) and kept
out of git.

Each TNTP dataset ships the published best-known equilibrium link flows
(``flow`` role) for use as regression fixtures when validating assignment
methods; see :func:`best_known_flows`.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import pandas as pd

from transport_flow_model.io import TNTPInstance, read_tntp, read_tntp_flows

_TN_BASE = "https://raw.githubusercontent.com/bstabler/TransportationNetworks/master"
_TN_LICENSE = (
    "Open data donated to the Transportation Networks for Research repository; "
    "provided as-is for research use, cite the repository"
)
_TN_CITATION = (
    "Transportation Networks for Research Core Team, Transportation Networks "
    "for Research, https://github.com/bstabler/TransportationNetworks"
)


@dataclass(frozen=True)
class DatasetFile:
    filename: str
    url: str | None  # None: vendored inside the package under data/tntp/
    sha256: str | None = None
    md5: str | None = None


@dataclass(frozen=True)
class Dataset:
    name: str
    description: str
    #: role -> file; TNTP datasets use roles "net", "trips", "flow"
    files: dict[str, DatasetFile] = field(default_factory=dict)
    license: str = ""
    provenance: str = ""
    citation: str = ""
    #: unpack zip archives after download
    extract: bool = False


def _tntp_dataset(
    name: str,
    directory: str,
    stem: str,
    description: str,
    checksums: dict[str, str],
    vendored: bool = False,
) -> Dataset:
    files = {}
    for role in ("net", "trips", "flow"):
        filename = f"{stem}_{role}.tntp"
        files[role] = DatasetFile(
            filename=filename,
            url=None if vendored else f"{_TN_BASE}/{directory}/{filename}",
            sha256=checksums[role],
        )
    return Dataset(
        name=name,
        description=description,
        files=files,
        license=_TN_LICENSE,
        provenance=f"https://github.com/bstabler/TransportationNetworks/tree/master/{directory}",
        citation=_TN_CITATION,
    )


DATASETS: dict[str, Dataset] = {
    dataset.name: dataset
    for dataset in [
        _tntp_dataset(
            "siouxfalls",
            "SiouxFalls",
            "SiouxFalls",
            "Sioux Falls, SD: 24 zones, 24 nodes, 76 links (vendored)",
            {
                "net": "ace99b24cec69c273ff0cf3d6d074110177f0cc0ae24b0c7a9f4f4cb5e27635c",
                "trips": "56f9566857f3f66730fd5c4232258d7ee3ac2931a476526331afd062f4958de7",
                "flow": "5d0b83a22ecc3ce79dabb2b2972162b78c5eda571dcb5b3687429d8397654fee",
            },
            vendored=True,
        ),
        _tntp_dataset(
            "anaheim",
            "Anaheim",
            "Anaheim",
            "Anaheim, CA: 38 zones, 416 nodes, 914 links",
            {
                "net": "99933b415e9500b13907829c37a43cfa9141714fad5af279081e28e5f9356f9a",
                "trips": "906893854cd0db4479c0b5f07678ce5616fa8e42e2b997f918c378309c66a94e",
                "flow": "eecd21c2a908b6a1c6729045ea260e96df1de026f05281d341fc281f2552cebe",
            },
        ),
        _tntp_dataset(
            "barcelona",
            "Barcelona",
            "Barcelona",
            "Barcelona, Spain: 110 zones, 1020 nodes, 2522 links",
            {
                "net": "74ea13010beca70c641417c38bc900d6d7a2a600f23f18f76e417f7090c69bbd",
                "trips": "de485bcc423ff66c8e6601ae718255614d19099c0d0536ffcdb62972e1fcbbe1",
                "flow": "cee8df9f7930e52d5779aa6d3c14b25bcede54582c827971b5b5b3fe2f6aa345",
            },
        ),
        _tntp_dataset(
            "chicago-sketch",
            "Chicago-Sketch",
            "ChicagoSketch",
            "Chicago sketch network: 387 zones, 933 nodes, 2950 links",
            {
                "net": "4396bff6101cb5ad3edaf0eb5b9aec051055cb7f0d85be907d24b43f98bf0027",
                "trips": "efe68abffc4af09e344cf1e175cfc048c08f4cd8f1f5454f74371b40e8245edc",
                "flow": "068ee541d0e4d06e1e829bebc629ecf18c0f1eaf3b8239aff82de7ac3631370b",
            },
        ),
        Dataset(
            name="usa-20-cities",
            description=(
                "Unified and validated traffic dataset for 20 US cities: "
                "GMNS-style link/node/OD CSVs per city plus reference "
                "TransCAD/AequilibraE/UXsim assignment results (276 MB zip)"
            ),
            files={
                "archive": DatasetFile(
                    filename="usa-20-cities.zip",
                    url="https://ndownloader.figshare.com/files/48908890",
                    sha256="afe5cfddbba8996290c29847e9d14f5c62fed67cd829dc5cadca7fee352a84e1",
                    md5="3f7632e00599588abecbcfc488f862b2",
                ),
            },
            license="CC BY 4.0",
            provenance="https://doi.org/10.6084/m9.figshare.24235696",
            citation=(
                "Xu, X., Zheng, Z., Hu, Z. et al. A unified dataset for the "
                "city-scale traffic assignment model in 20 U.S. cities. "
                "Sci Data 11, 325 (2024). "
                "https://doi.org/10.1038/s41597-024-03149-8"
            ),
            extract=True,
        ),
    ]
}


@dataclass(frozen=True)
class BestKnown:
    """Regression fixture for user-equilibrium assignment.

    ``objective`` is the Beckmann objective evaluated on the published
    best-known link flows (the dataset's ``flow`` file) with link cost
    ``cost * (1 + alpha * (x / capacity)^beta) + distance_cost * length``
    (``beta == 0`` links cost ``cost * (1 + alpha)``). The published
    equilibrium link costs are reproduced by this formula to ~1e-14, so the
    same convention must be used when solving the instance.
    """

    objective: float
    distance_cost: float = 0.0


#: Beckmann objectives computed from each dataset's published link flows.
BEST_KNOWN: dict[str, BestKnown] = {
    "siouxfalls": BestKnown(objective=4231335.28710744),
    "anaheim": BestKnown(objective=1286032.171096032),
    "barcelona": BestKnown(objective=1265654.9220317658),
    # Chicago's published solution uses generalized cost with 0.04/mile
    "chicago-sketch": BestKnown(objective=17313018.73874779, distance_cost=0.04),
}


def available() -> list[str]:
    """Names of all registered datasets."""
    return sorted(DATASETS)


def cache_dir() -> Path:
    """Dataset cache directory (``$TFM_CACHE_DIR`` or ``~/.cache``-based)."""
    root = os.environ.get("TFM_CACHE_DIR")
    if root:
        return Path(root)
    return Path.home() / ".cache" / "transport-flow-model"


def fetch(name: str, *, cache: str | Path | None = None) -> dict[str, Path]:
    """Return local paths for a dataset's files, downloading if needed.

    Returns a dict keyed by file role (``net``/``trips``/``flow`` for TNTP
    datasets, ``archive`` and ``dir`` for extracted archives). Downloads are
    verified against registered checksums; vendored files resolve to the
    package's own data directory without network access.
    """
    dataset = _get(name)
    base = Path(cache) if cache is not None else cache_dir()
    target_dir = base / "datasets" / dataset.name
    paths: dict[str, Path] = {}
    for role, file in dataset.files.items():
        if file.url is None:
            path = _vendored_path(file.filename)
        else:
            path = target_dir / file.filename
            if not path.exists():
                _download(file, path)
            _verify(file, path)
        paths[role] = path
        if dataset.extract and path.suffix == ".zip":
            extracted = target_dir / f"{path.stem}"
            if not extracted.is_dir():
                with zipfile.ZipFile(path) as archive:
                    archive.extractall(extracted)
            paths["dir"] = extracted
    return paths


def load_tntp(name: str, *, cache: str | Path | None = None) -> TNTPInstance:
    """Fetch a TNTP dataset and parse it into network and demand."""
    paths = fetch(name, cache=cache)
    if "net" not in paths:
        raise ValueError(f"Dataset {name!r} is not in TNTP format")
    return read_tntp(paths["net"], paths["trips"])


def best_known_flows(name: str, *, cache: str | Path | None = None) -> pd.DataFrame:
    """Published best-known equilibrium link flows for a TNTP dataset."""
    paths = fetch(name, cache=cache)
    if "flow" not in paths:
        raise ValueError(f"Dataset {name!r} has no published link flows")
    return read_tntp_flows(paths["flow"])


def _get(name: str) -> Dataset:
    try:
        return DATASETS[name]
    except KeyError:
        raise ValueError(
            f"Unknown dataset {name!r}; available: {', '.join(available())}"
        ) from None


def _vendored_path(filename: str) -> Path:
    stem = filename.split("_")[0]
    path = resources.files("transport_flow_model") / "data" / "tntp" / stem / filename
    return Path(str(path))


def _download(file: DatasetFile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        file.url, headers={"User-Agent": "transport-flow-model/datasets"}
    )
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            with urllib.request.urlopen(request) as response:
                shutil.copyfileobj(response, handle)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def _verify(file: DatasetFile, path: Path) -> None:
    checks = [("sha256", file.sha256), ("md5", file.md5)]
    for algorithm, expected in checks:
        if expected is None:
            continue
        digest = hashlib.new(algorithm)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected.lower():
            raise ValueError(
                f"Checksum mismatch for {path} ({algorithm}): "
                f"expected {expected}, got {digest.hexdigest()}. "
                "Delete the file to re-download."
            )
