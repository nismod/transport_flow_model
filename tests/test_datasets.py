import hashlib

import pytest

from transport_flow_model import datasets


def test_registry_is_complete():
    assert datasets.available() == [
        "anaheim",
        "barcelona",
        "chicago-sketch",
        "siouxfalls",
        "usa-20-cities",
    ]
    for dataset in datasets.DATASETS.values():
        assert dataset.description
        assert dataset.license
        assert dataset.provenance
        assert dataset.citation
        for file in dataset.files.values():
            assert file.sha256 or file.md5
            # vendored files must exist in the package
            if file.url is None:
                assert datasets._vendored_path(file.filename).exists()


def test_tntp_datasets_have_solution_fixtures():
    for name, dataset in datasets.DATASETS.items():
        if "net" in dataset.files:
            assert "flow" in dataset.files
            assert name in datasets.BEST_KNOWN
            assert datasets.BEST_KNOWN[name].objective > 0


def test_fetch_vendored_siouxfalls_is_offline():
    paths = datasets.fetch("siouxfalls")
    assert sorted(paths) == ["flow", "net", "trips"]
    for role, path in paths.items():
        assert path.exists(), role


def test_fetch_unknown_dataset_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        datasets.fetch("nope")


def test_load_tntp_siouxfalls():
    instance = datasets.load_tntp("siouxfalls")
    assert len(instance.network.to_dataframe()) == 76
    assert instance.od.to_dataframe()["flow"].sum() == pytest.approx(360600.0)


def test_best_known_flows_siouxfalls():
    flows = datasets.best_known_flows("siouxfalls")
    assert len(flows) == 76
    assert (flows["flow"] > 0).all()


def test_fetch_downloads_and_verifies(tmp_path, monkeypatch):
    payload = b"link data\n"
    source = tmp_path / "source.tntp"
    source.write_text(payload.decode())
    good = datasets.Dataset(
        name="test-good",
        description="test",
        files={
            "net": datasets.DatasetFile(
                filename="test_net.tntp",
                url=source.as_uri(),
                sha256=hashlib.sha256(payload).hexdigest(),
            )
        },
        license="test",
        provenance="test",
        citation="test",
    )
    bad = datasets.Dataset(
        name="test-bad",
        description="test",
        files={
            "net": datasets.DatasetFile(
                filename="test_net.tntp",
                url=source.as_uri(),
                sha256=hashlib.sha256(b"other").hexdigest(),
            )
        },
        license="test",
        provenance="test",
        citation="test",
    )
    monkeypatch.setitem(datasets.DATASETS, "test-good", good)
    monkeypatch.setitem(datasets.DATASETS, "test-bad", bad)

    paths = datasets.fetch("test-good", cache=tmp_path / "cache")
    assert paths["net"].read_bytes() == payload

    with pytest.raises(ValueError, match="Checksum mismatch"):
        datasets.fetch("test-bad", cache=tmp_path / "cache")
