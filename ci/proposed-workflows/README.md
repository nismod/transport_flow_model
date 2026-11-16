# Proposed workflow changes

These are updated versions of the files in `.github/workflows/`. They are kept
here because the automated session that wrote them is not allowed to push
changes to workflow files. To apply them:

```bash
cp ci/proposed-workflows/*.yml .github/workflows/
git rm -r ci/proposed-workflows
git commit -am "Apply release, test and docs workflow updates"
```

What changes:

- `publish.yml`: builds abi3 wheels for manylinux and musllinux (x86_64,
  aarch64), macOS (x86_64, arm64) and Windows (x64), plus an sdist, with
  `PyO3/maturin-action`. It smoke-tests the native wheels, checks that a
  release tag equals `v` + the `Cargo.toml` version, and publishes to PyPI
  with trusted publishing on `release: published` only. `workflow_dispatch`
  is a build-only dry run. The previous workflow built a single
  `linux_x86_64` wheel, which PyPI rejects.
- `test.yml`: also runs `cargo test`.
- `docs.yml`: sets up a Rust toolchain, since installing the package now
  builds the extension.

Not verified: these have not run on GitHub Actions. Check the runner labels
(`macos-15-intel`), the action versions, and that the PyPI trusted publisher
for `nismod-tfm` is registered for this repository, the `publish.yml`
workflow and the `pypi` environment.
