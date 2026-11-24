<p align="center">
  <strong>English</strong> | <a href="./README_CN.md">简体中文</a>
</p>

# Delayed Advice Audit

Research software and analyzed data for auditing fitted-policy performance, runtime message dependence and predefined message-channel value in robot simulation.

## Contents

- `src/advice_audit/`: simulation, policies, data collection, fitting and evaluation.
- `scripts/`: executable research workers and baseline utilities.
- `experiments/`: state, visual-pipeline, fixed-channel and decoder-control drivers.
- `data/analyzed/`: 20 numerical analysis/audit JSON records, three NPZ arrays and a file manifest.
- `reproduction/`: a compact NumPy audit of eight fitting/selection cells, 500 frozen choices and six primary paired-result vectors.
- `tests/`, `tests_metaworld/`, `tests_vision/`: verification suites for their respective environments.

## Basic installation and checks

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev,sim,plot]'
python -m pytest -q tests
python -m advice_audit.cli --help
```

MetaWorld and visual dependencies are listed in their separate requirements files; use separate environments. Native drivers require separately obtained demonstrations, actor checkpoints and the relevant simulation dependencies. Their recorded data/model locations refer to the extracted project root, not a particular machine. Start each new run in a new output directory.

## Offline audit

```sh
python -m pip install 'numpy>=1.24,<3'
python reproduction/reproduce_offline.py --out outputs/offline_check
```

This audit starts from cached normalized kernel distances and development labels. It verifies fitting/selection, frozen decisions and paired-result arithmetic; it does not independently verify visual feature extraction or distance construction and does not execute ACT or native simulation. The numerical inputs and comparison targets retain their original values. See `reproduction/README.md`.

## Release scope and licenses

The distribution contains code, numerical data, minimal usage documentation and license notices. Third-party demonstrations, pretrained neural-network weights and the complete native trajectory archive are not redistributed. Paper PDFs, TeX/BibTeX sources, rendered manuscript figures, cover letters and other submission documents are not release artifacts.

Original code: MIT (`LICENSE`). Numerical data: CC BY 4.0 (`DATA_LICENSE.md`). Third-party components retain their notices (`THIRD_PARTY_NOTICES.md`). Citation metadata are in `CITATION.cff`. Version 1.0.1.
