<p align="center">
  <a href="./README.md">简体中文</a> | <strong>English</strong>
</p>

<div align="center">

# Delayed Advice Audit

**Research Software & Analyzed Data for Auditing Fitted-Policy Performance, Runtime Message Dependence, and Delayed Message-Channel Value in Robot Simulation**

![Version](https://img.shields.io/badge/Version-1.0.1-6366f1?style=flat-square)
![Python](https://img.shields.io/badge/Python-%3E%3D3.8-3776AB?style=flat-square&logo=python&logoColor=white)
![Code License](https://img.shields.io/badge/Code-MIT-10b981?style=flat-square)
![Data License](https://img.shields.io/badge/Data-CC_BY_4.0-0ea5e9?style=flat-square)

</div>

---

## Overview

**Delayed Advice Audit** provides end-to-end simulation drivers, auditing utilities, offline reproducibility bundles, and analyzed numerical datasets for evaluating:

1. **Fitted-Policy Performance**: Comparing local/cheap policies, packet-augmented selectors, and full-action or reject baselines across state-based and vision-based manipulation tasks.
2. **Runtime Message Dependence**: Auditing how runtime gating and selection depend on delayed advice packets across independent training seeds and test environments.
3. **Predefined Message-Channel & Decoder Value**: Quantifying stale (`S`), fresh (`F`), and elapsed/control (`E`) channel interventions as well as time-sensitive decoder controls in native MuJoCo / MetaWorld simulations.

---

## Repository Structure

```text
Delayed-Advice-Audit/
├── src/advice_audit/                # Core Python package (installable via pyproject.toml)
│   ├── envs.py, metaworld_adapter.py # Simulation environments & MetaWorld adapters
│   ├── policies.py, vision.py        # Policy interfaces & visual feature pipelines
│   ├── act_core.py, act_bridge.py,   # Action Chunking with Transformers (ACT) bridge & adaptation
│   │   act_adaptation.py
│   ├── value_kernel.py, model.py,    # Value-kernel fitting, calibration, repair & selector models
│   │   calibration.py, repair.py,
│   │   repair_selector.py, features.py
│   ├── protocol.py, evaluation.py,   # Audit protocol, counterfactual evaluation & reporting
│   │   counterfactual.py, report.py
│   └── cli.py                        # CLI entry point (`advice-audit` / `python -m advice_audit.cli`)
│
├── experiments/                     # Study-specific experiment & analysis drivers
│   ├── state/                        # State-based benchmark campaign (drawer/door/button-press)
│   ├── visual_fit/                   # Visual critic-curve & arrival-confirmation fitting
│   ├── visual_pipeline/              # Multi-seed visual replication & channel-v2 pipelines
│   ├── visual_channel/               # Stale/Fresh/Elapsed (S/F/E) visual channel & temporal audits
│   └── decoder_control/              # Legacy vs. time-sensitive decoder control & replay audits
│
├── scripts/                         # Standalone workers, sweeps, ACT/vision utilities & baselines
├── configs/                         # Study configuration JSONs (alignment, repair selector, pilot)
├── data/analyzed/                   # 23 analyzed JSON/NPZ records & manifest.json
├── reproduction/                    # Compact NumPy offline audit bundle (8 cells, 500 frozen choices)
│   ├── cached_distances/             # Normalized kernel distances (pick-place-v3.npz, push-v3.npz)
│   ├── choices/                      # 500 frozen runtime decision records
│   └── reproduce_offline.py          # Zero-simulation NumPy verification script
│
├── tests/                           # Core unit & integration tests
├── tests_metaworld/                 # MetaWorld environment verification suite
├── tests_vision/                    # Vision pipeline verification suite
├── pyproject.toml                   # Package metadata & optional dependency groups (sim/dev/plot)
├── requirements-metaworld.txt       # Isolated dependencies for MetaWorld experiments
├── requirements-vision.txt          # Isolated dependencies for visual pipelines
├── CITATION.cff                     # Machine-readable citation metadata (v1.0.1)
├── LICENSE & DATA_LICENSE.md        # MIT (code) and CC BY 4.0 (numerical data)
└── THIRD_PARTY_NOTICES.md           # Third-party component notices
```

---

## Key Experimental Results & Audit Summary

All numbers below are taken directly from the verified records in `data/analyzed/`.

### 1. Audit Integrity & Zero-Error Replay Checks

| Study Audit File | Scope Checked | Key Verification Checks | Max Reward Error |
| :--- | :--- | :--- | :---: |
| `state_checks.json` | 360 cases, 126 dev groups | 18,000 replayed choices, 2,899 unique leaves, model hashes verified | **0** |
| `visual_replication_audit.json` | 300 logical cases (3 seeds × 100 envs) | 1,200 decisions, 1,584 native option tails, 300 source-100 trajectories | **Verified** |
| `visual_channel_audit.json` | 60 primary + 6 identity + 2 smoke envs | 816 decisions, 199 branches, 136 source packets | **0.0** |
| `decoder_audit.json` | 68 cases (60 main + 6 identity + 2 smoke) | 544 choices, 320 unique paths, 204 index & 136 packet hash checks | **0.0** |

### 2. State-Based Manipulation Benchmark (`data/analyzed/state.json`)

Evaluated across MetaWorld state tasks comparing the `reject` baseline, `fixed_expert32`, and paired contrasts (mean reward difference with confidence intervals):

| Task | `reject` Reward (Success) | `fixed_expert32` Reward (Success) | `packet_cheap − local_cheap` | `packet_full − packet_cheap` | `packet_full − reject` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `drawer-open-v3` | **8.0539** (100.0%) | 3.1336 (30.8%) | `0.0000` `[0.0000, 0.0000]` | `-0.4092` `[-0.7100, -0.1653]` | `-0.4092` `[-0.7022, -0.1620]` |
| `door-open-v3` | **8.5962** (94.2%) | 6.0932 (80.0%) | `+0.0051` `[-0.0125, 0.0541]` | `-0.1915` `[-0.3225, -0.0809]` | `-0.1933` `[-0.3220, -0.0747]` |
| `button-press-v3` | **1.1248** (100.0%) | 0.7551 (75.0%) | `0.0000` `[0.0000, 0.0000]` | `0.0000` `[0.0000, 0.0000]` | `0.0000` `[0.0000, 0.0000]` |

> **Takeaway**: On state tasks, blindly applying delayed full-action packets (`packet_full`) degrades return relative to `packet_cheap` and `reject` on `drawer-open-v3` and `door-open-v3`, while `fixed_expert32` suffers substantial drops when unguarded.

### 3. Visual Pipeline Multi-Seed Replication (`data/analyzed/visual_replication_analysis.json`)

3 independently trained visual systems evaluated on 100 independent test environments (300 logical arrival cases per task):

| Task | `reject` Return (Success) | `with_packet / cheap_actions` | `with_packet / all_actions` | `packet_cheap − local_cheap` | `packet_full − reject` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `push-v3` | 7.4215 (97.3%) | 7.2835 (94.7%) | **7.4893** (96.7%) | `-0.0094` `[-0.4274, 0.2602]` | `+0.0678` `[-0.5931, 0.7221]` |
| `pick-place-v3` | **6.1985** (93.3%) | 6.1774 (**95.3%**) | 5.8161 (92.0%) | `+0.4257` `[-0.3671, 1.5998]` | `-0.3824` `[-1.5000, 0.3193]` |

### 4. Visual Channel Intervention & Decoder Control (`n_env = 60` Main Cohort)

- **Visual Channel (`visual_channel_analysis.json`)**:
  - Under `with_packet / cheap_actions`, Stale (`S`), Fresh (`F`), and Elapsed (`E`) all achieve reward **6.5569** (`any_success = 88.3%`, `terminal_success = 58.3%`).
  - Under `with_packet / all_actions`, Stale (`S`) reaches **6.0389** (`83.3%`) vs. Fresh (`F`) **5.8962** (`80.0%`), yielding a primary contrast `refresh_minus_stale` of **`-0.1428`** (`CI [-0.4932, 0.2285]`).
- **Decoder Control (`decoder_analysis.json`)**:
  - Baseline `R`: reward **6.9003** (`any_success = 91.7%`, `terminal_success = 66.7%`).
  - Fixed channels: `fixed_S_legacy` **6.8716** (`93.3%`), `fixed_F_legacy` **6.4413** (`83.3%`), **`fixed_F_time` 7.0204** (`88.3%`, `terminal_success = 71.7%`).
  - Gated channels: `gate_S_legacy` **6.4493**, `gate_F_legacy` **6.2909**, `gate_F_time` **6.2611** (with `24` time-sensitive source choices, `24` decoder action changes, and `10` stale-vs-fresh gate choice changes).

---

## Quick Start

### 1. Fast Offline Audit (NumPy Only, ~Seconds)

Verify the 8 visual fitting/selection cells, 500 frozen runtime choices, and 6 primary paired-result vectors directly from cached kernel distances without installing MuJoCo or MetaWorld:

```bash
python -m pip install 'numpy>=1.24,<3'
python reproduction/reproduce_offline.py --out outputs/offline_check
```

> Note: The output directory (`outputs/offline_check`) must not already exist. See [`reproduction/README.md`](reproduction/README.md) for scope details.

### 2. Core Package Installation & Unit Tests

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -e '.[dev,sim,plot]'
python -m pytest -q tests
python -m advice_audit.cli --help
```

### 3. MetaWorld & Vision Environments

MetaWorld and visual pipeline dependencies are separated to avoid version conflicts; install them in dedicated virtual environments:

```bash
# MetaWorld suite
python -m pip install -r requirements-metaworld.txt
python -m pytest -q tests_metaworld

# Vision suite
python -m pip install -r requirements-vision.txt
python -m pytest -q tests_vision
```

> Native experiment drivers require separately obtained demonstrations, actor checkpoints, and simulator dependencies. Always start each new run in a fresh output directory.

---

## Release Scope, Licenses & Citation

- **Release Scope**: Includes source code, 23 analyzed numerical records (`data/analyzed/manifest.json`), compact offline audit data, and license notices. Third-party demonstrations, pretrained network weights, full native trajectory archives, and manuscript submission files are not redistributed.
- **Licenses**: Original code is licensed under **MIT** ([`LICENSE`](LICENSE)); numerical datasets are licensed under **CC BY 4.0** ([`DATA_LICENSE.md`](DATA_LICENSE.md)); third-party notices are listed in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
- **Citation** (see [`CITATION.cff`](CITATION.cff)):

```bibtex
@software{wang_su_2026_delayed_advice_audit,
  author  = {Wang, Junxiang and Su, Jiaoyang},
  title   = {Delayed Advice Audit: Code and Analyzed Data},
  version = {1.0.1},
  year    = {2026},
  license = {MIT},
  url     = {https://github.com/w13579xw/Delayed-Advice-Audit}
}
```
