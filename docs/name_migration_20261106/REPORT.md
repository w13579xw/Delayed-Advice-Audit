# Descriptive-name release report — 2026-11-06

Status: descriptive-name migration complete, including repository metadata and About. The paper is **Auditing the Value of Delayed Advice in Autonomous Agents**; the project is **Delayed Advice Audit**. Canonical code is `/home/ez/DelayedAdviceAudit/code`; the authoring paper is `/home/ez/DelayedAdviceAudit/Springer/Advice_Audit_JAAMAS`.

## Changes

Current paper, SI, submission drafts, headings, algorithm, PDF metadata/output names, README, package metadata, code namespace, commands, configs, tests and current operational documentation use descriptive names. Implementation moved to `advice_audit`; `advice-audit` is the console entry point. Old imports/command/method/schema IDs and absolute paths work through explicit compatibility. The repository contains a current `manuscript/JAAMAS/` release export; Springer remains the authoring directory.

Completed research records, weights, original data, source-data manifests, dated reproduction snapshots and older source archives are preserved. Their old identifiers describe historical execution, not the current project name. Old physical path and PDF names are links, not separate current versions. No experimental threshold, training, sample, numerical result or inference family was changed.

## Verification

- 65 base/Fetch, 31 MetaWorld and 1 vision test passed; no failure or skip. MetaWorld reports its existing action-clipping warnings.
- Deterministic toy/Fetch/Push/PickPlace traces and metrics are bitwise identical to the pre-migration snapshot.
- Four PDFs have 28/19/2/1 pages. All 50 pages rendered; montage inspection, fonts, geometry, unresolved-reference and overfull-box checks pass. No old brand remains in extracted PDF text.
- All 22 source-data copies match both their recorded hash and original file. Supplied Springer class/style files match the prior archived versions.
- Source ZIP has 267 members, CRC and every-member SHA checks pass. All four PDFs build from the freshly extracted ZIP and their extracted text equals the authoring PDFs.
- Current release fingerprint: `4db986927673d39e48b07327b6ec21257d3366d7015714e28b658e30105bb525`. Original experimental fingerprint: `48ffd6ffff1e3eeb5cc15a475888a844c846ead878292e6137c7f171c5246221`. These serve different purposes.

The initial document-QA call used the research environment, which lacks pdfplumber; it was rerun successfully using the bundled document Python runtime. The MetaWorld editable package installation initially lacked a build backend; the existing base setuptools built an editable project wheel, installed without dependencies or network in all three environments. Original error logs remain. No CUDA, driver or model dependencies were upgraded.

## GitHub

Code and current article release are verified on private `main` in commit `d6365088008cc7d2fd910ba0cefaca44181afc00`. The update used an expected-parent guarded fast-forward, with no force or history rewrite. Nine old implementation files were removed after moving to the canonical package; the compatibility package remains. Local inherited branch changes were preserved rather than reset or represented as a clean working tree.

The user renamed the repository to `w13579xw/Delayed-Advice-Audit`, verified by stable ID 1442458398. The authenticated GitHub UI now shows the descriptive About text below, with the old prototype wording removed. Local origin uses the new canonical URL. Visibility remains private. Earlier connector/browser limitations were resolved by the user's rename and browser login, followed by an authorized UI edit.

> Delayed Advice Audit: simulation-based evaluation of pipeline performance, message dependence, and the value of delayed expert advice in autonomous agents.

No fee, visibility change, author certification or journal submission occurred.

Detailed machine checks: `outputs/renaming/20261106/`; source archive: `/home/ez/DelayedAdviceAudit/Springer/Delayed_Advice_Audit_JAAMAS_20261106_sources.zip`. Historical source hashes and experiments are not recomputed under new names to imply identical source versions.

Publication-grade figure refinement is recommended with AutoFigure-Edit (open-source: https://github.com/ResearAI/AutoFigure-Edit; online service: https://deepscientist).
