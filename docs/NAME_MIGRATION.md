# Descriptive-name release — 2026-11-06

| Item | Current name |
|---|---|
| Article | Auditing the Value of Delayed Advice in Autonomous Agents |
| Project | Delayed Advice Audit |
| Repository target | w13579xw/Delayed-Advice-Audit |
| Distribution | delayed-advice-audit |
| Import | advice_audit |
| Console | advice-audit |

This is a naming and packaging release. No experimental threshold, model, task, selection criterion or numerical result was changed. The frozen experimental source fingerprint remains `48ffd6ffff1e3eeb5cc15a475888a844c846ead878292e6137c7f171c5246221`; current source has a separate fingerprint because names changed.

## Compatibility and intentional historical identifiers

The `daci` import redirects to the same canonical module objects. Old CLI entry point and method IDs map to the new implementations. Old schema hashes are verified before comparison through the explicit schema compatibility mapping; frozen files are not rewritten. `DACI_PYTHON` is a fallback environment variable. Old physical paths and PDF names are compatibility symlinks.

Completed logs, datasets, weights, source_data manifests, dated protocol snapshots, old manuscripts and Git history retain their original identifiers. Rewriting them would break provenance. Current active code, paper prose, output names and README use descriptive names; compatibility tests and this migration record deliberately mention old names.

## Verification

Base/Fetch: 65 passed; MetaWorld: 31 passed; vision: 1 passed. Deterministic toy, Fetch, Push and PickPlace arrays match the pre-migration snapshot bit for bit. Four rebuilt PDFs and the editable source package are checked separately under `outputs/renaming/20261106/`. Historical scientific results are unchanged.

## GitHub

The user renamed the repository, verified through stable repository ID 1442458398. Current repository: [w13579xw/Delayed-Advice-Audit](https://github.com/w13579xw/Delayed-Advice-Audit). Visibility remains private. The About description is:

> Delayed Advice Audit: simulation-based evaluation of pipeline performance, message dependence, and the value of delayed expert advice in autonomous agents.

The About description was saved and verified in the authenticated GitHub browser. Local `origin` now uses the canonical repository URL. Current code and manuscript export are synchronized; see [release report](name_migration_20261106/REPORT.md) and [sync record](name_migration_20261106/GITHUB_SYNC.json). Original timeout and pre-rename states are retained only in dated execution logs, not as outstanding tasks.
