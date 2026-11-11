# Delayed Advice Audit — Compact Offline Audit

Junxiang Wang and Jiaoyang Su. Version 1.0.0. Archive citation: see the root CITATION.cff file

Install NumPy, then run `python reproduce_offline.py --out reproduction_result` from this directory. The output directory must not exist. The bundle reproduces eight visual fitting/selection cells, 500 frozen runtime choices and six primary paired-result vectors. It starts from cached normalized kernel distances and development labels derived from the original frozen features; it does not independently verify feature extraction or distance construction. It does not execute ACT, native simulation, later training seeds or channel interventions. The separate analyzed-data ZIP contains current manuscript/supplement results for those studies. No third-party pretrained neural-network weights or demonstrations are included.

Original code is MIT licensed; cached numerical data are CC BY 4.0. See LICENSE and LICENSE-DATA.md. All fitting, prediction and paired-outcome equality checks compare against the recorded results. Provenance paths and opaque identifiers are omitted from exported metadata; original records remain with the authors.
