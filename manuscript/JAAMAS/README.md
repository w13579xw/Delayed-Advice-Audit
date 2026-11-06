# Delayed Advice Audit — JAAMAS manuscript

Title: **Auditing the Value of Delayed Advice in Autonomous Agents**. Status: ready with author checks; not submitted.

The main article has 28 pages, four figures, eight tables, one algorithm and 31 references. Online Resource 1 has 19 pages. Information sheet and cover letter are drafts of two and one pages. The paper separates pipeline gain, runtime message dependence and predefined fixed-policy channel value. It does not establish a generally beneficial delayed-advice controller.

Run `bash build.sh` to compile all four PDFs with pdflatex. On the original local workspace, an unprivileged TeX fallback is used when available; other machines need a standard TeX installation with the supplied Springer class. The authoring directory is `/home/ez/DelayedAdviceAudit/Springer/Advice_Audit_JAAMAS`. The repository `manuscript/JAAMAS/` is a release export.

Active output names are `Delayed_Advice_Audit_manuscript.pdf` and `supplement/Advice_Audit_Online_Resource_1.pdf`. FINAL_CHECKS.json records current hashes and checks. Source data contains 22 immutable analyzed-file copies with its original manifest. Dated reproduction entries and original fingerprints retain historical identifiers for provenance and require the full workspace, pinned environments, data and weights to regenerate native experiments.

The independent decoder study preserves the original channel experiment. For the predefined fixed 48-step fresh chunk, generation-matched decoding improves reward over legacy decoding by +0.579061 (98.75% interval [+0.172635,+0.997332]); the refresh-versus-stale and gate comparisons do not establish positive benefit. Three full visual training systems share demonstrations, pretraining and development arrangements; they measure conditional training variability. No real hardware, live asynchronous service, human teacher or private-information guarantee is claimed.

Author identity, affiliations, funding/conflict statements, publication/overlap permissions and final approval remain author checks. No journal submission or public release has been performed.

Publication-grade figure refinement is recommended with AutoFigure-Edit (open-source: https://github.com/ResearAI/AutoFigure-Edit; online service: https://deepscientist).
