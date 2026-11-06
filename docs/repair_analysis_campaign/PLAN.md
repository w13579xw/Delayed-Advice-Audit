# Repair/source failure audit: local campaign charter

## 1. Objective
Campaign id: repair-failure-20261102. Parent: outputs/repair_selector/*_20261102T112453Z. Selected outline: paper/measurement_outline.md, sections 3/6. User requests an honest publication claim and additional innovation. Question: why does a realized repair oracle fail to translate into learned expert-specific benefit? Main result is still being finalized; no test-driven refit is permitted.

## 2. Boundary and comparability
Keep task, original local, 80-step deadline, sources expert/P6/P12, five fixed repairs unchanged. Slice A reuses immutable held-out targets, purely descriptive posthoc. Slice B changes only unrevealed future disturbances at a fixed arrival state and is a NEW exploratory conditional-future pilot, not a new test of the frozen learner. No new model selection.

## 3. Slice plan
|exp_id|slice_id|tier|class|type|question|priority|placement|code change|extra baseline|
|---|---|---|---|---|---|---|---|---|---|
|R4|portfolio_regret|appendix|supporting|error analysis|How much portfolio opportunity survives the strong cheap family and how large is estimation regret?|1|appendix|helper only|none|
|R5|conditional_future|appendix|supporting|boundary|Does choosing with known future overstate conditional, cross-fitted repair opportunity?|2|appendix|new diagnostic|same cheap family|

Both map section_id=6, item_id=failure_attribution; claim_links=C3; paper_role=limitation. Exp ids and evidence paths are mirrored in paper/paper_experiment_matrix.md.

## 4. Hypotheses
A positive realized oracle does not guarantee a useful predictor. Expert portfolio increment can be small compared with prediction regret. R5 may show a lower cross-fitted value than the realized oracle; an absent gap would contradict that explanation. Monte Carlo selection error is a competing explanation; no claim of pure irreducible uncertainty or information-theoretic novelty.

## 5. Assets and dependencies
Available: frozen test data and models, Toy/MuJoCo snapshots and branch code. No external data, downloads, APIs or extra credentials. DeepScientist bash_exec/memory/artifact runtime is absent (tool discovery returned none); this local charter/log/output workflow is a documented fallback, not a claim of full runtime compliance.

## 6. Execution
A after parent completion; verify exact algebraic regret identity per row/model. B fresh seed74000–74003, request steps0/24, delays0/4/12, 16 futures (two disjoint folds of8), horizon80. Future shift time conditional on not having happened is uniform max(20,t+1)..59, direction isotropic magnitude.15. If shift already occurred, future is deterministic and reused. Current observed state and physical snapshot must stay identical. All options share each sampled future. Fixed full latent physical state/history gives more information than gate inputs. Smoke one episode/two futures on Toy before formal runs; Toy then Fetch sequentially. Abort on snapshot/paired identity failure. Expect minutes, monitor phase progress every <=60sec. No source edits during parent or slice execution.

## 7. Reporting
Report oracle, sample conditional maximum, cross-fitted selection and source portfolio increment; these are different estimands. 4 independent physical episodes are a pilot, not proof. Preserve null/negative cases. No tuning or new learner chosen on these outcomes. Explicit route after reporting: return to idea/task-source design if current method lacks benefit.

## 8. Checklist
CHECKLIST.md; next step: wait for parent completion and run R4.

## 9. Revision log
2026-11-02: frozen before either slice. R4 is explicitly posthoc, R5 prospective exploratory design. No changes to parent protocol.

Closeout: R4/R5 complete, all checks passed. Current learned-method claim contradicted; source/repair measurement distinction supported within reach. Future-unpredictability explanation unresolved/unsupported by this pilot. Route: return to idea/task-source design. Full report docs/REPAIR_VALUE_RESULTS.md.
