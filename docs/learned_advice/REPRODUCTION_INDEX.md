# Local reproduction and evidence index

This index points to already executed artifacts. It is not a claim of a clean-machine reproduction, a new training replicate or an official MetaWorld benchmark submission. Run from the repository root and preserve every old output. All long experiments use one simulator job at a time, native500, MUJOCO_GL=egl and OMP/BLAS thread caps. Never use an unsupported --resume flag or overwrite a completed run.

|Evidence stage|Environment and entry point|Accepted artifact|
|---|---|---|
|Native simulation and regression|.venv for Toy/Fetch; .venv-metaworld for native contact|server logs, original59base/31contact tests, subsequent numeric2tests|
|Frozen original ACT source|.venv-act CPU LeRobot0.3.3; act_chunk_worker.py|learned_source_intake/act_checkpoint, pinned HF680a3b3 revision andSHA7f2472be...|
|GPU encoder and visual heads|.venv-vision Python3.10 Torch2.4.1+cu118; train_visual_heads.py|encoder_assets_20261103,features_200x2_20261103_v1,head_training_push/pick_20261103_v1|
|Explicit standalone ACT core parity|act_adaptation.py plus licensedact_core.py|act_parity_20261103_v2,notfullLeRobotGPUcompatibility|
|Task-matched3k source training|train_matched_act.py,unchanged180/20demos|act_matched_push/pick_20261103_v1|
|Frozen20episode source-development|qualify_matched_act.py|matched_source_development_20261103_v1,240episodes,slices1/20/100|
|Old nativeMSAA4qualification|evaluate_visual_heads.py plusqualify_matched_act.py|matched_qualification_20261103_v1,50paired/task,chosenvariablecadences|
|ExplicitMSAA0visualreplay|audit_visual_replay.py andDeterministicCamera|visual_replay_msaa0_20261103_v2,12full-tailcases|
|15kbudget/samecadence100test|run_act_budget_control_20261103.py storedinoutputs|act_budget_control_20261103_v1,100paired/task,4configs;commands.json capturesexactexecution|
|Single-requestdecisiondevelopment|collect_visual_value_data.py|value_data_40_20261103_v1,40uniqueepisodes,12relatedoptions,14train6validation/task|
|Valuelearnerfeasibility|output-onlytrain_value_kernel_pilot_v3_20261104.py|value_kernel_pilot_20261104_v3,frozen,noprospectivetestclaim|
|Independentconfirmation|confirm_visual_arrival.py|arrival_confirmation_20261104_v1,complete500episodes; analysis/audit passed|

Read each run's protocol.json/run.json/checks.json before its metrics. Main source/checkpoint/split choices are recorded in their freeze files and source snapshots. Source fingerprints include src/**/*.py, scripts/*.py, tests/*.py and pyproject.toml. Do not edit those while an active fingerprinted job runs. Output-only orchestration/analysis scripts have their own snapshots and must be read with the relevant run. Current managed branch names do not replace these uncommitted source snapshots as provenance.

The GPU worker environment deliberately has no installed simulator. Native MetaWorld3.1.1 uses MuJoCo3.3.0 in its separate Python3.12env; old Fetch usesMuJoCo3.3.7. The licensed ACT core is a pinned standalone adapter with strict51,593,092parameter load and separate CPU/GPU/FP16feature-path parity, not an unsupported edit to an installed upstream library. Requirements and per-run package freezes reside under outputs. The original model uses RGBcorner4plusrobot4 and a blackwristplaceholder, no languageconditioning. Source x-actionflip and verticalimageflip are explicitly verified, not silently inferred.

The26.2GBfeaturedataset replays200000native observations exactly across400fullteacher episodes. Features use image-normalization withoutEPS thenFP16quantization; this approximation is separately audited. Traininglabels can include trueobject/goalpose but deployed actors/critics receive only RGB-derivedmaps androbot4. All180/20splits are wholeepisodes. Cheapheads use20epochs; ACT budgets3k/15kupdates have a different loss and target100futureactions. Samecorpus is not complete equalizationofoptimization/pretraining/computebudgets.

No submitted dataset or finished output is overwritten. A new reproduction should create unique run directories and rebuild configs with its own repository paths. Some archived artifacts contain absolute local model/assets paths, so copying an artifact directory alone is not a portable clean-room install. The scripts/configs and command logs provide an auditable recipe; an independent clean-machine package remains a separate validation task.

Paper-facing use must distinguish nativeMSAA4historic qualifications from newMSAA0counterfactual studies, conditionalgrossarrival utility from netacquisitionvalue, sourcebenchmark competence from marginaladvicevalue, and modeltraining seeds from environmentepisode pairs. The old paper draft is internal; current conclusions are at CURRENT_STATUS.md, review boundaries at outputs/novelty_review_20261104/SYNTHESIS.md, and the full run journal at outputs/server/HANDOFF_STATUS.md.


## Completed offline paper reproduction

`paper/visual_advice_audit/reproduce_offline.py --out <new-directory>` uses NumPy
and the standard library, reads paths relative to its own file, and imports no
simulator or Torch. With the supplied local1151-file152MBasset bundle, a run
from /tmp reproduced all96validation grid entries, eightselected models,
500decisions and six paired vectors with exactly zero error in4.99seconds.
Accepted artifact: outputs/vision/offline_paper_reproduction_20261104_v1.
The assets are excluded from Git and not externally released; the script alone
is insufficient without them. This is not a native-physics/source-inference
rerun or clean-machine replication. Exact source identities, split IDs and
software versions are in paper/visual_advice_audit/REPRODUCIBILITY_SUPPLEMENT.md.


## Completed supplementary replication and workspace relocation

The full repository, environments, datasets, checkpoints, outputs and manuscripts now reside at `/home/ez/DACI/code`. The old project-local repository path is a compatibility symlink. Frozen absolute paths and hashes were not rewritten. See `outputs/server/migration_20261104.json` and `WORKSPACE_LAYOUT.md`.

|Evidence|Entry point under outputs/vision|Completed artifact|
|---|---|---|
|Three-seed3k/local competence|run_training_replication_20261104.py|training_replication_20261104_v1;8newtrainingruns,600trajectories,100independent environments|
|Independent trace/model/split audit|audit_training_replication_20261104.py|training_replication_audit_20261104_v1;rawreward and aggregate error0|
|Development split/tolerance diagnostic|analyze_development_resplits_20261104.py|development_resplits_20261104_v1;200resplits/4000selectionconfigs,developmentonly|
|Observed per-call/episode costs|report_replication_cost_20261104.py|replication_cost_20261104_v1;all151500calls|
|Post-migration path-independent offline verification|paper/visual_advice_audit/reproduce_offline.py from/tmp|offline_migration_reproduction_20261104_v1;5.136s,500choices/8selections/6vectors equal|

All listed runs completed with exit0 and checks passed. The orchestration and audit scripts reject an existing output directory. Do not rerun them into completed paths or invent a resume flag. For a new independent reproduction, create a distinct run configuration/output and retain the original provenance. The3k competence supplement does not establish training-randomness robustness for the15k source or arrival policies. Journal selection is undecided; no MDPI conversion or external submission has been performed.


## Latest completed evidence and JAAMAS draft (2026-11-05 CST)
V11criticcurveallcomplete,80development/200test,8primarycontrastsnonestrictlypositive. J1–J4newstatesuitecomplete:3tasks/5criticseeds/126development/360test/18000choices; allaudits passed. Fullreport docs/learned_advice/jaamas_campaign_20261105/REPORT.md. Currentpaper /home/ez/DACI/Springer/DACI_JAAMAS/DACI_JAAMAS_manuscript.pdf (23pages), editablemain.tex,2pageinformation sheet,1pageletter,AUTHOR_CHECKLIST. TargetJAAMAS,notMDPI/Robotica. Completevisual15k/fullarrivalactorseedreplicationstillunrun; no realrobot/async/publicrelease/authorapproval/submissionclaim.
