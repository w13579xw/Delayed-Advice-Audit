# Literature delta and claim boundary, 2026-11-04

This document records retrieved primary sources, not an endorsement of a new method. Reused survey coverage is in RELATED_WORK_UPDATE.md (RTC,REMAC,FutureRTC,PACT-VLA,LeRobotcursor alignment). New search specifically asked whether returned expert output can serve as an input without executing the recommended action.

## Newly retrieved comparisons

1. Pugnana et al. (2025), *To Ask or Not to Ask: Learning to Require Human Feedback*, arXiv2510.08314, https://arxiv.org/abs/2510.08314 . LearningtoAsk explicitly augments a predictor with expertfeedback and studiesbudgeted querying. Thus expert-output-as-feature is already established, not our newtheory. Primaryabstract and author-paper text retrieved.
2. Alur et al. (2024), *Integrating Expert Judgment and Algorithmic Decision Making: An Indistinguishability Framework*, arXiv2410.08783v2, https://arxiv.org/html/2410.08783v2 . Feasible-predictor classes and expert-sideinformation offer a principled antecedent for measuringcomplementarity. OurACT receives thesameencodedimage as cheapmodels, so no extraShannoninformation is established.
3. Shihab et al. (2026), *When Is Language-Model Advice Worth Buying? A Matched-Budget Study of Predictive Gating in Reinforcement Learning*, arXiv2609.29548v2, https://arxiv.org/html/2609.29548v2 . This is the October16revisedtitle; discoverysearch also returned the older *Certified Predictive Value-of-Advice Gating...* title. Response-contingentvalue/cost decisions and matchedquerybudget are already covered. The present work cannot claim theirinvention.
4. Watkins et al. (2021/updated2023arXiv), *Teachable Reinforcement Learning via Advice Distillation*, https://papers.nips.cc/paper/2021/file/37cfff3c04f95b22bcf166df586cd7a9-Paper.pdf and arXiv2203.11197 . Advice-conditionedpolicies followed bydistillation establish that advice neednot beexecutedverbatim. Distillation alone isnot noveltyhere.
5. Whitmeyer (2022/updated2026), *Making Information More Valuable*, arXiv2210.04418v6, https://arxiv.org/html/2210.04418v6 . Generaldecision-theoreticbackground for interactionof actionsets and information. Ourfinite-arm utilitydifferences are elementarydecompositions, not newinformationtheory.
6. Rosman andRamamoorthy (2014), *Giving Advice to Agents with Hidden Goals*, https://rad.inf.ed.ac.uk/data/publications/2014/icra14rosman.pdf . Advice-guidedlocalpolicyselection is established. Nativecontinuouscontrol with delayedfixedpackets remains an empiricalsetting distinction, notproof ofpriority.

## Reused directly relevant mechanisms

7. Black et al. (2025), *Real-Time Execution of Action Chunking Flow Policies*, arXiv2506.07339 . Prefixalignment/inpainting forasyncchunks.
8. Zhu et al. (2026), *PACT-VLA*, arXiv2610.28019 . Currentobservations verify/selectwholeprecommitcandidatechunks. Broadarrival-awarechunkselection isnotnew. Itsflowresidualscorer isnot reproduced by thisACTCVAE experiment.
9. *FutureRTC*, arXiv2607.24008 . Executiontimecontext/proprioceptivecorrection overlaps proposedphase-alignment repairs.

## Current bounded verdict

A plainarrivalvaluegate,expertoutputasfeature,conformalcalibration,ordelay-awareprefixtrimming is insufficiently differentiated as a standalonealgorithm claim. The candidate empirical distinction is a2×2controlled comparison of computationalfeatures and permittedactionsets underdelayednativeinterventions, withstrongsensory-matchedcheaprepairs and source-trainingbudgetcontrols. Sourcepacket is a deterministicfunctionofrequestvisualfeatures/robot4, so its value isconditionalon thefinite learnedcritic/computeclass. NoBayes-levelinformationgain is implied. The6episodevalidationpilot doesnot establish a publishableeffect. Current resultsdonot justify a newtheorem, a newarchitecture or a generalVLAperformanceclaim.

Unresolved work includes directfairperformancecomparisons to recentasyncmethods on compatiblemodel families, actualmultiqueryasynchronous execution, source/cheaptrainingseedreplication, broader contacttasks, and whether any strengthenedselector has independentpositivevalue. These are missingevidence boundaries, notcompleted validations.
