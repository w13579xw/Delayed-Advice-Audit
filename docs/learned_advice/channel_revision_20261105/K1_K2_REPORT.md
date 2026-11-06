# K1/K2 fixed-channel outcomes and finite-pool headroom

Status complete, v2offlineauditpassed; native cases and models unchanged. No model was retrained between channels. This estimates the declared source-refresh/erasure channel contrasts, not intrinsic/private expert information. F is a valid new source call on the arrival observation, but its joint input relation differs from S-only training. Erasure totalization is a predeclared wrapper, not a zero-vector input.

Native main:180independentcase snapshots, 728240 executedsteps,2280recordedbranchpaths. 10800decisionrecords checked (7200nonmissing decisions recomputed),360sourceprovenancechecks,60zero-delaypacketidentities; allrewardmeans exact andmodelsunchanged. The fivegateinitializations remainfrozen; no adversecase removed.

|Task|Contrast|Mean|99.5833% approximate interval|Any-success difference|
|---|---|---:|---|---:|
|drawer-open-v3|refresh_minus_stale_cheap|+0.000000|[+0.000000,+0.000000]|+0.0000|
|drawer-open-v3|refresh_minus_stale_full|+0.037667|[-0.023193,+0.207637]|+0.0000|
|drawer-open-v3|stale_minus_erasure_full|-0.075647|[-0.251302,-0.011677]|+0.0000|
|drawer-open-v3|refresh_minus_erasure_full|-0.037980|[-0.072814,-0.012675]|+0.0000|
|door-open-v3|refresh_minus_stale_cheap|-0.000047|[-0.063529,+0.063522]|+0.0000|
|door-open-v3|refresh_minus_stale_full|+0.038542|[-0.101599,+0.207933]|+0.0000|
|door-open-v3|stale_minus_erasure_full|-0.102586|[-0.290619,-0.000380]|+0.0000|
|door-open-v3|refresh_minus_erasure_full|-0.064043|[-0.214783,+0.000253]|+0.0000|
|button-press-v3|refresh_minus_stale_cheap|+0.000000|[+0.000000,+0.000000]|+0.0000|
|button-press-v3|refresh_minus_stale_full|+0.000000|[+0.000000,+0.000000]|+0.0000|
|button-press-v3|stale_minus_erasure_full|+0.000000|[+0.000000,+0.000000]|+0.0000|
|button-press-v3|refresh_minus_erasure_full|+0.000000|[+0.000000,+0.000000]|+0.0000|

No positive refresh effect is established. Availability S−E is negative on Drawer/Door underapproximateintervals; F−EnegativeonDrawer,unresolvedonDoor. AllAny-success differences0; not a finite-sampleequivalence claim. Buttonzero/rejectchannelidentity structural.

Finitepooloracleheadroom is tiny on average: stale fullpool Drawer.006573,Door.000727,Button.000262; source-exclusive.001501,.000419,.000248. Expert improvements above1e-8 occur in55%,3.3%,60%respectively but zero cases have sourcegain>.25. Tinygain signs mustnotbeinterpretedaslargeusefulopportunities. Frozenfullpolicymeanregret .082221,.103312,.000262. Regretidentitychecked:Uchosen−UR=(Uoracle−UR)−(Uoracle−Uchosen). Offlineupperboundonly, notdeployable performance.

S sourcepositive precision/recall:Drawer24.1%/25.5%,Door2.45%/50%,Buttonundefined/0. Negative gains are dominated bysourceplans in thisnewstatecohort, whereas originalvisualcasewasdominatedbylocalalternatives. Distinct mechanisms shouldnotbe pooled. DetailedsourcepredictionMSE/choiceclasses/delay,q,prefixsuccessdescriptives retained.

FirstanalyzerfailedonlyinNumPybooldiagnosticserialization. v1partial andfailedstatepreserved;v2castsboolandrerunsidentical12contrasts/RNG/rawcasefiles. No native data rerun, new test or fitting change.
