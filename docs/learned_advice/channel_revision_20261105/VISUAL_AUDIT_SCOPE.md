# What the new visual audits do and do not verify
- Reconstruct each saved complete500reward mean from raw traces, verify finite actions/bounds/shapes and choice-to-outcome mapping.
- Recompute every frozen kernel choice/prediction from saved permitted inputs and packet; verify packet and frozen entry/model hashes.
- Independently reconstruct development-only normalization, distance scales, all grid validation MSEs and global tie selection. Training and validation are wholeepisode disjoint.
- Native success labels originate from native seen_success/last_success fields written at rollout time. Saved visual action/reward NPZs do not contain a complete native state/successflag history, so offline reward auditing does not independently rederive those labels. State K1 traces do save flags and permit that separate reconstruction.
- New native collection smoke repeats the complete reject tail and accepted-source/feature-export prediction parity. Original visual renderer/path testing additionally contains12complete tails with bitwiseimage/action/reward/state/success parity. Do not describe every formal visual option path as twicephysically rerun.
- Three actorseed systems share initialenvironment seeds and demo/developmentschedules. 300logicalarrivalcases are100independenttestenvironments; headroom subset60logicalcases are20independentinitialenvironments.
- Native fullsource100 competence is a separate reset trajectory, not the commonlocalprefix arrivalcomparison. Do not mix them into one pairedprefix interventionestimate.
