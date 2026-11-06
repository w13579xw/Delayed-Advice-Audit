# Critic learning-curve result

Completed recovery run: original collection preserved; mkdir-only recovery, original seeds and inference contract unchanged. All 200 fresh test environments, 658 distinct option tails, 305480 test native steps and independent replay/prediction/hash checks passed. Collection: 80 episodes, 407176 steps. All 24 fitted cells nonzero. No primary contrast has a strictly positive lower confidence bound.

|Task|Contrast|Mean|99.375% approximate interval|
|---|---|---:|---|
|push-v3|n28_total_vs_reject|0.013091|[-0.565647, 0.592656]|
|push-v3|n14_to_n28_data_increment|0.173083|[-0.494196, 0.861917]|
|push-v3|n28_packet_input_on_cheap|-0.011405|[-0.057025, 0.000000]|
|push-v3|n28_actuation_given_packet|0.395787|[-0.172843, 0.968289]|
|pick-place-v3|n28_total_vs_reject|-0.086780|[-0.332858, 0.154601]|
|pick-place-v3|n14_to_n28_data_increment|0.235327|[-0.063347, 0.567692]|
|pick-place-v3|n28_packet_input_on_cheap|-0.000165|[-0.121435, 0.128387]|
|pick-place-v3|n28_actuation_given_packet|0.050365|[-0.129690, 0.237744]|

One nested training draw with 12 fixed validation episodes/task; actor seed0 unchanged. Intervals do not establish equivalence, lack of benefit in general, or causal information value. n28-vs-reject is +.01309 Push and -.08678 PickPlace. No test-driven retuning or extension.

Figure v2 was rendered and visually inspected after fixing inherited style hex parsing: readable axes and legends, no overlaps; descriptive curves only. Internal review status, not publication certification.
