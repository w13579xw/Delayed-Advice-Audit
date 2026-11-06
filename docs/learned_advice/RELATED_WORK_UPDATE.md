# Novelty check — 2026-11-03

The old manuscript is an internal empirical draft. New methods must be judged against current primary sources, not only the earlier reviewer packet.

- Black et al., [Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/abs/2506.07339), submitted2025-06-09: inference-time prefix freezing/inpainting of diffusion/flow chunks during asynchronous control. Simple overlap/alignment is not new.
- [REMAC](https://hatchetproject.github.io/remac-async/) / ICLR2026: masked chunk training addresses perception/execution mismatch inside a chunk. A delay-aware training mask alone is not a new contribution here.
- [FutureRTC](https://arxiv.org/abs/2607.24008), submitted2026-07-27: predicts execution-time contexts and corrects proprioceptive discrepancies. A rolled-forward observation alone is not a new contribution.
- Zhu et al., [PACT-VLA](https://arxiv.org/abs/2610.28019), submitted2026-10-20: newer pre-commit observations choose among complete stale-observation candidate chunks through frozen-model FlowRes scoring. Broad arrival-aware candidate verification/selection is already addressed. Its advertised gains are prior-work claims, not independently reproduced by this project.
- [LeRobot remote inference](https://huggingface.co/docs/lerobot/main/remote_inference): ordinary chunk cursor alignment and blending are standard implementation choices, not evidence of novelty.

Potential narrower distinction requiring experiments: decision value is relative to the best *deployable* cheap alternative under equal requests/compute, rather than action plausibility or a hindsight cheap union. A proposal would need to show that this distinction changes actual decisions and yields held-out utility/success gains after including cheap repairs, stalls, and strong local control. Existing tanh regressors failed that gate. No new neural architecture or publication readiness is claimed.

The public [ACT checkpoint](https://huggingface.co/RoboHP/metaworld_mt50_act) is MIT and explicitly warns that image/action flips must match its dataset; evaluation code is marked WIP and linked fork404. Strict checkpoint load and source competence qualify infrastructure only. It is not a contribution by itself.
