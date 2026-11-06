# Full-text novelty correction: fixed-pool branching already exists

Verified during confirmation, 2026-11-04. This updates the contribution boundary;
it changes no frozen models, episodes, primary effects or analysis rules.

PACT-VLA, arXiv:2610.28019v1, Appendix D.2, already evaluates selectors on fixed
candidate pools. Choices precede branch rollout, each candidate starts from the
same saved state, and a common continuation controller finishes each branch.
The paper distinguishes this single-decision estimand from repeatedly applying
a selector. Source: [verified primary full text](https://arxiv.org/html/2610.28019v1).

Consequently, fixed-packet branching, prospective selection and common
continuation are not standalone first contributions here. Broad use of newer
observations to choose an action chunk is also covered. The current ACT/RBF
study does not implement FlowRes or compare its performance; ACT's CVAE is not
the frozen flow-policy interface used by that method.

The remaining candidate contribution is empirical: a bounded study against
executable local alternatives, with packet-representation and action-set
variants, explicit null-model interpretation, local renderer/replay auditing,
and prospective practical-effect precision. Those components do not establish
a new neural method, and their combination alone is not proof of publication
significance. The complete confirmation must determine the actual finding.

Earlier partial-evidence review reports remain immutable. This new source check
is a root-agent addition after those reports, not an amendment to reviewer
opinions or an external peer-review decision.
