# Further learned-source feasibility

Public metadata was checked for RoboHP/metaworld_mt50_act, revision680a3b396860b88adcf4036a735fe124b4d850da. The inference safetensors file is206684616 bytes; optimizer state is not needed for evaluation. Model card states MIT and uses corner4, with image/action flipping requirements and evaluation code marked WIP. These are author-provided properties, not our reproduced results.

No weights were downloaded and no PyTorch/VLA model was installed in this turn. Metadata alone does not establish compatibility with our pinned Meta-World3.1.1. Next-stage use requires checking the original evaluation implementation, observation/state preprocessing, action orientation, policy queue/history, renderer and standalone clean competence before delayed-action comparisons. Large pi0.5 metadata also exists but is4B FP32 with no model card; it is not a qualified baseline here.

Primary source https://huggingface.co/RoboHP/metaworld_mt50_act . Saved metadata outputs/contact/checkpoint_feasibility/RoboHP_metadata.json. Do not substitute advertised success rates for our experiments.
