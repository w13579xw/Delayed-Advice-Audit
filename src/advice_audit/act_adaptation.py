"""Explicit standalone ACT adapter; frozen visual features and robot4D only.

The core is verbatim pinned LeRobot code, with licensed standalone imports.
No installed third-party code is patched. Original CPU LeRobot remains intact.
"""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file
from .act_core import ACT


def core_config(raw):
    values=dict(raw);values['pretrained_backbone_weights']=None
    values['robot_state_feature']=SimpleNamespace(shape=[4])
    values['action_feature']=SimpleNamespace(shape=[4])
    values['env_state_feature']=None
    values['image_features']={key:SimpleNamespace(shape=value['shape']) for key,value in raw['input_features'].items() if key.startswith('observation.images.')}
    assert len(values['image_features'])==2 and raw['chunk_size']==100
    return SimpleNamespace(**values)


def checkpoint_statistics(weights):
    result={}
    for short,key in [('image','normalize_inputs.buffer_observation_images_base'),('wrist','normalize_inputs.buffer_observation_images_wrist'),('robot','normalize_inputs.buffer_observation_state'),('target_action','normalize_targets.buffer_action'),('output_action','unnormalize_outputs.buffer_action')]:
        for moment in ('mean','std'):result[short+'_'+moment]=weights[key+'.'+moment].flatten().tolist()
    return result


class FixedGridPosition(nn.Module):
    """Original CPU positional map for the fixed16x16feature grid.

    Its values depend on shape only. Precomputation avoids Torch2.4CUDA's
    unsupported deterministic cumsum without disabling deterministic checks.
    """
    def __init__(self, original):
        super().__init__()
        with torch.no_grad():
            embedding=original(torch.empty((1,512,16,16),device='cpu'))
        self.register_buffer('embedding',embedding,persistent=False)

    def forward(self, value):
        if tuple(value.shape[-2:])!=(16,16):raise ValueError('Fixed positional grid requires16x16')
        return self.embedding


class AdaptableACT(nn.Module):
    def __init__(self, raw_config, weights, statistics, device='cuda'):
        super().__init__()
        self.raw_config=dict(raw_config);self.statistics=dict(statistics)
        self.core=ACT(core_config(raw_config))
        self.core.load_state_dict(weights,strict=True)
        self.core.encoder_cam_feat_pos_embed=FixedGridPosition(self.core.encoder_cam_feat_pos_embed)
        self.core.backbone.requires_grad_(False)
        for key,values in statistics.items():
            shape=(1,3,1,1) if key.startswith(('image_','wrist_')) else (4,)
            self.register_buffer(key,torch.tensor(values,dtype=torch.float32).reshape(shape))
        self.register_buffer('source_sign',torch.tensor([-1.,1.,1.,1.]))
        self.register_buffer('wrist_feature',None,persistent=False)
        self._raw_backbone_forward=self.core.backbone.forward
        self._feature_iterator=None
        self.core.backbone.forward=self._backbone_forward
        self.to(device)

    def _backbone_forward(self,image):
        if self._feature_iterator is None:return self._raw_backbone_forward(image)
        return {'feature_map':next(self._feature_iterator)}

    @property
    def device(self):return self.robot_mean.device

    def _robot(self,robot):
        value=torch.as_tensor(robot,device=self.device,dtype=torch.float32)
        if value.ndim==1:value=value[None]
        if value.ndim!=2 or value.shape[1]!=4 or not torch.isfinite(value).all():raise ValueError('Only finite robot4D may enter ACT')
        return (value-self.robot_mean)/(self.robot_std+1e-8)

    def _rgb(self,images):
        if isinstance(images,np.ndarray):images=torch.from_numpy(np.ascontiguousarray(images)).to(self.device)
        if images.ndim==3:images=images[None]
        if images.dtype!=torch.uint8 or tuple(images.shape[1:])!=(512,512,3):raise ValueError('Need Bx512x512x3uint8 RGB')
        return images.permute(0,3,1,2).float()/255.

    def normalize_targets(self,native_actions):
        return (native_actions*self.source_sign-self.target_action_mean)/(self.target_action_std+1e-8)

    def unnormalize(self,actions):
        return actions*self.output_action_std+self.output_action_mean

    def raw_output(self,robot,images):
        """FP32raw-input path for original-checkpoint implementation parity."""
        rgb=self._rgb(images);wrist=torch.zeros_like(rgb)
        batch={'observation.state':self._robot(robot),'observation.images':[(rgb-self.image_mean)/(self.image_std+1e-8),(wrist-self.wrist_mean)/(self.wrist_std+1e-8)]}
        prediction,_=self.core(batch)
        return self.unnormalize(prediction)

    def cached_output(self,robot,features,actions=None,padding=None):
        """Normalized predictions/VAE parameters for train or eval.

        Future actions/padding are training supervision; the core consumes them
        only in training mode. Deployment supplies neither.
        """
        robot=self._robot(robot);batch_size=len(robot)
        if tuple(features.shape)!=(batch_size,512,16,16):raise ValueError('Bad visual feature shape')
        if self.wrist_feature is None:
            with torch.inference_mode(False), torch.no_grad():
                black=torch.zeros((1,3,512,512),device=self.device)
                self.wrist_feature=self._raw_backbone_forward((black-self.wrist_mean)/(self.wrist_std+1e-8))['feature_map'].detach()
        dummy=torch.empty((batch_size,3,1,1),device=self.device)
        batch={'observation.state':robot,'observation.images':[dummy,dummy]}
        if actions is not None:
            assert tuple(actions.shape)==(batch_size,100,4) and tuple(padding.shape)==(batch_size,100)
            batch.update(action=actions,action_is_pad=padding)
        self._feature_iterator=iter([features,self.wrist_feature.expand(batch_size,-1,-1,-1)])
        try:result=self.core(batch)
        finally:self._feature_iterator=None
        return result

    def cached_image_output(self,robot,images):
        rgb=self._rgb(images)
        # Match the frozen visual-baseline feature data definition, including
        # its recorded float16 feature quantization (not bitwise legacy ACT).
        feature=self._raw_backbone_forward((rgb-self.image_mean)/self.image_std)['feature_map'].half().float()
        prediction,_=self.cached_output(robot,feature)
        return self.unnormalize(prediction)


def original_adapter(checkpoint,device='cuda'):
    root=Path(checkpoint);weights=load_file(str(root/'model.safetensors'))
    state={key[len('model.'):]:value for key,value in weights.items() if key.startswith('model.')}
    return AdaptableACT(json.loads((root/'config.json').read_text()),state,checkpoint_statistics(weights),device)


def adapted_adapter(checkpoint,device='cuda'):
    root=Path(checkpoint)
    return AdaptableACT(json.loads((root/'source_config.json').read_text()),load_file(str(root/'core.safetensors')),json.loads((root/'normalization.json').read_text()),device)
