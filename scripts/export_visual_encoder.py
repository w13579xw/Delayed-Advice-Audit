"""Export an exactly matched frozen encoder from the pinned public ACT model."""
import argparse,hashlib,json
from pathlib import Path
from safetensors.torch import load_file,save_file
from torchvision.models import resnet18
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);a=p.parse_args();source=Path(a.checkpoint);root=Path(a.out)
if root.exists():p.error('new directory required')
sha=hashlib.sha256(source.read_bytes()).hexdigest()
if sha!='7f2472be5ecf68cf99b031583c79bbd2bf8981145a4fbcba9ba3c4f69cb70d04':p.error('Unexpected source checkpoint hash')
weights=load_file(str(source));prefix='model.backbone.';filtered={k[len(prefix):]:v for k,v in weights.items() if k.startswith(prefix)};encoder=IntermediateLayerGetter(resnet18(weights=None,norm_layer=FrozenBatchNorm2d),return_layers={'layer4':'feature_map'});encoder.load_state_dict(filtered,strict=True)
stats={name:weights[key].flatten().tolist() for name,key in [('image_mean','normalize_inputs.buffer_observation_images_base.mean'),('image_std','normalize_inputs.buffer_observation_images_base.std'),('robot_mean','normalize_inputs.buffer_observation_state.mean'),('robot_std','normalize_inputs.buffer_observation_state.std')]}
root.mkdir(parents=True);save_file(filtered,str(root/'encoder.safetensors'));(root/'normalization.json').write_text(json.dumps(stats,indent=2));manifest=dict(parent_checkpoint_sha256=sha,encoder_sha256=hashlib.sha256((root/'encoder.safetensors').read_bytes()).hexdigest(),strict_keys=len(filtered),architecture='ResNet18 FrozenBatchNorm2d, layer4,512x16x16',assets='frozen component, not a trained visual controller');(root/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
