import pytest
import torch
from advice_audit.vision import VisualHead

def test_visual_head_rejects_native_object_goal_observation():
    model=VisualHead(channels=4,hidden=16)
    features=torch.zeros((2,512,16,16))
    assert model(features,torch.zeros((2,4))).shape==(2,4)
    with pytest.raises(ValueError,match='robot4D'):
        model(features,torch.zeros((2,39)))
