import numpy as np
import pytest
from advice_audit.act_bridge import source_action_to_native, source_image


def test_source_convention_does_not_mutate_or_alias_inputs():
    source = np.array([[.2, -.3, 2., -2.], [-.9, .8, .1, .4]])
    original = source.copy()
    actual = source_action_to_native(source)
    np.testing.assert_array_equal(source, original)
    np.testing.assert_allclose(actual, [[-.2, -.3, 1., -1.], [.9, .8, .1, .4]])
    image = np.arange(512*512*3, dtype=np.uint8).reshape(512, 512, 3)
    flipped = source_image(image)
    np.testing.assert_array_equal(flipped[0], image[-1])
    assert flipped.flags.c_contiguous
    assert not np.shares_memory(flipped, image)


@pytest.mark.parametrize('bad', [np.zeros(3), np.array([0, 0, 0, np.nan])])
def test_source_nonfinite_or_wrong_action_rejected(bad):
    with pytest.raises(ValueError):
        source_action_to_native(bad)
