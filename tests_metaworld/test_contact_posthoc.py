import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]/"scripts"))
from analyze_contact_posthoc import main, upper_positive_probability


def test_zero_discordance_does_not_imply_zero_population_benefit():
    assert np.isclose(upper_positive_probability(0, 20, .05), 1-.05**(1/20))
    assert upper_positive_probability(0, 20, .05/12) > .23
    assert upper_positive_probability(20, 20, .05) == 1


def test_posthoc_numpy_counts_serialize_and_pair_by_request(tmp_path, monkeypatch):
    parent, out = tmp_path/'parent', tmp_path/'out'
    parent.mkdir()
    (parent/'run.json').write_text(json.dumps(dict(status='complete')))
    (parent/'protocol.json').write_text(json.dumps(dict(config=dict(tasks=['push-v3'], test_episodes=1))))
    g = np.zeros((3, 3, 4));g[:, 0, 3] = [1, -.5, -1]
    np.savez(parent/'push-v3_test.npz', episode=np.array([1, 1, 1]), request_step=[24,24,24], delay=[0,4,12],
             return_gain=g, plan_id=np.full((3,3),'cached'))
    rows=[dict(seed=1, method=method, delay=delay, success=1) for method in ('reject','full_fixed','cheap_fixed') for delay in (0,4,12)]
    (parent/'push-v3_closed.json').write_text(json.dumps(rows))
    monkeypatch.setattr(sys, 'argv', ['audit','--parent',str(parent),'--out',str(out)])
    main()
    result=json.loads((out/'analysis.json').read_text())['tasks']['push-v3']
    assert result['raw_public12_dense_signflip_groups']==1
    assert result['paired_arrival_contrasts']['12-0']['mean']==-2
