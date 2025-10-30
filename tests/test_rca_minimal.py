
from resonant_learner import ResonantCallback

def test_rca_runs_and_stats():
    rca = ResonantCallback(verbose=False, patience_steps=2, min_delta=0.5)
    for v in [1.0, 0.8, 0.8, 0.79, 0.79]:
        rca(val_loss=v)
    stats = rca.get_statistics()
    assert isinstance(stats, dict)
    assert "beta" in stats and "omega" in stats
