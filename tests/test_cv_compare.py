import numpy as np
from scipy import stats

from ttc4900.cv_compare import holm, nadeau_bengio


def test_holm_hand_computed():
    # sıralı: 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> monotonluk: 0.06
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert holm([0.9, 0.8]) == [1.0, 1.0]  # 0.8*2=1.6 -> 1'e kırpılır, monotonluk


def test_nadeau_bengio_reduces_to_ttest_and_widens():
    d = np.array([0.01, 0.02, -0.005, 0.015, 0.008, 0.012])
    naive = nadeau_bengio(d, val_train_ratio=0.0)
    ref = stats.ttest_1samp(d, 0.0)
    assert np.isclose(naive["p"], ref.pvalue) and np.isclose(naive["t"], ref.statistic)
    nb = nadeau_bengio(d, val_train_ratio=0.25)
    assert nb["se_corrected"] > naive["se_corrected"] and nb["p"] > naive["p"]
    lo, hi = nb["ci95"]
    assert lo < d.mean() < hi
