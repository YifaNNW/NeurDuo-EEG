"""CPU tests for alarm_lib. Run: python scripts/rebuttal/false_alarm/test_alarm_lib.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import alarm_lib as A


def B(s):
    return np.array([c == '1' for c in s])


def fold(y, score, rec, idx=None, sub=None):
    n = len(y)
    if idx is None:
        idx = np.concatenate([np.arange((np.asarray(rec) == r).sum()) for r in dict.fromkeys(rec)])
    return dict(y_true=np.asarray(y), y_score=np.asarray(score, float), recording=np.asarray(rec),
                epoch_idx=np.asarray(idx), subject=np.zeros(n, int) if sub is None else np.asarray(sub))


def test_runs():
    assert A.runs(B('0110011')) == [(1, 3), (5, 7)]
    assert A.runs(B('0000')) == []
    assert A.runs(B('1')) == [(0, 1)]


def test_merge_and_split():
    assert A.events(B('1001'), A.Rule(merge_gap=3)) == [(0, 4)]
    assert A.events(B('10001'), A.Rule(merge_gap=3)) == [(0, 1), (4, 5)]
    assert A.events(B('1' * 25), A.Rule(max_units=10)) == [(0, 10), (10, 20), (20, 25)]


def test_strict_scoring_and_delay():
    r = A.score_sequence(B('00110000'), B('00011010'))
    assert (r['n_seizures'], r['n_detected'], r['n_alarms'], r['n_false']) == (1, 1, 2, 1)
    assert r['delays'] == [60.0]                       # onset unit missed, second unit alarmed
    r = A.score_sequence(B('00110000'), B('01110000'))
    assert r['delays'] == [30.0] and r['n_already_on'] == 1 and r['n_false'] == 0


def test_tolerance():
    y = B('0001100000')
    before, after2, after3 = B('0010000000'), B('0000001000'), B('0000000100')
    for p in (before, after2, after3):
        r = A.score_sequence(y, p)
        assert (r['n_detected'], r['n_false']) == (0, 1)
    tol = A.Rule(pre_units=1, post_units=2)
    r = A.score_sequence(y, before, tol)
    assert (r['n_detected'], r['n_false'], r['delays']) == (1, 0, [0.0])
    assert A.score_sequence(y, after2, tol)['n_detected'] == 1
    assert A.score_sequence(y, after3, tol)['n_detected'] == 0


def test_recording_boundary():
    # recording 0 ends alarmed without a seizure, recording 1 starts with an alarmed seizure
    p = fold(y=[0, 0, 0, 1, 0, 0], score=[0, 0, 1, 1, 0, 0], rec=[0, 0, 0, 1, 1, 1])
    joined = A.score_fold(p, 0.5, per_recording=False)
    split = A.score_fold(p, 0.5, per_recording=True)
    assert (joined['n_alarms'], joined['n_false']) == (1, 0)
    assert (split['n_alarms'], split['n_false']) == (2, 1)
    assert split['n_detected'] == joined['n_detected'] == 1
    assert abs(split['hours'] - 6 * 30 / 3600) < 1e-12
    assert split['by_subject'][0]['n_false'] == 1


def test_missing_unit_breaks_an_alarm():
    p = fold(y=[0, 0, 0, 0], score=[1, 1, 1, 1], rec=[0, 0, 0, 0], idx=[0, 1, 3, 4])
    assert A.score_fold(p, 0.5)['n_alarms'] == 2


def test_threshold_and_rates():
    y = np.array([0, 1, 1, 1, 0, 1])
    s = np.array([.9, .1, .2, .4, .0, .8])
    thr = A.threshold_at_sensitivity(y, s, 0.5)
    assert abs(thr - 0.3) < 1e-12 and ((s[y == 1] >= thr).mean() == 0.5)
    p = fold(y=[0, 1, 1, 0], score=[1, 0, 1, 0], rec=[0, 0, 0, 0])
    t = A.pool([A.score_fold(p, 0.5), A.score_fold(p, 0.5)])
    r = A.rates(t)
    assert t['n_false'] == 2 and abs(r['fp_per_hour'] - 2 / (8 * 30 / 3600)) < 1e-9
    assert r['unit_sens'] == 0.5 and r['event_sens'] == 1.0 and r['delay_median_s'] == 60.0


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
        print('ok', t.__name__)
    print(len(tests), 'tests passed')
