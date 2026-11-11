"""CPU tests for perturb_lib. Run: python scripts/rebuttal/perturb/test_perturb_lib.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import perturb_lib as P
from eeg_fm.data.normalize import normalize_sequence


def window(seed=0, C=15):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(P.CONTEXT_LEN, C, P.UNIT_SEC * P.FS, generator=g) * 40 + 7
    x[:, 3] *= 5                                    # channels of different amplitude
    return x


def test_chunk_order_matches_tokens_continuous():
    x = window()
    ref = x.unfold(-1, P.CHUNK, P.CHUNK).permute(0, 2, 1, 3).contiguous()
    ref = ref.view(1, -1, x.shape[1], P.CHUNK).squeeze(0)       # as in net_eegfm.tokens_continuous
    assert torch.equal(P.to_chunks(x), ref)


def test_stats_reproduce_normalize_sequence_exactly():
    for seed in range(3):
        c = P.to_chunks(window(seed))
        mine = P.apply_stats(c, *P.seq_stats(c))
        ref = normalize_sequence(c.unsqueeze(0)).squeeze(0)
        assert torch.equal(mine, ref), (mine - ref).abs().max()
    c = P.to_chunks(window())
    c[:, 5] = 0                                                  # a dead channel
    assert torch.equal(P.apply_stats(c, *P.seq_stats(c)), normalize_sequence(c.unsqueeze(0)).squeeze(0))


def test_material_uses_the_window_statistics():
    c = P.to_chunks(window(0))
    mean, scale = P.seq_stats(c)
    other = P.to_chunks(window(1)[:1])
    got = P.apply_stats(other, mean, scale)
    want = (other - mean.squeeze(0)) / scale
    assert torch.allclose(got, want)
    # and not with its own statistics
    assert not torch.allclose(got, P.apply_stats(other, *P.seq_stats(other)))


def test_spans():
    spans = P.make_spans([0, 4, 16, 32, 60, 120, 300, 540], [4, 30], 30)
    assert len(spans) == 24
    s = P.Span("A", 4, 32)
    assert (s.start_sec, s.start_chunk, s.n_chunks) == (534, 1068, 8)
    s = P.Span("B", 30, 540)
    assert (s.start_sec, s.start_chunk) == (0, 0)
    s = P.Span("A", 4, 0)
    assert s.start_chunk + s.n_chunks == P.TARGET_START_CHUNK          # lag 0 ends at 9:30
    for bad in (P.Span("A", 4, 567), P.Span("A", 30, 545), P.Span("A", 4, 3)):
        try:
            bad.check()
        except AssertionError:
            continue
        raise AssertionError(f"{bad} should be rejected")


def test_perturb_changes_only_the_span():
    c = P.to_chunks(window())
    base = P.apply_stats(c, *P.seq_stats(c))
    for span in P.make_spans([0, 4, 16, 32, 60, 120, 300, 540], [4, 30], 30):
        mat = torch.full((span.n_chunks,) + tuple(base.shape[1:]), 3.0)
        pert = P.perturb(base, span, mat)
        P.check_only_span_differs(base, pert, span)
        a = span.start_chunk
        assert torch.equal(pert[a:a + span.n_chunks], mat)
        assert not torch.equal(pert, base)
    try:                                       # a broken copy must be caught
        bad = base.clone()
        bad[P.TARGET_START_CHUNK + 5] += 1
        P.check_only_span_differs(base, bad, P.Span("A", 4, 32))
    except AssertionError:
        pass
    else:
        raise AssertionError("change in the target unit was not caught")


def test_window_index():
    # two recordings of subject 1 (rows 0-49, 50-99), one of subject 2 (rows 100-149)
    rec = np.repeat([0, 1, 2], 50)
    sub = np.repeat([1, 1, 2], 50)
    ep = np.concatenate([np.arange(50)] * 3)
    y = np.zeros(150, np.int64)
    y[70:74] = 1                          # seizure in recording 1: core units 71, 72
    y[130:132] = 1                        # seizure in recording 2: no core unit
    y[25] = 1                             # one seizure unit in recording 0, inside window 20-39
    idx = P.WindowIndex(y, sub, rec, ep, seed=0)
    assert idx.core_rows.tolist() == [71, 72]

    w = np.arange(0, 20)                  # first window of recording 0: no previous unit
    assert idx.eligible_a(w) is None
    w = np.arange(20, 40)
    assert idx.eligible_a(w) == 19
    assert idx.donor_b(w, (0, 1)) is None            # has a seizure (row 25)
    w = np.arange(26, 46)
    assert idx.eligible_a(w) is None                 # previous unit 25 is a seizure
    w = np.arange(30, 50)
    assert idx.donor_b(w, (0, 2)) in (71, 72)        # donor from the other recording
    assert idx.donor_b(w, (0, 2)) == idx.donor_b(w, (0, 2))   # fixed seed
    w = np.arange(51, 71)
    assert idx.eligible_a(w) == 50
    assert idx.target_info(w) == (1, False)          # target is row 70, a seizure unit
    assert idx.target_info(np.arange(50, 70)) == (0, True)   # target 69, onset at 70
    w = np.arange(110, 130)
    assert idx.donor_b(w, (0, 3)) is None            # subject 2 has no core unit elsewhere
    assert idx.target_info(w) == (0, True)           # onset at 130, right after the target
    assert idx.eligible_a(np.arange(130, 140)) is None   # short window


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
        print('ok', t.__name__)
    print(f'{len(tests)} tests passed')
