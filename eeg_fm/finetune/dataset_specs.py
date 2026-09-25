"""Settings of each downstream task: channel order, sampling rate, unit length, task type and readout budget."""

FACED_CHANNELS = [
    'Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5', 'FC6',
    'Cz', 'C3', 'C4', 'T7', 'T8', 'CP1', 'CP2', 'CP5', 'CP6', 'Pz', 'P3',
    'P4', 'P7', 'P8', 'PO3', 'PO4', 'Oz', 'O1', 'O2',
]
KAGGLE_ERN_CHANNELS = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8',
    'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2',
]
SEED_VIG_CHANNELS = [
    'FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8', 'CP1', 'CP2', 'P1', 'PZ', 'P2',
    'PO3', 'POZ', 'PO4', 'O1', 'OZ', 'O2',
]
CHB_MIT_CHANNELS = [
    'FP1', 'F7', 'T7', 'P7', 'F3', 'C3', 'P3', 'FP2', 'F8', 'P8', 'F4', 'C4',
    'P4', 'FZ', 'CZ',
]

DATASET_SPECS = {
    'faced': dict(
        base_fs=250,
        seg_sec=10,
        task='multiclass',
        n_outputs=9,
        label_dtype='long',
        channels=FACED_CHANNELS,
        readout_budget=19200,
    ),
    'errorern': dict(
        base_fs=200,
        seg_sec=2,
        task='binary',
        n_outputs=1,
        label_dtype='float',
        channels=KAGGLE_ERN_CHANNELS,
        readout_budget=3840,
        n_folds=4,
        val_frac=0.15,
    ),
    'seedvig': dict(
        base_fs=200,
        seg_sec=8,
        task='regression',
        n_outputs=1,
        label_dtype='float',
        channels=SEED_VIG_CHANNELS,
        readout_budget=19200,
    ),
    'sleepedf': dict(
        base_fs=100,
        seg_sec=30,
        task='seq2seq',
        n_outputs=5,
        channels=['Fpz', 'Pz'],
        context_len=20,
        readout_budget=12000,
        n_folds=5,
        cont_normalize='window',
    ),
    'chbmit': dict(
        base_fs=256,
        seg_sec=30,
        task='seq2seq',
        n_outputs=2,
        channels=CHB_MIT_CHANNELS,
        context_len=20,
        readout_budget=90000,
        seq_select_key='auc_pr',
        n_folds=5,
        cont_normalize='sequence',
    ),
    'chbmit_2ch': dict(
        base_fs=256,
        seg_sec=30,
        task='seq2seq',
        n_outputs=2,
        channels=['T7', 'P8'],
        channel_subset=(2, 9),
        context_len=20,
        readout_budget=12000,
        seq_select_key='auc_pr',
        n_folds=5,
        cont_normalize='sequence',
    ),
}


def get_spec(name, seg_dir=None):
    spec = dict(DATASET_SPECS[name])
    spec['seg_dir'] = seg_dir
    spec['name'] = name
    spec['n_channels'] = len(spec['channels'])
    return spec
