"""Load the Mango DMC NIR dataset (Anderson et al. 2020, Mendeley file v3), build fruit IDs and check counts."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data' / 'MangoDMC_NIR_Data_v3.csv'

TARGET = 'DM'
SETS = ['Cal', 'Tuning', 'Val Ext', 'Val Ext 2']
TRAIN_SETS = ['Cal', 'Tuning']
# The file has no fruit ID: duplicate spectra of one fruit share these fields, including the reference DM value.
FRUIT_KEYS = ['Set', 'Pop', 'Type', 'Cultivar', 'DM']

EXPECTED_SHAPE = (12011, 315)
EXPECTED_N_WAVELENGTHS = 306  # 285-1200 nm, 3 nm step
# (spectra, fruit) per set. Tuning and Val Ext match Anderson et al. 2020 (Table 1);
# the paper reports 2,864 Cal fruit, the FRUIT_KEYS grouping yields 2,866.
EXPECTED_COUNTS = {
    'Cal': (7413, 2866),
    'Tuning': (2830, 1086),
    'Val Ext': (1448, 725),
    'Val Ext 2': (320, 160),
}


def wavelength_columns(df):
    return [c for c in df.columns if c.isdigit()]


def wavelengths(df):
    return np.array([int(c) for c in wavelength_columns(df)])


def add_fruit_id(df):
    out = df.copy()
    out['fruit'] = out.groupby(FRUIT_KEYS, sort=False, dropna=False).ngroup()
    return out


def count_table(df):
    g = df.groupby('Set', sort=False)
    return pd.DataFrame({'spectra': g.size(), 'fruit': g['fruit'].nunique()}).reindex(SETS)


def check_counts(df):
    """Raise ValueError if the number of wavelengths or the per-set spectra/fruit counts differ from the expected ones."""
    errors = []
    n_wl = len(wavelength_columns(df))
    if n_wl != EXPECTED_N_WAVELENGTHS:
        errors.append(f'{n_wl} wavelength columns, expected {EXPECTED_N_WAVELENGTHS}')
    counts = count_table(df)
    for s, expected in EXPECTED_COUNTS.items():
        got = tuple(counts.loc[s, ['spectra', 'fruit']])
        if got != expected:
            errors.append(f'{s}: {got[0]} spectra / {got[1]} fruit, expected {expected[0]} / {expected[1]}')
    if errors:
        raise ValueError('Dataset check failed:\n  ' + '\n  '.join(errors))
    return counts


def load_data(path=DATA_PATH, validate=True):
    df = pd.read_csv(path)
    if validate and df.shape != EXPECTED_SHAPE:
        raise ValueError(f'Expected shape {EXPECTED_SHAPE}, got {df.shape}')
    df = add_fruit_id(df)
    if validate:
        check_counts(df)
    return df


def set_masks(df):
    return {s: (df['Set'] == s).to_numpy() for s in SETS}


def fruit_table(df, pred):
    """One row per fruit: reference DM (y) and the mean prediction (p) over the fruit's duplicate spectra."""
    return (df.assign(pred=pred)
              .groupby('fruit')
              .agg(Set=('Set', 'first'), Pop=('Pop', 'first'), Cultivar=('Cultivar', 'first'), Type=('Type', 'first'),
                   y=(TARGET, 'first'), p=('pred', 'mean'), n_spectra=('pred', 'size')))
