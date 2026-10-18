import numpy as np

from src.data import EXPECTED_COUNTS, count_table, wavelengths


def test_shape_and_wavelengths(df):
    assert len(df) == 12011
    wl = wavelengths(df)
    assert len(wl) == 306 and wl[0] == 285 and wl[-1] == 1200
    assert set(np.diff(wl)) == {3}


def test_counts_per_set(df):
    counts = count_table(df)
    for s, (n_spectra, n_fruit) in EXPECTED_COUNTS.items():
        assert counts.loc[s, 'spectra'] == n_spectra, s
        assert counts.loc[s, 'fruit'] == n_fruit, s


def test_training_spectra(df):
    # Anderson et al. 2020: 10,243 spectra in the first three seasons after outlier removal
    assert df['Set'].isin(['Cal', 'Tuning']).sum() == 10243


def test_populations_and_seasons(df):
    # Anderson et al. 2020: 66 calibration, 28 tuning and 18 test populations; season 4 is the test season
    pops = df.groupby('Set')['Pop'].nunique()
    assert (pops['Cal'], pops['Tuning'], pops['Val Ext']) == (66, 28, 18)
    assert set(df.loc[df['Set'].isin(['Cal', 'Tuning']), 'Season']) == {1, 2, 3}
    assert set(df.loc[df['Set'] == 'Val Ext', 'Season']) == {4}


def test_season5_set(df):
    # Walsh et al. 2024: season 5, population 128 (R2E2) and 129 (KP), all hard green
    v2 = df[df['Set'] == 'Val Ext 2']
    assert set(v2['Season']) == {5}
    assert v2.groupby('Pop')['Cultivar'].unique().apply(list).to_dict() == {128: ['R2E2'], 129: ['KP']}
    assert set(v2['Type']) == {'Hard Green'}


def test_fruit_scanned_on_one_day(df):
    # Spectra and the reference analysis of a fruit were done within a single day (Anderson et al. 2020, 2.1),
    # so a fruit ID spanning several dates would mean two fruit were merged.
    assert (df.groupby('fruit')['Date'].nunique() == 1).all()
