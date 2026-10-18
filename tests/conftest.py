import pytest

from src.data import DATA_PATH, load_data


@pytest.fixture(scope='session')
def df():
    if not DATA_PATH.exists():
        pytest.skip(f'data file not found: {DATA_PATH} (download it from Mendeley Data, see README.md)')
    return load_data()
