import pandas as pd
import pytest

from src import data


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def write_signal(directory, name: str, rows: list[tuple[str, str, float]], zone: str = "DE") -> None:
    frame = pd.DataFrame(
        [(ts(t), zone, ts(a), v) for t, a, v in rows],
        columns=["target_time", "zone_key", "available_at", "value"],
    )
    frame.to_parquet(directory / data.CATALOG[name].filename)


def _clear_caches() -> None:
    data.load_signal.cache_clear()
    data._panel_slice.cache_clear()
    data._target_frame.cache_clear()


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Redirect src.data to a temp directory and reset all parquet caches."""
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    _clear_caches()
    yield tmp_path
    _clear_caches()
