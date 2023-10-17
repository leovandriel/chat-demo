import pytest
import fix_import

from src.util import read_secret


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_read_secret(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    key = "test_key"
    with (tmp_path / f"{key}.txt").open("w") as f:
        f.write("secret1")
    assert read_secret(key) == "secret1"
    with pytest.raises(RuntimeError):
        read_secret("other")
