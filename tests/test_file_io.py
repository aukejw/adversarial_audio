from pathlib import Path

import pytest

from mlx_audio_opt.file_io import to_json, try_load_json


def test_to_json(tmp_path: Path):
    test_dict = {"key1": "value1", "key2": [1, 2, 3], "key3": {"nested": "value"}}

    # save
    temp_path = tmp_path / "test.json"
    to_json(test_dict, temp_path, verbose=False)
    assert temp_path.exists()

    # load
    loaded_dict = try_load_json(temp_path)
    assert loaded_dict == test_dict


def test_try_load_json_with_dict():
    test_dict = {
        "key1": "value1",
        "key2": [1, 2, 3],
    }
    loaded_dict = try_load_json(test_dict)
    assert loaded_dict is test_dict


def test_try_load_json_file_not_found():
    non_existent_file = "/path/that/does/not/exist/file.json"
    with pytest.raises(FileNotFoundError):
        try_load_json(non_existent_file)
