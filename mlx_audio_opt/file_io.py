import json
from pathlib import Path
from typing import Dict, Union


def to_json(
    dictionary: Dict,
    output_file: Union[str, Path],
    verbose: bool = True,
) -> None:
    """Write a dictionary to a json file.

    Args:
        dictionary: The dictionary to write.
        output_file: The output file to write to.
        verbose: Whether to print a message when the file is saved.

    """
    with Path(output_file).open("w") as f:
        json.dump(dictionary, f, indent=4)
    if verbose:
        print(f"Saved json to '{output_file}'")


def try_load_json(
    input_file: Union[Dict, str, Path],
):
    """Try to load a dict from json. If already a dict, return that.

    Args:
        input_file: The input file to load, or a dict.

    """
    if isinstance(input_file, (str, Path)):
        input_file = Path(input_file)
        if not input_file.exists():
            raise FileNotFoundError(f"File '{input_file}' does not exist")
        with input_file.open("r") as f:
            return dict(json.load(f))

    assert isinstance(input_file, dict), type(input_file)
    return input_file
