import os

from pathlib import Path

def validate_path(path: str):
    assert isinstance(path, str | Path), f"File paths must be type 'str' or 'pathlib.Path'. Got {type(path)}"
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)