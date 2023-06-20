import os

def validate_path(path: str):
    assert isinstance(path, str), f"File paths must be type 'str'. Got {type(path)=}"
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)