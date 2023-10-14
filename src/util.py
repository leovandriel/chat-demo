import os


def get_secret(key: str) -> str:
    filename = f"{key}.txt"
    if os.path.isfile(filename):
        with open(filename) as f:
            return f.read().strip()
    else:
        raise Exception(
            f"Unable to find {key}.txt. Please create this file and add your secret."
        )
