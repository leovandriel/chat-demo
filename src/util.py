"""Utility functions for the project."""

from pathlib import Path


def get_secret(key: str) -> str:
    """Get a secret from a file."""
    file = Path(f"{key}.txt")
    if file.is_file():
        with file.open() as f:
            return f.read().strip()
    else:
        msg = f"Unable to find {key}.txt. Please create this file and add your secret."
        raise RuntimeError(
            msg,
        )
