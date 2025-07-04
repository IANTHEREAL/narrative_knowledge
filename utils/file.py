from typing import Tuple
from pathlib import Path


def read_file_content(path: str) -> str:
    """Reads the entire content of a file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at {path}")
    except Exception as e:
        raise IOError(f"Error reading file {path}: {e}")


def extract_file_info(file_path: str) -> Tuple[str, str]:
    """
    Extract file name and extension from a file path.

    Parameters:
    - file_path: Path to the file

    Returns:
    - Tuple of (file_name, file_extension)
    """
    path = Path(file_path)
    return path.stem, path.suffix
