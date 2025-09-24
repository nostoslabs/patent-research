"""
File utilities for handling compressed and uncompressed files
"""
import json
import lzma
import gzip
from pathlib import Path
from typing import Iterator, Union, TextIO
import contextlib

def smart_open(file_path: Union[str, Path], mode: str = 'r', **kwargs):
    """
    Smart file opener that handles compressed and uncompressed files automatically.

    Supports:
    - .xz files (lzma compression)
    - .gz files (gzip compression)
    - Regular text files

    Args:
        file_path: Path to the file
        mode: File open mode (default: 'r')
        **kwargs: Additional arguments passed to the underlying open function

    Returns:
        File handle (context manager)
    """
    file_path = Path(file_path)

    # Handle different compression formats
    if file_path.suffix == '.xz':
        return lzma.open(file_path, mode='rt' if 'r' in mode else 'wt', **kwargs)
    elif file_path.suffix == '.gz':
        return gzip.open(file_path, mode='rt' if 'r' in mode else 'wt', **kwargs)
    else:
        return open(file_path, mode, **kwargs)

def read_jsonl_file(file_path: Union[str, Path]) -> Iterator[dict]:
    """
    Read a JSONL file (compressed or uncompressed) line by line.

    Args:
        file_path: Path to the JSONL file

    Yields:
        dict: Parsed JSON objects from each line
    """
    file_path = Path(file_path)

    # Check if compressed version exists if uncompressed is requested
    if not file_path.exists():
        compressed_path = file_path.with_suffix(file_path.suffix + '.xz')
        if compressed_path.exists():
            file_path = compressed_path
            print(f"📦 Using compressed version: {compressed_path}")

    with smart_open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line.strip())
                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON decode error on line {line_num}: {e}")
                    continue

def find_best_file_version(file_path: Union[str, Path]) -> Path:
    """
    Find the best available version of a file (compressed or uncompressed).

    Preference order:
    1. Exact file path if it exists
    2. .xz compressed version
    3. .gz compressed version
    4. Original path (even if it doesn't exist)

    Args:
        file_path: Base file path

    Returns:
        Path: Best available file path
    """
    file_path = Path(file_path)

    # If exact path exists, use it
    if file_path.exists():
        return file_path

    # Try compressed versions
    xz_path = file_path.with_suffix(file_path.suffix + '.xz')
    if xz_path.exists():
        return xz_path

    gz_path = file_path.with_suffix(file_path.suffix + '.gz')
    if gz_path.exists():
        return gz_path

    # Return original path (caller will handle if it doesn't exist)
    return file_path

@contextlib.contextmanager
def smart_jsonl_reader(file_path: Union[str, Path]):
    """
    Context manager for reading JSONL files with automatic compression detection.

    Args:
        file_path: Path to the JSONL file

    Yields:
        Iterator[dict]: JSON objects from the file
    """
    best_path = find_best_file_version(file_path)

    if not best_path.exists():
        raise FileNotFoundError(f"File not found: {file_path} (also checked compressed versions)")

    if best_path != file_path:
        print(f"📦 Using {best_path.suffix} compressed version: {best_path}")

    yield read_jsonl_file(best_path)