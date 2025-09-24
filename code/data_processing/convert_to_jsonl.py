"""Convert JSON patent data to JSONL format."""

import json
from pathlib import Path
from typing import Any


def convert_json_to_jsonl(
    input_file: str = "patent_abstracts.json",
    output_file: str = "patent_abstracts.jsonl"
) -> int:
    """
    Convert JSON file to JSONL format.

    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSONL file

    Returns:
        Number of records converted
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file {input_path} not found")

    print(f"Converting {input_path} to {output_path}...")

    # Load JSON data
    with open(input_path, encoding="utf-8") as f:
        data: list[dict[str, Any]] = json.load(f)

    # Write as JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for record in data:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

    print(f"Converted {len(data)} records to {output_path}")
    return len(data)


def main() -> None:
    """Convert patent abstracts from JSON to JSONL format."""
    count = convert_json_to_jsonl()
    print(f"Successfully converted {count} patent records to JSONL format")


if __name__ == "__main__":
    main()
