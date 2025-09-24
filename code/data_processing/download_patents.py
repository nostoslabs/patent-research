"""Download and sample patent abstracts from the BigPatent dataset."""

import json
import random
from pathlib import Path
from typing import Any

from datasets import load_dataset


def download_and_sample_patents(
    sample_size: int = 1000,
    output_file: str = "patent_abstracts.jsonl",
    seed: int = 42
) -> list[dict[str, Any]]:
    """
    Download BigPatent dataset and sample patent abstracts.

    Args:
        sample_size: Number of patents to sample
        output_file: Output JSONL file name
        seed: Random seed for reproducibility

    Returns:
        List of sampled patent data
    """
    print("Loading BigPatent dataset...")

    # Load the dataset - using streaming to avoid loading everything into memory
    # Use mteb/big-patent dataset which is available in parquet format
    print("Loading mteb/big-patent dataset (test split)...")
    dataset = load_dataset("mteb/big-patent", split="test", streaming=True)

    print(f"Collecting {sample_size} random patents...")

    # First, let's examine the structure of the first few records
    print("Examining dataset structure...")
    patents = []
    for i, patent in enumerate(dataset):
        if i == 0:  # Print structure of first record
            print(f"Available fields: {list(patent.keys())}")
            for key, value in patent.items():
                value_preview = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                print(f"  {key}: {value_preview}")
            print()

        # Extract available fields - mteb dataset has 'sentences' and 'labels'
        full_text = patent.get("sentences", "")

        # Extract abstract from the patent text (usually at the beginning)
        # Look for common patterns in patent abstracts
        abstract = ""
        if full_text:
            lines = full_text.split('\n')
            # Look for abstract section or take first meaningful paragraph
            for j, line in enumerate(lines[:50]):  # Check first 50 lines
                line = line.strip()
                if any(keyword in line.upper() for keyword in ['ABSTRACT', 'SUMMARY', 'FIELD OF']):
                    # Found abstract section, get next few lines
                    abstract_lines = []
                    for k in range(j + 1, min(j + 20, len(lines))):
                        next_line = lines[k].strip()
                        if next_line and not next_line.startswith('[') and len(next_line) > 20:
                            abstract_lines.append(next_line)
                        elif len(abstract_lines) > 3:  # Stop if we have enough
                            break
                    abstract = ' '.join(abstract_lines)
                    break

            # If no abstract section found, use the first substantial paragraph
            if not abstract:
                for line in lines[:20]:
                    line = line.strip()
                    if len(line) > 100 and not line.startswith('[') and '.' in line:
                        abstract = line
                        break

        patent_data = {
            "id": f"patent_{i}",
            "abstract": abstract,
            "full_text": full_text[:1000] + "..." if len(full_text) > 1000 else full_text,  # Truncated for storage
            "classification": str(patent.get("labels", ""))
        }
        patents.append(patent_data)

        # Stop after collecting enough for a good sample pool
        if len(patents) >= sample_size * 10:  # 10x sample size for good randomness
            break

    print(f"Collected {len(patents)} patents. Sampling {sample_size}...")

    # Sample randomly
    random.seed(seed)
    sampled_patents = random.sample(patents, min(sample_size, len(patents)))

    # Filter out patents without abstracts
    valid_patents = [p for p in sampled_patents if p["abstract"].strip()]

    print(f"Found {len(valid_patents)} patents with valid abstracts")

    # Save to JSONL file
    output_path = Path(output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        for patent in valid_patents:
            json.dump(patent, f, ensure_ascii=False)
            f.write("\n")

    print(f"Saved {len(valid_patents)} patent abstracts to {output_path}")

    return valid_patents


def main() -> None:
    """Main function to run the patent download script."""
    patents = download_and_sample_patents()

    # Print some stats
    if patents:
        abstract_lengths = [len(p["abstract"]) for p in patents if p["abstract"]]
        if abstract_lengths:
            avg_length = sum(abstract_lengths) / len(abstract_lengths)

            print("\nDataset statistics:")
            print(f"Total patents: {len(patents)}")
            print(f"Patents with abstracts: {len(abstract_lengths)}")
            print(f"Average abstract length: {avg_length:.1f} characters")
            print(f"Min abstract length: {min(abstract_lengths)}")
            print(f"Max abstract length: {max(abstract_lengths)}")

            # Show a sample abstract
            sample_patent = next(p for p in patents if p["abstract"])
            print(f"\nSample abstract (ID: {sample_patent['id']}):")
            print(f"Classification: {sample_patent['classification']}")
            print(f"Abstract: {sample_patent['abstract'][:300]}...")
        else:
            print("\nNo patents found with valid abstracts!")
            print("Check the dataset field mapping.")
    else:
        print("\nNo patents collected!")


if __name__ == "__main__":
    main()

