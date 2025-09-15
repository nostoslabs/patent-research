"""Generate embeddings for patent abstracts using Ollama."""

import json
import time
from pathlib import Path
from typing import Any

import ollama


def load_jsonl(file_path: str) -> list[dict[str, Any]]:
    """
    Load data from JSONL file.

    Args:
        file_path: Path to JSONL file

    Returns:
        List of records
    """
    records = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_jsonl(records: list[dict[str, Any]], file_path: str) -> None:
    """
    Save records to JSONL file.

    Args:
        records: List of records to save
        file_path: Output file path
    """
    with open(file_path, "w", encoding="utf-8") as f:
        for record in records:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")


def generate_embedding(
    text: str, model: str = "embeddinggemma", client: ollama.Client | None = None
) -> list[float]:
    """
    Generate embedding for given text using Ollama.

    Args:
        text: Text to embed
        model: Ollama model name
        client: Ollama client instance

    Returns:
        Embedding vector as list of floats
    """
    if client is None:
        client = ollama.Client()

    try:
        response = client.embeddings(model=model, prompt=text)
        return response["embedding"]
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise


def add_embeddings_to_patents(
    input_file: str = "patent_abstracts.jsonl",
    output_file: str = "patent_abstracts_with_embeddings.jsonl",
    model: str = "embeddinggemma",
    batch_size: int = 10,
    max_records: int | None = None,
) -> list[dict[str, Any]]:
    """
    Add embeddings to patent records.

    Args:
        input_file: Input JSONL file path
        output_file: Output JSONL file path
        model: Embedding model name
        batch_size: Number of records to process before saving progress
        max_records: Maximum number of records to process (None for all)

    Returns:
        List of processed records
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file {input_path} not found")

    print(f"Loading patents from {input_path}...")
    records = load_jsonl(str(input_path))

    if max_records:
        records = records[:max_records]
        print(f"Processing first {len(records)} records")

    print(f"Generating embeddings using model: {model}")

    # Initialize Ollama client
    client = ollama.Client()

    # Check if model is available
    try:
        print(f"Testing connection to {model}...")
        test_response = client.embeddings(model=model, prompt="test")
        embedding_dim = len(test_response["embedding"])
        print(f"Model {model} ready. Embedding dimension: {embedding_dim}")
    except Exception as e:
        print(f"Error connecting to model {model}: {e}")
        print("Make sure the model is installed: ollama pull embeddinggemma")
        raise

    processed_records = []
    start_time = time.time()

    for i, record in enumerate(records):
        try:
            # Generate embedding for the abstract
            abstract = record.get("abstract", "")
            if not abstract.strip():
                print(f"Skipping record {i}: empty abstract")
                continue

            print(f"Processing record {i+1}/{len(records)}: {record['id']}")

            embedding = generate_embedding(abstract, model, client)

            # Create new record with embedding data
            enhanced_record = record.copy()

            # Initialize embeddings array if it doesn't exist
            if "embeddings" not in enhanced_record:
                enhanced_record["embeddings"] = []

            # Add embedding with model info
            embedding_data = {
                "model": model,
                "embedding": embedding,
                "text_source": "abstract",
                "embedding_dim": len(embedding),
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            enhanced_record["embeddings"].append(embedding_data)
            processed_records.append(enhanced_record)

            # Save progress every batch_size records
            if (i + 1) % batch_size == 0:
                print(f"Saving progress... ({i+1}/{len(records)} records)")
                save_jsonl(processed_records, str(output_path))

            # Small delay to avoid overwhelming the model
            time.sleep(0.1)

        except Exception as e:
            print(f"Error processing record {i}: {e}")
            continue

    # Save final results
    save_jsonl(processed_records, str(output_path))

    elapsed = time.time() - start_time
    print(f"\nCompleted processing {len(processed_records)} records in {elapsed:.1f}s")
    print(f"Average time per embedding: {elapsed/len(processed_records):.2f}s")
    print(f"Results saved to {output_path}")

    return processed_records


def main() -> None:
    """Generate embeddings for patent abstracts."""
    # Process first 50 records as a test
    records = add_embeddings_to_patents(
        max_records=50,  # Start with smaller batch for testing
        batch_size=5,
    )

    if records:
        # Show sample embedding info
        sample = records[0]
        embedding_info = sample["embeddings"][0]
        print("\nSample embedding info:")
        print(f"Model: {embedding_info['model']}")
        print(f"Dimension: {embedding_info['embedding_dim']}")
        print(f"First 5 values: {embedding_info['embedding'][:5]}")


if __name__ == "__main__":
    main()
