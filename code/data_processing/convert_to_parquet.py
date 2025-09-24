"""Convert JSONL patent data to Parquet format for Apple Embedding Atlas."""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_jsonl_with_embeddings(file_path: str) -> List[Dict[str, Any]]:
    """
    Load JSONL data with embeddings.

    Args:
        file_path: Path to JSONL file

    Returns:
        List of records with embeddings
    """
    records = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                # Only include records that have embeddings
                if "embeddings" in record and len(record["embeddings"]) > 0:
                    records.append(record)
    return records


def extract_embedding_data(
    records: List[Dict[str, Any]], model_name: str = "embeddinggemma"
) -> pd.DataFrame:
    """
    Extract embedding data and metadata for Parquet format.

    Args:
        records: List of patent records with embeddings
        model_name: Name of the embedding model to extract

    Returns:
        DataFrame with columns suitable for Embedding Atlas
    """
    data_rows = []

    for record in records:
        # Find the embedding for the specified model
        embedding_data = None
        for emb in record.get("embeddings", []):
            if emb.get("model") == model_name:
                embedding_data = emb
                break

        if embedding_data is None:
            print(f"Warning: No embedding found for model {model_name} in record {record.get('id', 'unknown')}")
            continue

        # Extract embedding vector
        embedding = embedding_data.get("embedding", [])
        if not embedding:
            continue

        # Create row with required columns for Embedding Atlas
        row = {
            "id": record.get("id", ""),
            "text": record.get("abstract", ""),  # Main text content for embedding
            "full_text": record.get("full_text", ""),  # Additional context
            "classification": record.get("classification", ""),
            "embedding_model": embedding_data.get("model", ""),
            "embedding_dim": len(embedding),
            "generated_at": embedding_data.get("generated_at", ""),
            # Store embedding as list (will be converted to array column)
            "embedding": embedding,
        }

        # Add abstract length as metadata
        row["abstract_length"] = len(record.get("abstract", ""))
        
        # Add truncated text for display
        abstract = record.get("abstract", "")
        row["display_text"] = abstract[:200] + "..." if len(abstract) > 200 else abstract

        data_rows.append(row)

    df = pd.DataFrame(data_rows)
    
    # Convert embedding list to numpy array column
    if not df.empty:
        df["embedding"] = df["embedding"].apply(lambda x: np.array(x, dtype=np.float32))
    
    return df


def save_parquet_for_atlas(
    df: pd.DataFrame,
    output_file: str = "patent_embeddings_atlas.parquet",
    include_embeddings: bool = True,
) -> None:
    """
    Save DataFrame to Parquet format optimized for Embedding Atlas.

    Args:
        df: DataFrame with embedding data
        output_file: Output Parquet file path
        include_embeddings: Whether to include embedding vectors (large file)
    """
    output_path = Path(output_file)
    
    # Create a copy to avoid modifying original
    save_df = df.copy()
    
    if not include_embeddings and "embedding" in save_df.columns:
        # Remove embedding column to create smaller file for testing
        save_df = save_df.drop(columns=["embedding"])
        print("Note: Embeddings excluded from Parquet file (set include_embeddings=True to include)")
    
    # Save to Parquet
    save_df.to_parquet(output_path, index=False)
    
    file_size = output_path.stat().st_size / (1024 * 1024)  # MB
    print(f"Saved {len(save_df)} records to {output_path}")
    print(f"File size: {file_size:.1f} MB")
    
    # Print sample of data structure
    print("\nParquet file structure:")
    print(f"Columns: {list(save_df.columns)}")
    print(f"Dtypes:\n{save_df.dtypes}")
    
    if not save_df.empty:
        print(f"\nSample record:")
        sample = save_df.iloc[0]
        for col, val in sample.items():
            if col == "embedding" and hasattr(val, "shape"):
                print(f"  {col}: array shape {val.shape}, dtype {val.dtype}")
            else:
                val_str = str(val)[:100] + "..." if len(str(val)) > 100 else str(val)
                print(f"  {col}: {val_str}")


def create_atlas_ready_parquet(
    input_file: str = "patent_abstracts_with_embeddings.jsonl",
    output_file: str = "patent_embeddings_atlas.parquet",
    model_name: str = "embeddinggemma",
    include_embeddings: bool = True,
) -> pd.DataFrame:
    """
    Create Parquet file ready for Apple Embedding Atlas.

    Args:
        input_file: Input JSONL file with embeddings
        output_file: Output Parquet file
        model_name: Embedding model name to extract
        include_embeddings: Whether to include embedding vectors

    Returns:
        DataFrame that was saved
    """
    print(f"Loading data from {input_file}...")
    records = load_jsonl_with_embeddings(input_file)
    print(f"Loaded {len(records)} records with embeddings")

    if not records:
        raise ValueError("No records with embeddings found")

    print(f"Extracting embedding data for model: {model_name}")
    df = extract_embedding_data(records, model_name)
    
    if df.empty:
        raise ValueError(f"No embeddings found for model {model_name}")

    print(f"Processed {len(df)} records with embeddings")
    
    # Save to Parquet
    save_parquet_for_atlas(df, output_file, include_embeddings)
    
    return df


def main() -> None:
    """Create Parquet files for Embedding Atlas."""
    print("Creating Parquet file for Apple Embedding Atlas...")
    print("=" * 60)
    
    try:
        # Create full version with embeddings
        df = create_atlas_ready_parquet(
            input_file="patent_abstracts_with_embeddings.jsonl",
            output_file="patent_embeddings_atlas.parquet",
            include_embeddings=True
        )
        
        print(f"\nSuccessfully created Parquet file with {len(df)} patent embeddings")
        print("Ready for use with: embedding-atlas patent_embeddings_atlas.parquet")
        
        # Also create a smaller version without embeddings for quick testing
        create_atlas_ready_parquet(
            input_file="patent_abstracts_with_embeddings.jsonl",
            output_file="patent_metadata_only.parquet",
            include_embeddings=False
        )
        
    except Exception as e:
        print(f"Error: {e}")
        return


if __name__ == "__main__":
    main()