"""Generate embeddings for all patent abstracts."""

from generate_embeddings import add_embeddings_to_patents

if __name__ == "__main__":
    print("Starting embedding generation for all patents...")
    print("This will run in the background and save progress every 10 records.")
    print("Output file: patent_abstracts_with_embeddings.jsonl")
    print("-" * 60)
    
    # Process all records (no max_records limit)
    records = add_embeddings_to_patents(
        max_records=None,  # Process all records
        batch_size=10,     # Save progress every 10 records
    )
    
    print(f"\n{'='*60}")
    print(f"COMPLETED: Generated embeddings for {len(records)} patents")
    print(f"Output saved to: patent_abstracts_with_embeddings.jsonl")