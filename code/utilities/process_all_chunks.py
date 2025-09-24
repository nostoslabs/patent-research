#!/usr/bin/env python3
"""Process all OpenAI embedding chunks sequentially."""

import os
import json
import time
import logging
from pathlib import Path
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def submit_and_wait_for_chunk(chunk_file):
    """Submit a chunk and wait for completion."""
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    logger.info(f"Submitting {chunk_file}")

    # Upload file
    with open(chunk_file, 'rb') as f:
        uploaded_file = client.files.create(file=f, purpose='batch')

    # Create batch
    batch = client.batches.create(
        input_file_id=uploaded_file.id,
        endpoint="/v1/embeddings",
        completion_window="24h",
        metadata={"description": f"Patent embeddings chunk: {chunk_file}"}
    )

    batch_id = batch.id
    logger.info(f"Batch created: {batch_id}")

    # Wait for completion
    while True:
        batch_status = client.batches.retrieve(batch_id)
        status = batch_status.status

        if status == 'completed':
            logger.info(f"Batch {batch_id} completed successfully!")
            return batch_status
        elif status in ['failed', 'expired', 'cancelled']:
            logger.error(f"Batch {batch_id} failed with status: {status}")
            return None
        else:
            logger.info(f"Batch {batch_id} status: {status}")
            time.sleep(30)  # Check every 30 seconds

def download_and_append_results(batch_status, output_file):
    """Download results and append to output file."""
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    if not batch_status.output_file_id:
        logger.error("No output file available")
        return False

    # Download results
    logger.info(f"Downloading results for batch {batch_status.id}")
    file_response = client.files.content(batch_status.output_file_id)

    # Process and append results
    processed_count = 0
    error_count = 0

    with open(output_file, 'a') as outfile:  # Append mode
        for line in file_response.content.decode('utf-8').strip().split('\n'):
            if not line:
                continue

            try:
                result = json.loads(line)

                if result.get('response', {}).get('status_code') == 200:
                    custom_id = result['custom_id']
                    patent_id = custom_id.replace('embedding-', '')

                    embedding_data = result['response']['body']['data'][0]
                    embedding = embedding_data['embedding']

                    embedding_entry = {
                        'id': patent_id,
                        'embedding': embedding,
                        'model': 'text-embedding-3-small',
                        'usage': result['response']['body'].get('usage', {})
                    }

                    outfile.write(json.dumps(embedding_entry) + '\n')
                    processed_count += 1
                else:
                    error_count += 1
                    logger.warning(f"Failed request: {result.get('custom_id')} - {result.get('error', {})}")

            except json.JSONDecodeError:
                error_count += 1
                continue

    logger.info(f"Processed {processed_count} embeddings, {error_count} errors")
    return True

def process_all_chunks():
    """Process all chunks sequentially."""
    chunk_files = sorted([f for f in os.listdir('.') if f.startswith('openai_batch_chunk_') and f.endswith('.jsonl')])

    if not chunk_files:
        logger.error("No chunk files found!")
        return

    logger.info(f"Found {len(chunk_files)} chunks to process")
    output_file = "results/openai_embeddings_all_chunks.jsonl"

    # Clear output file
    Path("results").mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        pass  # Clear file

    total_processed = 0

    for i, chunk_file in enumerate(chunk_files):
        logger.info(f"Processing chunk {i+1}/{len(chunk_files)}: {chunk_file}")

        # Submit and wait for completion
        batch_status = submit_and_wait_for_chunk(chunk_file)

        if batch_status:
            # Download and append results
            if download_and_append_results(batch_status, output_file):
                # Count total embeddings
                with open(output_file, 'r') as f:
                    current_count = sum(1 for line in f)

                total_processed = current_count
                logger.info(f"Total embeddings so far: {total_processed}")
            else:
                logger.error(f"Failed to download results for {chunk_file}")
        else:
            logger.error(f"Failed to process {chunk_file}")
            # Continue with next chunk

    logger.info(f"All chunks processed! Total embeddings: {total_processed}")
    return total_processed

if __name__ == "__main__":
    total = process_all_chunks()
    print(f"Final total: {total} OpenAI embeddings")