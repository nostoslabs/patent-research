"""Submit OpenAI batch embedding requests and monitor progress."""

import os
import json
import time
from pathlib import Path
from openai import OpenAI


def submit_batch(batch_file: str, description: str = "Patent embeddings for ground truth evaluation"):
    """Submit batch file to OpenAI Batch API."""

    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY environment variable not set")
        return None

    try:
        # Upload the batch file
        print(f"Uploading batch file: {batch_file}")
        with open(batch_file, 'rb') as f:
            uploaded_file = client.files.create(
                file=f,
                purpose='batch'
            )

        print(f"File uploaded with ID: {uploaded_file.id}")

        # Create the batch job
        batch = client.batches.create(
            input_file_id=uploaded_file.id,
            endpoint="/v1/embeddings",
            completion_window="24h",
            metadata={"description": description}
        )

        print(f"Batch created with ID: {batch.id}")
        print(f"Status: {batch.status}")
        print(f"Created at: {batch.created_at}")

        # Save batch info
        batch_info = {
            "batch_id": batch.id,
            "file_id": uploaded_file.id,
            "status": batch.status,
            "created_at": batch.created_at,
            "input_file": batch_file,
            "description": description
        }

        with open("openai_batch_info.json", 'w') as f:
            json.dump(batch_info, f, indent=2)

        print(f"Batch info saved to openai_batch_info.json")

        return batch

    except Exception as e:
        print(f"Error submitting batch: {e}")
        return None


def check_batch_status(batch_id: str = None):
    """Check the status of a batch job."""

    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    if not batch_id:
        # Load from saved file
        if Path("openai_batch_info.json").exists():
            with open("openai_batch_info.json", 'r') as f:
                batch_info = json.load(f)
                batch_id = batch_info["batch_id"]
        else:
            print("No batch ID provided and no saved batch info found")
            return None

    try:
        batch = client.batches.retrieve(batch_id)

        print(f"Batch ID: {batch.id}")
        print(f"Status: {batch.status}")
        print(f"Created: {batch.created_at}")
        print(f"In progress: {batch.in_progress_at}")
        print(f"Completed: {batch.completed_at}")
        print(f"Failed: {batch.failed_at}")
        print(f"Expired: {batch.expired_at}")
        print(f"Finalizing: {batch.finalizing_at}")

        if hasattr(batch, 'request_counts'):
            print(f"Total requests: {batch.request_counts.total}")
            print(f"Completed: {batch.request_counts.completed}")
            print(f"Failed: {batch.request_counts.failed}")

        return batch

    except Exception as e:
        print(f"Error checking batch status: {e}")
        return None


def download_results(batch_id: str = None, output_file: str = "openai_batch_results.jsonl"):
    """Download batch results when complete."""

    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    if not batch_id:
        # Load from saved file
        if Path("openai_batch_info.json").exists():
            with open("openai_batch_info.json", 'r') as f:
                batch_info = json.load(f)
                batch_id = batch_info["batch_id"]
        else:
            print("No batch ID provided and no saved batch info found")
            return None

    try:
        batch = client.batches.retrieve(batch_id)

        if batch.status != "completed":
            print(f"Batch not completed yet. Status: {batch.status}")
            return None

        if not batch.output_file_id:
            print("No output file available")
            return None

        # Download the results
        print(f"Downloading results from file ID: {batch.output_file_id}")
        result = client.files.content(batch.output_file_id)

        with open(output_file, 'wb') as f:
            f.write(result.content)

        print(f"Results saved to {output_file}")

        # Also download error file if it exists
        if batch.error_file_id:
            error_file = f"openai_batch_errors.jsonl"
            error_result = client.files.content(batch.error_file_id)
            with open(error_file, 'wb') as f:
                f.write(error_result.content)
            print(f"Errors saved to {error_file}")

        return output_file

    except Exception as e:
        print(f"Error downloading results: {e}")
        return None


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  Submit batch: python submit_openai_batch.py submit <batch_file>")
        print("  Check status: python submit_openai_batch.py status [batch_id]")
        print("  Download results: python submit_openai_batch.py download [batch_id]")
        return

    command = sys.argv[1]

    if command == "submit":
        if len(sys.argv) < 3:
            print("Please provide batch file path")
            return
        batch_file = sys.argv[2]
        submit_batch(batch_file)

    elif command == "status":
        batch_id = sys.argv[2] if len(sys.argv) > 2 else None
        check_batch_status(batch_id)

    elif command == "download":
        batch_id = sys.argv[2] if len(sys.argv) > 2 else None
        download_results(batch_id)

    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()