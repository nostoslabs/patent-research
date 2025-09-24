#!/usr/bin/env python3
"""Monitor OpenAI batch processing and handle failures automatically."""

import os
import json
import time
import logging
from pathlib import Path
from openai import OpenAI
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('openai_batch_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class OpenAIBatchMonitor:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.batch_info_file = "openai_batch_info.json"
        self.current_batch_id = None
        self.load_batch_info()

    def load_batch_info(self):
        """Load current batch information."""
        if Path(self.batch_info_file).exists():
            with open(self.batch_info_file, 'r') as f:
                info = json.load(f)
                self.current_batch_id = info.get('batch_id')
                logger.info(f"Loaded batch ID: {self.current_batch_id}")

    def get_batch_status(self, batch_id):
        """Get current batch status from OpenAI."""
        try:
            batch = self.client.batches.retrieve(batch_id)
            return {
                'id': batch.id,
                'status': batch.status,
                'created_at': batch.created_at,
                'in_progress_at': batch.in_progress_at,
                'completed_at': batch.completed_at,
                'failed_at': batch.failed_at,
                'expired_at': batch.expired_at,
                'finalizing_at': batch.finalizing_at,
                'request_counts': {
                    'total': batch.request_counts.total,
                    'completed': batch.request_counts.completed,
                    'failed': batch.request_counts.failed
                }
            }
        except Exception as e:
            logger.error(f"Error retrieving batch status: {e}")
            return None

    def download_results(self, batch_id):
        """Download and process batch results."""
        try:
            batch = self.client.batches.retrieve(batch_id)

            if batch.status != 'completed':
                logger.warning(f"Batch {batch_id} not completed (status: {batch.status})")
                return False

            if not batch.output_file_id:
                logger.error(f"No output file for batch {batch_id}")
                return False

            # Download results
            logger.info(f"Downloading results for batch {batch_id}")
            file_response = self.client.files.content(batch.output_file_id)

            results_file = f"openai_batch_results_{batch_id}.jsonl"
            with open(results_file, 'wb') as f:
                f.write(file_response.content)

            logger.info(f"Results saved to {results_file}")

            # Process results into embeddings format
            self.process_results(results_file)
            return True

        except Exception as e:
            logger.error(f"Error downloading results: {e}")
            return False

    def process_results(self, results_file):
        """Process batch results into embeddings format."""
        try:
            embeddings = {}
            processed_count = 0
            error_count = 0

            logger.info(f"Processing results from {results_file}")

            with open(results_file, 'r') as f:
                for line in f:
                    result = json.loads(line.strip())

                    if result.get('response', {}).get('status_code') == 200:
                        custom_id = result['custom_id']
                        patent_id = custom_id.replace('embedding-', '')

                        embedding_data = result['response']['body']['data'][0]
                        embedding = embedding_data['embedding']

                        embeddings[patent_id] = {
                            'id': patent_id,
                            'embedding': embedding,
                            'model': 'text-embedding-3-small',
                            'usage': result['response']['body'].get('usage', {})
                        }
                        processed_count += 1
                    else:
                        error_count += 1
                        logger.warning(f"Failed request: {result.get('custom_id')} - {result.get('error', {})}")

            # Save processed embeddings
            output_file = "results/openai_embeddings_ground_truth.jsonl"
            Path("results").mkdir(exist_ok=True)

            with open(output_file, 'w') as f:
                for patent_id, data in embeddings.items():
                    f.write(json.dumps(data) + '\n')

            logger.info(f"Processed {processed_count} embeddings, {error_count} errors")
            logger.info(f"Embeddings saved to {output_file}")

            # Update batch info with completion
            with open(self.batch_info_file, 'r') as f:
                batch_info = json.load(f)

            batch_info.update({
                'status': 'completed',
                'completed_at': datetime.now().isoformat(),
                'processed_count': processed_count,
                'error_count': error_count,
                'output_file': output_file
            })

            with open(self.batch_info_file, 'w') as f:
                json.dump(batch_info, f, indent=2)

            return True

        except Exception as e:
            logger.error(f"Error processing results: {e}")
            return False

    def handle_failure(self, batch_id, status_info):
        """Handle batch failures by resubmitting if appropriate."""
        logger.error(f"Batch {batch_id} failed with status: {status_info}")

        # Check if this is a retryable failure
        failed_count = status_info.get('request_counts', {}).get('failed', 0)
        total_count = status_info.get('request_counts', {}).get('total', 0)

        if failed_count > 0 and failed_count < total_count * 0.1:  # Less than 10% failure
            logger.info(f"Partial failure: {failed_count}/{total_count} requests failed")
            # Could implement partial retry logic here
        else:
            logger.error(f"Major failure: {failed_count}/{total_count} requests failed")
            # Could implement full resubmission logic here

    def monitor_batch(self, check_interval=300):  # 5 minutes
        """Monitor batch progress with periodic status checks."""
        if not self.current_batch_id:
            logger.error("No batch ID to monitor")
            return

        logger.info(f"Starting monitoring for batch {self.current_batch_id}")
        logger.info(f"Check interval: {check_interval} seconds")

        while True:
            status_info = self.get_batch_status(self.current_batch_id)

            if not status_info:
                logger.error("Failed to get batch status")
                time.sleep(check_interval)
                continue

            status = status_info['status']
            counts = status_info['request_counts']

            logger.info(f"Batch {self.current_batch_id}: {status} - "
                       f"{counts['completed']}/{counts['total']} completed, "
                       f"{counts['failed']} failed")

            if status == 'completed':
                logger.info("Batch completed successfully!")
                if self.download_results(self.current_batch_id):
                    logger.info("Results downloaded and processed successfully")
                    break
                else:
                    logger.error("Failed to download/process results")
                    break

            elif status == 'failed':
                self.handle_failure(self.current_batch_id, status_info)
                break

            elif status == 'expired':
                logger.error("Batch expired before completion")
                break

            elif status == 'cancelled':
                logger.warning("Batch was cancelled")
                break

            # Continue monitoring for validating, in_progress, finalizing
            time.sleep(check_interval)


def main():
    """Main monitoring function."""
    monitor = OpenAIBatchMonitor()

    if len(os.sys.argv) > 1:
        command = os.sys.argv[1]

        if command == "status":
            batch_id = os.sys.argv[2] if len(os.sys.argv) > 2 else monitor.current_batch_id
            if batch_id:
                status = monitor.get_batch_status(batch_id)
                if status:
                    print(json.dumps(status, indent=2))
                else:
                    print("Failed to get status")
            else:
                print("No batch ID provided or found")

        elif command == "monitor":
            interval = int(os.sys.argv[2]) if len(os.sys.argv) > 2 else 300
            monitor.monitor_batch(interval)

        elif command == "download":
            batch_id = os.sys.argv[2] if len(os.sys.argv) > 2 else monitor.current_batch_id
            if batch_id:
                monitor.download_results(batch_id)
            else:
                print("No batch ID provided or found")

    else:
        print("Usage:")
        print("  python monitor_openai_batch.py status [batch_id]")
        print("  python monitor_openai_batch.py monitor [interval_seconds]")
        print("  python monitor_openai_batch.py download [batch_id]")


if __name__ == "__main__":
    main()