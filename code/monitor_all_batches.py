#!/usr/bin/env python3
"""Monitor all OpenAI batches and automatically complete the research."""

import os
import json
import time
import glob
import requests
import yaml
from pathlib import Path
from openai import OpenAI
from datetime import datetime

class BatchMonitor:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.output_file = "results/openai_embeddings_final.jsonl"

        # Initialize Pushover
        self.pushover_enabled = False
        try:
            config_path = Path.home() / ".ntfy.yml"
            if config_path.exists():
                config = yaml.safe_load(config_path.read_text())
                self.user_key = config['pushover']['user_key']
                self.app_token = 'ag3vr23p62zk7f947h17c6fjgyydyz'  # From your script
                self.pushover_enabled = True
                print("📱 Pushover notifications enabled")
            else:
                print("📱 Pushover notifications disabled (no .ntfy.yml found)")
        except Exception as e:
            print(f"📱 Pushover setup failed: {e}")
            self.pushover_enabled = False

    def send_pushover_notification(self, message, priority=0):
        """Send Pushover notification."""
        if not self.pushover_enabled:
            return False

        try:
            response = requests.post(
                "https://api.pushover.net/1/messages.json",
                data={
                    'token': self.app_token,
                    'user': self.user_key,
                    'message': message,
                    'priority': priority,
                    'title': '🧬 Patent Research Update'
                },
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            print(f"📱 Pushover notification failed: {e}")
            return False

    def get_all_batch_ids(self):
        """Extract batch IDs from all result files."""
        result_files = glob.glob("openai_batch_results_*.jsonl")
        batch_ids = []

        for file in result_files:
            # Extract batch ID from filename
            if "batch_" in file:
                batch_id = file.split("batch_")[1].split(".")[0]
                batch_ids.append("batch_" + batch_id)

        return batch_ids

    def check_batch_status(self, batch_id):
        """Check status of a single batch."""
        try:
            batch = self.client.batches.retrieve(batch_id)
            return {
                'id': batch_id,
                'status': batch.status,
                'total': batch.request_counts.total,
                'completed': batch.request_counts.completed,
                'failed': batch.request_counts.failed,
                'output_file_id': batch.output_file_id
            }
        except Exception as e:
            return {'id': batch_id, 'status': 'error', 'error': str(e)}

    def download_batch_results(self, batch_id, output_file_id):
        """Download results for a completed batch."""
        try:
            print(f"Downloading results for {batch_id}...")
            file_response = self.client.files.content(output_file_id)

            # Save raw results
            raw_file = f"openai_batch_results_{batch_id}.jsonl"
            with open(raw_file, 'w') as f:
                f.write(file_response.content.decode('utf-8'))

            return raw_file
        except Exception as e:
            print(f"Error downloading {batch_id}: {e}")
            return None

    def consolidate_all_results(self):
        """Consolidate all downloaded results."""
        result_files = glob.glob("openai_batch_results_*.jsonl")

        if not result_files:
            print("No result files found")
            return 0

        Path("results").mkdir(exist_ok=True)
        total_embeddings = 0
        all_patent_ids = set()

        print(f"Consolidating {len(result_files)} result files...")

        with open(self.output_file, 'w') as outfile:
            for result_file in sorted(result_files):
                with open(result_file, 'r') as infile:
                    for line in infile:
                        if not line.strip():
                            continue

                        try:
                            result = json.loads(line)

                            if result.get('response', {}).get('status_code') == 200:
                                custom_id = result['custom_id']
                                patent_id = custom_id.replace('embedding-', '')

                                # Avoid duplicates
                                if patent_id in all_patent_ids:
                                    continue

                                all_patent_ids.add(patent_id)

                                embedding_data = result['response']['body']['data'][0]
                                embedding = embedding_data['embedding']

                                embedding_entry = {
                                    'id': patent_id,
                                    'embedding': embedding,
                                    'model': 'text-embedding-3-small'
                                }

                                outfile.write(json.dumps(embedding_entry) + '\n')
                                total_embeddings += 1

                        except (json.JSONDecodeError, KeyError):
                            continue

        print(f"✅ Consolidated {total_embeddings} embeddings to {self.output_file}")

        # Send notification
        if total_embeddings > 1000:
            self.send_pushover_notification(
                f"✅ OpenAI embeddings consolidated: {total_embeddings:,} embeddings ready for benchmark!"
            )

        return total_embeddings

    def run_final_benchmark(self):
        """Run the comprehensive benchmark with OpenAI embeddings."""
        print("🚀 Running final benchmark with OpenAI embeddings...")

        try:
            import subprocess
            result = subprocess.run(
                ["uv", "run", "python", "code/comprehensive_embedding_benchmark.py"],
                capture_output=True, text=True, timeout=600
            )

            if result.returncode == 0:
                print("✅ Final benchmark completed successfully!")
                print("📊 Results saved to results/COMPREHENSIVE_BENCHMARK_REPORT.md")

                # Send completion notification
                self.send_pushover_notification(
                    "🎯 PATENT RESEARCH COMPLETE! ✅ Final benchmark finished. OpenAI vs nomic-embed-text results ready!",
                    priority=1  # High priority for completion
                )

                return True
            else:
                print(f"❌ Benchmark failed: {result.stderr}")

                # Send failure notification
                self.send_pushover_notification(
                    f"❌ Benchmark failed: {result.stderr[:100]}...",
                    priority=1
                )

                return False

        except Exception as e:
            print(f"❌ Error running benchmark: {e}")
            return False

    def monitor_all_batches(self, check_interval=60):
        """Monitor all batches until completion."""
        print("🔍 Starting comprehensive batch monitoring...")
        print(f"Check interval: {check_interval} seconds ({check_interval//60} minutes)")

        # Get all active batches
        active_batches = set()

        # Try to get current batch from info file
        if Path("openai_batch_info.json").exists():
            with open("openai_batch_info.json", 'r') as f:
                info = json.load(f)
                if info.get('batch_id'):
                    active_batches.add(info['batch_id'])

        # Also check for other batches from result files
        existing_batch_ids = self.get_all_batch_ids()
        active_batches.update(existing_batch_ids)

        # If no batches found, try to find from chunk files
        if not active_batches:
            print("No active batches found. Checking for recent batches...")
            # List recent batches
            try:
                batches = self.client.batches.list(limit=20)
                for batch in batches.data:
                    if batch.status in ['validating', 'in_progress', 'finalizing']:
                        active_batches.add(batch.id)
            except Exception as e:
                print(f"Error listing batches: {e}")

        if not active_batches:
            print("No active batches found to monitor!")
            return

        print(f"Monitoring {len(active_batches)} batches:")
        for batch_id in active_batches:
            print(f"  - {batch_id}")

        completed_batches = set()

        while len(completed_batches) < len(active_batches):
            print(f"\n⏰ {datetime.now().strftime('%H:%M:%S')} - Checking batch status...")

            for batch_id in active_batches:
                if batch_id in completed_batches:
                    continue

                status_info = self.check_batch_status(batch_id)
                status = status_info.get('status', 'unknown')

                if status == 'completed':
                    print(f"✅ {batch_id}: COMPLETED ({status_info.get('completed', 0)}/{status_info.get('total', 0)})")

                    # Download results
                    if status_info.get('output_file_id'):
                        self.download_batch_results(batch_id, status_info['output_file_id'])

                    completed_batches.add(batch_id)

                elif status in ['failed', 'expired', 'cancelled']:
                    print(f"❌ {batch_id}: {status.upper()}")
                    completed_batches.add(batch_id)  # Don't wait for failed batches

                elif status in ['validating', 'in_progress', 'finalizing']:
                    total = status_info.get('total', 0)
                    completed = status_info.get('completed', 0)
                    print(f"🔄 {batch_id}: {status.upper()} ({completed}/{total})")

                else:
                    print(f"❓ {batch_id}: {status}")

            if len(completed_batches) < len(active_batches):
                print(f"⏳ Waiting {check_interval} seconds ({check_interval//60} minutes) before next check...")
                time.sleep(check_interval)

        print(f"\n🎉 All {len(active_batches)} batches completed!")

        # Send batch completion notification
        completed_count = len([b for b in completed_batches if b])
        self.send_pushover_notification(
            f"🎉 All OpenAI batches completed! {completed_count}/{len(active_batches)} batches finished. Starting consolidation..."
        )

        # Consolidate results
        total_embeddings = self.consolidate_all_results()

        if total_embeddings > 1000:  # Reasonable threshold
            print(f"📊 Ready for benchmark with {total_embeddings} embeddings")

            # Run final benchmark
            if self.run_final_benchmark():
                print("\n🎯 RESEARCH COMPLETE!")
                print("📁 Check results/COMPREHENSIVE_BENCHMARK_REPORT.md for final results")
            else:
                print("\n⚠️  Benchmark failed - check embeddings and try manually")
                self.send_pushover_notification(
                    "⚠️ Final benchmark failed - manual intervention needed",
                    priority=1
                )
        else:
            print(f"⚠️  Only {total_embeddings} embeddings found - may need manual intervention")

def main():
    """Main monitoring function."""
    monitor = BatchMonitor()

    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not set!")
        return

    print("=" * 60)
    print("🤖 OpenAI Batch Monitor - Patent Research Project")
    print("=" * 60)

    # Send startup notification
    monitor.send_pushover_notification("🤖 Patent research monitoring started - checking OpenAI batches every 15 minutes")

    try:
        monitor.monitor_all_batches(check_interval=900)  # Check every 15 minutes (900 seconds)
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Monitoring error: {e}")

if __name__ == "__main__":
    main()