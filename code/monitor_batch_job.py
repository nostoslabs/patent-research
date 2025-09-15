"""Monitor Gemini batch job progress with real-time updates."""

import time
import json
import logging
import argparse
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

from gemini_batch_client import GeminiBatchClient, BatchJob

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BatchJobMonitor:
    """Monitor batch job progress with real-time updates."""
    
    def __init__(self, client: GeminiBatchClient):
        self.client = client
        self.start_time = datetime.now()
    
    def monitor_job(self, 
                   job_id: str,
                   check_interval: int = 60,
                   auto_download: bool = False,
                   download_path: str = None) -> Optional[str]:
        """Monitor job progress until completion."""
        logger.info(f"Starting monitoring for job: {job_id}")
        logger.info(f"Check interval: {check_interval} seconds")
        logger.info(f"Auto-download: {auto_download}")
        
        last_state = None
        last_percentage = -1
        check_count = 0
        
        try:
            while True:
                check_count += 1
                current_time = datetime.now()
                elapsed = current_time - self.start_time
                
                try:
                    job = self.client.get_batch_status(job_id)
                    
                    # Check if state changed
                    if job.state != last_state:
                        logger.info(f"State changed: {last_state} -> {job.state}")
                        last_state = job.state
                    
                    # Check if progress changed significantly
                    if abs(job.completion_percentage - last_percentage) >= 1.0:
                        self._print_progress_update(job, elapsed, check_count)
                        last_percentage = job.completion_percentage
                    
                    # Check for completion
                    if job.state in ["STATE_SUCCEEDED", "STATE_COMPLETED"]:
                        logger.info("🎉 Batch job completed successfully!")
                        self._print_final_summary(job, elapsed)
                        
                        if auto_download:
                            return self._auto_download_results(job_id, download_path)
                        else:
                            logger.info("Use 'download' command to retrieve results")
                            return None
                    
                    elif job.state == "STATE_FAILED":
                        logger.error("❌ Batch job failed!")
                        self._print_final_summary(job, elapsed)
                        return None
                    
                    elif job.state == "STATE_CANCELLED":
                        logger.warning("⚠️ Batch job was cancelled")
                        self._print_final_summary(job, elapsed)
                        return None
                    
                    # Continue monitoring
                    time.sleep(check_interval)
                    
                except KeyboardInterrupt:
                    logger.info("\\n👋 Monitoring interrupted by user")
                    logger.info("Job is still running. Use 'status' command to check progress.")
                    return None
                    
                except Exception as e:
                    logger.error(f"Error checking job status: {e}")
                    logger.info(f"Retrying in {check_interval} seconds...")
                    time.sleep(check_interval)
                    continue
        
        except Exception as e:
            logger.error(f"Monitoring failed: {e}")
            return None
    
    def _print_progress_update(self, job: BatchJob, elapsed: timedelta, check_count: int):
        """Print formatted progress update."""
        hours, remainder = divmod(elapsed.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        elapsed_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        
        # Estimate remaining time
        if job.completion_percentage > 0:
            total_estimated = elapsed.total_seconds() / (job.completion_percentage / 100)
            remaining_seconds = total_estimated - elapsed.total_seconds()
            if remaining_seconds > 0:
                rem_hours, rem_remainder = divmod(remaining_seconds, 3600)
                rem_minutes, _ = divmod(rem_remainder, 60)
                eta_str = f"{int(rem_hours):02d}:{int(rem_minutes):02d}"
            else:
                eta_str = "Soon"
        else:
            eta_str = "Unknown"
        
        # Create progress bar
        bar_width = 30
        filled = int(bar_width * job.completion_percentage / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        print(f"\\r📊 Progress: [{bar}] {job.completion_percentage:5.1f}% | "
              f"Completed: {job.completed_count:,}/{job.request_count:,} | "
              f"Failed: {job.failed_count:,} | "
              f"Elapsed: {elapsed_str} | ETA: {eta_str}", end="", flush=True)
        
        # Print full update every 10 checks
        if check_count % 10 == 0:
            print()  # New line
            logger.info(f"Status Update #{check_count}")
            logger.info(f"  State: {job.state}")
            logger.info(f"  Progress: {job.completion_percentage:.1f}%")
            logger.info(f"  Completed: {job.completed_count:,}/{job.request_count:,}")
            logger.info(f"  Failed: {job.failed_count:,}")
            logger.info(f"  Elapsed time: {elapsed_str}")
            logger.info(f"  Estimated remaining: {eta_str}")
    
    def _print_final_summary(self, job: BatchJob, elapsed: timedelta):
        """Print final job summary."""
        print()  # New line after progress bar
        logger.info("=" * 60)
        logger.info("BATCH JOB SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Job ID: {job.job_id}")
        logger.info(f"Final state: {job.state}")
        logger.info(f"Total requests: {job.request_count:,}")
        logger.info(f"Completed: {job.completed_count:,}")
        logger.info(f"Failed: {job.failed_count:,}")
        logger.info(f"Success rate: {job.completed_count/job.request_count*100:.1f}%")
        
        hours, remainder = divmod(elapsed.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info(f"Total time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
        
        if job.request_count > 0:
            avg_time = elapsed.total_seconds() / job.request_count
            logger.info(f"Average time per request: {avg_time:.2f} seconds")
        
        logger.info("=" * 60)
    
    def _auto_download_results(self, job_id: str, download_path: str = None) -> str:
        """Automatically download results when job completes."""
        if not download_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            download_path = f"batch_results_{timestamp}.json"
        
        logger.info(f"Auto-downloading results to: {download_path}")
        
        try:
            result_file = self.client.download_results(job_id, download_path)
            logger.info(f"✅ Results downloaded successfully: {result_file}")
            return result_file
        except Exception as e:
            logger.error(f"❌ Failed to download results: {e}")
            return None
    
    def print_job_info(self, job_id: str):
        """Print current job information."""
        try:
            job = self.client.get_batch_status(job_id)
            
            print("📋 Batch Job Information")
            print("-" * 40)
            print(f"Job ID: {job.job_id}")
            print(f"State: {job.state}")
            print(f"Created: {job.create_time}")
            print(f"Updated: {job.update_time}")
            print(f"Total requests: {job.request_count:,}")
            print(f"Completed: {job.completed_count:,}")
            print(f"Failed: {job.failed_count:,}")
            print(f"Progress: {job.completion_percentage:.1f}%")
            
            if job.completion_percentage > 0:
                print(f"Success rate: {job.completed_count/(job.completed_count + job.failed_count)*100:.1f}%")
            
            print("-" * 40)
            
        except Exception as e:
            logger.error(f"Failed to get job info: {e}")


def main():
    """CLI interface for batch job monitoring."""
    parser = argparse.ArgumentParser(description="Monitor Gemini batch job progress")
    parser.add_argument("job_id", help="Batch job ID to monitor")
    parser.add_argument("--interval", type=int, default=60, 
                       help="Check interval in seconds (default: 60)")
    parser.add_argument("--auto-download", action="store_true",
                       help="Automatically download results when complete")
    parser.add_argument("--download-path", 
                       help="Path for downloaded results (auto-generated if not provided)")
    parser.add_argument("--info-only", action="store_true",
                       help="Just print job info and exit")
    
    args = parser.parse_args()
    
    try:
        client = GeminiBatchClient()
        monitor = BatchJobMonitor(client)
        
        if args.info_only:
            monitor.print_job_info(args.job_id)
        else:
            result_file = monitor.monitor_job(
                args.job_id,
                check_interval=args.interval,
                auto_download=args.auto_download,
                download_path=args.download_path
            )
            
            if result_file:
                logger.info(f"Results available at: {result_file}")
                logger.info("Use 'process' command to convert to ground truth format")
    
    except Exception as e:
        logger.error(f"Monitoring failed: {e}")
        raise


if __name__ == "__main__":
    main()