"""Monitor ground truth generation progress in real-time."""

import time
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

class GroundTruthMonitor:
    """Monitor progress of ground truth generation."""
    
    def __init__(self, output_file: str = "ground_truth_10k.jsonl"):
        self.output_file = output_file
        self.partial_files = self._find_partial_files()
        self.start_time = datetime.now()
        self.last_count = 0
        self.last_check_time = datetime.now()
        
    def _find_partial_files(self) -> list:
        """Find all partial ground truth files."""
        pattern = "ground_truth_partial_*.jsonl"
        partial_files = list(Path(".").glob(pattern))
        return sorted(partial_files, key=lambda x: int(x.stem.split('_')[-1]))
    
    def get_current_progress(self) -> Dict[str, Any]:
        """Get current progress from partial files."""
        # Update partial files list
        self.partial_files = self._find_partial_files()
        
        total_processed = 0
        successful = 0
        failed = 0
        
        # Count from partial files
        for partial_file in self.partial_files:
            try:
                with open(partial_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            total_processed += 1
                            try:
                                data = json.loads(line.strip())
                                if data.get('success', False):
                                    successful += 1
                                else:
                                    failed += 1
                            except json.JSONDecodeError:
                                failed += 1
            except FileNotFoundError:
                continue
        
        # Check final output file
        if os.path.exists(self.output_file):
            with open(self.output_file, 'r', encoding='utf-8') as f:
                final_count = sum(1 for line in f if line.strip())
                total_processed = max(total_processed, final_count)
                # Assume final file only contains successful results
                successful = final_count
                failed = 0
        
        return {
            'total_processed': total_processed,
            'successful': successful,
            'failed': failed,
            'latest_partial_file': self.partial_files[-1] if self.partial_files else None
        }
    
    def calculate_eta(self, current_count: int, target: int = 10000) -> Optional[str]:
        """Calculate estimated time to completion."""
        now = datetime.now()
        elapsed = now - self.start_time
        
        if current_count <= 0 or elapsed.total_seconds() < 60:
            return "Calculating..."
        
        # Calculate rate based on progress since last check
        time_since_last = (now - self.last_check_time).total_seconds()
        if time_since_last > 0:
            recent_rate = (current_count - self.last_count) / time_since_last
        else:
            recent_rate = current_count / elapsed.total_seconds()
        
        self.last_count = current_count
        self.last_check_time = now
        
        if recent_rate <= 0:
            return "Unknown"
        
        remaining = target - current_count
        remaining_seconds = remaining / recent_rate
        
        if remaining_seconds < 3600:
            return f"{remaining_seconds/60:.0f} minutes"
        else:
            hours = remaining_seconds / 3600
            return f"{hours:.1f} hours"
    
    def format_progress_bar(self, current: int, total: int = 10000, width: int = 40) -> str:
        """Create a visual progress bar."""
        if total == 0:
            percentage = 0
        else:
            percentage = min(100, (current / total) * 100)
        
        filled = int(width * percentage / 100)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}] {percentage:5.1f}%"
    
    def print_status(self, progress: Dict[str, Any]):
        """Print current status with formatting."""
        current = progress['total_processed']
        successful = progress['successful']
        failed = progress['failed']
        
        elapsed = datetime.now() - self.start_time
        hours, remainder = divmod(elapsed.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        elapsed_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        
        eta = self.calculate_eta(current)
        
        progress_bar = self.format_progress_bar(current)
        
        # Calculate rate
        rate = current / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
        
        print(f"\r{progress_bar} {current:,}/10,000 | "
              f"Success: {successful:,} | Failed: {failed:,} | "
              f"Rate: {rate*60:.1f}/min | "
              f"Elapsed: {elapsed_str} | ETA: {eta}", 
              end="", flush=True)
    
    def monitor(self, check_interval: int = 30, target: int = 10000):
        """Monitor progress continuously."""
        print("🔍 Ground Truth Generation Monitor")
        print(f"Target: {target:,} pairs")
        print(f"Output file: {self.output_file}")
        print(f"Check interval: {check_interval} seconds")
        print("=" * 80)
        
        try:
            iteration = 0
            while True:
                iteration += 1
                progress = self.get_current_progress()
                
                # Print status line
                self.print_status(progress)
                
                # Print detailed update every 10 iterations
                if iteration % 10 == 0:
                    print()  # New line
                    now = datetime.now()
                    print(f"\n📊 Status Update #{iteration} - {now.strftime('%H:%M:%S')}")
                    print(f"   Total processed: {progress['total_processed']:,}")
                    print(f"   Successful: {progress['successful']:,}")
                    print(f"   Failed: {progress['failed']:,}")
                    if progress['successful'] > 0:
                        success_rate = progress['successful'] / progress['total_processed'] * 100
                        print(f"   Success rate: {success_rate:.1f}%")
                    
                    if progress['latest_partial_file']:
                        print(f"   Latest file: {progress['latest_partial_file']}")
                
                # Check if completed
                if progress['total_processed'] >= target:
                    print()  # New line
                    print("\n🎉 Ground truth generation completed!")
                    print(f"Final count: {progress['total_processed']:,}")
                    print(f"Successful: {progress['successful']:,}")
                    print(f"Failed: {progress['failed']:,}")
                    elapsed = datetime.now() - self.start_time
                    print(f"Total time: {elapsed}")
                    break
                
                time.sleep(check_interval)
        
        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped by user")
            progress = self.get_current_progress()
            print(f"Current progress: {progress['total_processed']:,}/10,000")
        
        except Exception as e:
            print(f"\n❌ Monitoring error: {e}")


def main():
    """CLI interface for monitoring."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor ground truth generation progress")
    parser.add_argument("--output", default="ground_truth_10k.jsonl", 
                       help="Output file to monitor")
    parser.add_argument("--interval", type=int, default=30,
                       help="Check interval in seconds")
    parser.add_argument("--target", type=int, default=10000,
                       help="Target number of pairs")
    parser.add_argument("--once", action="store_true",
                       help="Check once and exit")
    
    args = parser.parse_args()
    
    monitor = GroundTruthMonitor(args.output)
    
    if args.once:
        progress = monitor.get_current_progress()
        monitor.print_status(progress)
        print()
    else:
        monitor.monitor(args.interval, args.target)


if __name__ == "__main__":
    main()