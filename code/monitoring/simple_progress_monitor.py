"""Simple progress monitor that watches bash job output."""

import time
import re
import subprocess
from datetime import datetime, timedelta

def get_job_progress(bash_id: str):
    """Get progress from bash job output."""
    try:
        # Use your BashOutput equivalent or read from bash job
        # For now, let's parse the latest progress from a simple command
        result = subprocess.run([
            'python', '-c', 
            f'''
import subprocess
result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
lines = result.stdout.split("\\n")
ground_truth_lines = [line for line in lines if "ground_truth_generator.py" in line and "{bash_id}" not in line]
if ground_truth_lines:
    print("RUNNING")
else:
    print("NOT_FOUND")
'''
        ], capture_output=True, text=True)
        
        # Check if job is still running
        if "RUNNING" in result.stdout:
            # Check for partial files to get current count
            result = subprocess.run(['ls', '-1', 'ground_truth_partial_*.jsonl'], 
                                  capture_output=True, text=True, cwd='.')
            
            if result.returncode == 0:
                files = result.stdout.strip().split('\n')
                if files and files[0]:
                    # Extract highest number from filenames
                    numbers = []
                    for f in files:
                        match = re.search(r'ground_truth_partial_(\d+)\.jsonl', f)
                        if match:
                            numbers.append(int(match.group(1)))
                    
                    if numbers:
                        return max(numbers), "RUNNING"
            
            return 0, "RUNNING"
        else:
            return 0, "COMPLETED_OR_NOT_FOUND"
    
    except Exception as e:
        return 0, f"ERROR: {e}"

def format_progress(current: int, total: int = 10000):
    """Format progress display."""
    percentage = (current / total) * 100
    bar_width = 40
    filled = int(bar_width * percentage / 100)
    bar = "█" * filled + "░" * (bar_width - filled)
    return f"[{bar}] {percentage:5.1f}% ({current:,}/{total:,})"

def monitor_progress(check_interval: int = 30):
    """Monitor progress with simple display."""
    print("🔍 Ground Truth Generation Progress Monitor")
    print("=" * 60)
    
    start_time = datetime.now()
    last_count = 0
    last_time = start_time
    
    try:
        while True:
            current_count, status = get_job_progress("55dded")
            current_time = datetime.now()
            
            if status == "COMPLETED_OR_NOT_FOUND":
                print(f"\n✅ Job completed or not found. Final count: {current_count:,}")
                break
            
            elif "ERROR" in status:
                print(f"\n❌ Error monitoring: {status}")
                break
            
            # Calculate rate and ETA
            elapsed = current_time - start_time
            if current_count > last_count and elapsed.total_seconds() > 0:
                recent_elapsed = current_time - last_time
                recent_rate = (current_count - last_count) / recent_elapsed.total_seconds()
                
                if recent_rate > 0:
                    remaining = 10000 - current_count
                    eta_seconds = remaining / recent_rate
                    eta = timedelta(seconds=eta_seconds)
                    eta_str = str(eta).split('.')[0]  # Remove microseconds
                else:
                    eta_str = "Unknown"
            else:
                eta_str = "Calculating..."
            
            # Display progress
            progress_bar = format_progress(current_count)
            elapsed_str = str(elapsed).split('.')[0]
            
            print(f"\r{progress_bar} | Elapsed: {elapsed_str} | ETA: {eta_str}", 
                  end="", flush=True)
            
            # Update for next iteration
            last_count = current_count
            last_time = current_time
            
            time.sleep(check_interval)
    
    except KeyboardInterrupt:
        print(f"\n\n👋 Monitoring stopped. Last seen: {current_count:,}/10,000")

if __name__ == "__main__":
    import sys
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    monitor_progress(interval)