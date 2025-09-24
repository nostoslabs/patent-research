#!/usr/bin/env python3
"""Monitor all running experiments with comprehensive status updates."""

import time
import json
import subprocess
import os
from datetime import datetime
from pathlib import Path
from experiment_tracker import ExperimentTracker

def format_table_row(columns, widths):
    """Format a table row with proper alignment."""
    formatted_cols = []
    for i, (col, width) in enumerate(zip(columns, widths)):
        if i == 0:  # Left align first column
            formatted_cols.append(f"{str(col):<{width}}")
        elif isinstance(col, (int, float)) or str(col).replace('.','').isdigit():  # Right align numbers
            formatted_cols.append(f"{str(col):>{width}}")
        else:  # Center align others
            formatted_cols.append(f"{str(col):^{width}}")
    return " │ ".join(formatted_cols)

def format_size(size_bytes):
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0B"
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f}TB"

def get_recent_files():
    """Get recent embedding and result files with formatted display."""
    files_info = []
    
    # Look for embedding and result files in data directory
    data_dir = Path("data")
    if data_dir.exists():
        patterns = ["*embeddings.jsonl", "*batch_results.json"]
        for pattern in patterns:
            for file_path in data_dir.glob(pattern):
                if file_path.is_file():
                    stat = file_path.stat()
                    files_info.append({
                        'name': file_path.name,
                        'size': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime),
                        'type': 'embeddings' if 'embeddings' in file_path.name else 'results'
                    })
    
    # Sort by modification time (newest first)
    files_info.sort(key=lambda x: x['modified'], reverse=True)
    return files_info[:8]  # Return top 8

def main():
    """Main monitoring loop."""
    tracker = ExperimentTracker()
    
    print("🔬 Multi-Model Experiment Monitor")
    print("=" * 100)
    print("Press Ctrl+C to stop monitoring\n")
    
    try:
        while True:
            # Clear screen (works on most terminals)
            print("\033[2J\033[H")
            
            # Header with timestamp
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("🔬 PATENT EMBEDDING EXPERIMENT MONITOR")
            print("=" * 100)
            print(f"📅 Last Updated: {current_time} │ 🔄 Refreshing every 30 seconds")
            print()
            
            # Get summary data
            summary = tracker.get_progress_summary()
            
            # Status Overview
            print("📊 EXPERIMENT STATUS OVERVIEW")
            print("─" * 100)
            status_widths = [12, 8, 8, 8, 8, 8]
            status_headers = ["Status", "Pending", "Running", "Complete", "Failed", "Total"]
            print(" │ " + format_table_row(status_headers, status_widths))
            print("─┼─" + "─┼─".join("─" * w for w in status_widths))
            
            status_counts = summary["by_status"]
            status_row = [
                "Count",
                status_counts.get("pending", 0),
                status_counts.get("running", 0), 
                status_counts.get("completed", 0),
                status_counts.get("failed", 0),
                summary["total_experiments"]
            ]
            print(" │ " + format_table_row(status_row, status_widths))
            print()
            
            # Running Experiments Table
            if summary["running_experiments"]:
                print("🔄 CURRENTLY RUNNING EXPERIMENTS")
                print("─" * 100)
                
                run_widths = [28, 20, 10, 10, 12, 12]
                run_headers = ["Experiment ID", "Model", "Progress", "ETA", "Rate/min", "Elapsed"]
                
                print("   " + format_table_row(run_headers, run_widths))
                print("─" + "─┼─".join("─" * (w + 2) for w in run_widths))
                
                for exp in summary["running_experiments"]:
                    progress = f"{exp['progress_percent']:.1f}%"
                    eta = f"{exp['eta_minutes']:.1f}m" if exp['eta_minutes'] > 0 else "∞"
                    rate = f"{exp['rate_per_minute']:.1f}"
                    elapsed = f"{exp['elapsed_minutes']:.1f}m"
                    
                    row = [
                        exp['experiment_id'][:26] + ".." if len(exp['experiment_id']) > 28 else exp['experiment_id'],
                        exp['model'][:18] + ".." if len(exp['model']) > 20 else exp['model'],
                        progress,
                        eta,
                        rate,
                        elapsed
                    ]
                    print("   " + format_table_row(row, run_widths))
                print()
            else:
                print("🔄 CURRENTLY RUNNING EXPERIMENTS")
                print("─" * 100)
                print("   No experiments currently running")
                print()
            
            # Recent Completions Table
            if summary["recent_completions"]:
                print("✅ RECENT COMPLETIONS (Last 24 Hours)")
                print("─" * 100)
                
                comp_widths = [28, 20, 12, 12, 15]
                comp_headers = ["Experiment ID", "Model", "Completed", "Duration", "Status"]
                
                print("   " + format_table_row(comp_headers, comp_widths))
                print("─" + "─┼─".join("─" * (w + 2) for w in comp_widths))
                
                for exp in summary["recent_completions"]:
                    completed_ago = f"{exp['completed_ago_hours']:.1f}h ago"
                    duration = f"{exp['total_time_minutes']:.1f}m" if exp['total_time_minutes'] else "N/A"
                    
                    row = [
                        exp['experiment_id'][:26] + ".." if len(exp['experiment_id']) > 28 else exp['experiment_id'],
                        exp['model'][:18] + ".." if len(exp['model']) > 20 else exp['model'],
                        completed_ago,
                        duration,
                        "✅ Success"
                    ]
                    print("   " + format_table_row(row, comp_widths))
                print()
            else:
                print("✅ RECENT COMPLETIONS (Last 24 Hours)")
                print("─" * 100)  
                print("   No recent completions")
                print()
            
            # Recent Files Table
            recent_files = get_recent_files()
            print("📁 RECENT OUTPUT FILES")
            print("─" * 100)
            
            if recent_files:
                file_widths = [45, 8, 12, 25]
                file_headers = ["File Name", "Size", "Type", "Modified"]
                
                print("   " + format_table_row(file_headers, file_widths))
                print("─" + "─┼─".join("─" * (w + 2) for w in file_widths))
                
                for file_info in recent_files:
                    file_name = file_info['name']
                    if len(file_name) > 43:
                        file_name = file_name[:40] + "..."
                    
                    size = format_size(file_info['size'])
                    file_type = "🧬 Embeddings" if file_info['type'] == 'embeddings' else "📊 Results"
                    modified = file_info['modified'].strftime('%m/%d %H:%M')
                    
                    row = [file_name, size, file_type, modified]
                    print("   " + format_table_row(row, file_widths))
            else:
                print("   No output files found yet")
            
            print()
            print("─" * 100)
            print(f"🕒 Next refresh in 30 seconds... │ Press Ctrl+C to stop")
            
            # Wait before next update
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped")
        print("=" * 100)

if __name__ == "__main__":
    main()