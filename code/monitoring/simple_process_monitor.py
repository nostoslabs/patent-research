#!/usr/bin/env python3
"""Simple monitor for background processes and recent files."""

import time
import subprocess
import os
from datetime import datetime
from pathlib import Path

def format_table_row(columns, widths):
    """Format a table row with proper alignment."""
    formatted_cols = []
    for i, (col, width) in enumerate(zip(columns, widths)):
        if i == 0:  # Left align first column
            formatted_cols.append(f"{str(col):<{width}}")
        elif isinstance(col, (int, float)) or str(col).replace('.','').replace('-','').isdigit():  # Right align numbers
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

def get_python_processes():
    """Get running Python processes related to our experiments."""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        
        python_procs = []
        for line in lines[1:]:  # Skip header
            fields = line.split()
            if len(fields) >= 11 and 'python' in fields[10]:
                cmd = ' '.join(fields[10:])
                # Filter for our experiment scripts
                if any(script in cmd for script in [
                    'run_multimodel_experiments.py', 
                    'ground_truth_generator.py',
                    'generate_all_embeddings.py',
                    'download_large_diverse_patents.py'
                ]):
                    python_procs.append({
                        'pid': fields[1],
                        'cpu': fields[2],
                        'mem': fields[3],
                        'time': fields[9],
                        'command': cmd[:70] + "..." if len(cmd) > 70 else cmd
                    })
        
        return python_procs
    except Exception as e:
        return []

def get_recent_files():
    """Get recent embedding and result files."""
    files_info = []
    
    # Look for files in data directory
    data_dir = Path("data")
    if data_dir.exists():
        patterns = ["*embeddings.jsonl", "*batch_results.json", "*ground_truth*.jsonl"]
        for pattern in patterns:
            for file_path in data_dir.glob(pattern):
                if file_path.is_file():
                    stat = file_path.stat()
                    files_info.append({
                        'name': file_path.name,
                        'size': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime),
                        'type': 'embeddings' if 'embeddings' in file_path.name else 
                               'ground_truth' if 'ground_truth' in file_path.name else 'results'
                    })
    
    # Sort by modification time (newest first)  
    files_info.sort(key=lambda x: x['modified'], reverse=True)
    return files_info[:10]  # Return top 10

def main():
    """Main monitoring loop."""
    print("🔬 Simple Process Monitor")
    print("=" * 100)
    print("Press Ctrl+C to stop monitoring\n")
    
    try:
        while True:
            # Clear screen (works on most terminals)
            print("\033[2J\033[H")
            
            # Header with timestamp
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("🔬 PATENT RESEARCH PROCESS MONITOR")
            print("=" * 100)
            print(f"📅 Last Updated: {current_time} │ 🔄 Refreshing every 15 seconds")
            print()
            
            # Running Python Processes
            processes = get_python_processes()
            print("🐍 RUNNING PYTHON PROCESSES")
            print("─" * 100)
            
            if processes:
                proc_widths = [8, 6, 6, 10, 70]
                proc_headers = ["PID", "CPU%", "MEM%", "TIME", "Command"]
                
                print("   " + format_table_row(proc_headers, proc_widths))
                print("─" + "─┼─".join("─" * (w + 2) for w in proc_widths))
                
                for proc in processes:
                    row = [
                        proc['pid'],
                        proc['cpu'],
                        proc['mem'],
                        proc['time'],
                        proc['command']
                    ]
                    print("   " + format_table_row(row, proc_widths))
            else:
                print("   No Python experiment processes currently running")
            
            print()
            
            # Recent Files
            recent_files = get_recent_files()
            print("📁 RECENT OUTPUT FILES")
            print("─" * 100)
            
            if recent_files:
                file_widths = [45, 10, 15, 20]
                file_headers = ["File Name", "Size", "Type", "Modified"]
                
                print("   " + format_table_row(file_headers, file_widths))
                print("─" + "─┼─".join("─" * (w + 2) for w in file_widths))
                
                for file_info in recent_files:
                    file_name = file_info['name']
                    if len(file_name) > 43:
                        file_name = file_name[:40] + "..."
                    
                    size = format_size(file_info['size'])
                    
                    file_type = {
                        'embeddings': '🧬 Embeddings',
                        'ground_truth': '🎯 Ground Truth',
                        'results': '📊 Results'
                    }.get(file_info['type'], '📄 Other')
                    
                    modified = file_info['modified'].strftime('%m/%d %H:%M')
                    
                    row = [file_name, size, file_type, modified]
                    print("   " + format_table_row(row, file_widths))
            else:
                print("   No recent output files found")
            
            print()
            print("─" * 100)
            print("🕒 Next refresh in 15 seconds... │ Press Ctrl+C to stop")
            
            # Wait before next update
            time.sleep(15)
            
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped")
        print("=" * 100)

if __name__ == "__main__":
    main()