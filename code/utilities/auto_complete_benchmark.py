#!/usr/bin/env python3
"""Automatically complete the comprehensive benchmark when OpenAI embeddings are ready."""

import os
import json
import time
import logging
from pathlib import Path
from monitor_openai_batch import OpenAIBatchMonitor
from comprehensive_embedding_benchmark import EmbeddingBenchmark

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_openai_embeddings_ready():
    """Check if OpenAI embeddings are available."""
    openai_files = [
        "results/openai_embeddings_ground_truth.jsonl",
        "openai_embeddings_ground_truth.jsonl"
    ]

    for file_path in openai_files:
        if Path(file_path).exists():
            # Check if file has content
            try:
                with open(file_path, 'r') as f:
                    first_line = f.readline()
                    if first_line:
                        data = json.loads(first_line.strip())
                        if 'embedding' in data:
                            logger.info(f"OpenAI embeddings found in {file_path}")
                            return True
            except:
                continue

    return False


def run_final_benchmark():
    """Run the comprehensive benchmark with all models including OpenAI."""
    logger.info("Running final comprehensive benchmark with OpenAI embeddings")

    try:
        benchmark = EmbeddingBenchmark()
        results = benchmark.run_benchmark()

        if results:
            logger.info("Final benchmark completed successfully!")

            # Save final results with timestamp
            timestamp = int(time.time())
            final_results_file = f"results/FINAL_benchmark_results_{timestamp}.json"

            with open(final_results_file, 'w') as f:
                json.dump(results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)

            logger.info(f"Final results saved to {final_results_file}")

            # Update research report
            update_research_report(results)

            return True
        else:
            logger.error("Benchmark failed to produce results")
            return False

    except Exception as e:
        logger.error(f"Error running final benchmark: {e}")
        return False


def update_research_report(results):
    """Update the research report with final OpenAI results."""
    try:
        report_file = "RESEARCH_REPORT.md"

        if not Path(report_file).exists():
            logger.warning("Research report not found, skipping update")
            return

        # Get OpenAI results
        correlations = results.get('correlations', {})
        openai_results = correlations.get('text-embedding-3-small', {})

        if not openai_results:
            logger.warning("No OpenAI results found in benchmark")
            return

        # Read current report
        with open(report_file, 'r') as f:
            content = f.read()

        # Find the results table and update it
        # This is a simple approach - in production you'd want more sophisticated parsing
        if 'text-embedding-3-small' not in content:
            # Add OpenAI results to the table
            table_marker = "| **bge-m3** |"
            if table_marker in content:
                openai_row = (f"| **text-embedding-3-small** | {openai_results['pearson_r']:.3f} | "
                             f"{openai_results['spearman_r']:.3f} | {openai_results['pearson_p']:.2e} | "
                             f"{openai_results['n_pairs']:,} | {openai_results['mean_embedding_sim']:.3f} | "
                             f"{openai_results['mean_llm_score']:.3f} |")

                content = content.replace(table_marker, f"{table_marker}\n{openai_row}")

                # Update key findings if needed
                if openai_results['pearson_r'] > 0.516:  # Better than nomic-embed-text
                    content = content.replace(
                        "**Best Performing Model**: nomic-embed-text (r = 0.516)",
                        f"**Best Performing Model**: text-embedding-3-small (r = {openai_results['pearson_r']:.3f})"
                    )

                # Add timestamp
                content += f"\n\n*Report updated with OpenAI results: {time.strftime('%Y-%m-%d %H:%M:%S')}*"

                with open(report_file, 'w') as f:
                    f.write(content)

                logger.info("Research report updated with OpenAI results")

    except Exception as e:
        logger.error(f"Error updating research report: {e}")


def main():
    """Main auto-completion function."""
    logger.info("Starting auto-completion monitoring")

    monitor = OpenAIBatchMonitor()
    check_interval = 600  # 10 minutes

    while True:
        # Check if OpenAI embeddings are already available
        if check_openai_embeddings_ready():
            logger.info("OpenAI embeddings are ready! Running final benchmark...")
            if run_final_benchmark():
                logger.info("Auto-completion successful!")
                break
            else:
                logger.error("Final benchmark failed")
                break

        # Check batch status
        if monitor.current_batch_id:
            status_info = monitor.get_batch_status(monitor.current_batch_id)

            if status_info:
                status = status_info['status']
                logger.info(f"Batch status: {status}")

                if status == 'completed':
                    logger.info("Batch completed! Downloading results...")
                    if monitor.download_results(monitor.current_batch_id):
                        logger.info("Results downloaded successfully")
                        # Check again for embeddings
                        continue
                    else:
                        logger.error("Failed to download results")
                        break

                elif status in ['failed', 'expired', 'cancelled']:
                    logger.error(f"Batch {status} - auto-completion stopped")
                    break

        logger.info(f"Waiting {check_interval} seconds before next check...")
        time.sleep(check_interval)


if __name__ == "__main__":
    main()