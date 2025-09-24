"""Complete workflow for chunking strategy experiments."""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

from generate_embeddings_experimental import process_patents_experimental
from analyze_chunking_performance import ChunkingAnalyzer
from visualize_chunking_analysis import ChunkingVisualizer


def run_full_chunking_experiment(
    input_file: str,
    max_records: Optional[int] = None,
    model: str = "embeddinggemma",
    output_prefix: str = "chunking_experiment"
) -> None:
    """Run complete chunking experiment workflow."""
    
    print("🔬 COMPREHENSIVE CHUNKING STRATEGY EXPERIMENT")
    print("=" * 60)
    print(f"Input file: {input_file}")
    print(f"Max records: {max_records or 'All'}")
    print(f"Model: {model}")
    print(f"Output prefix: {output_prefix}")
    print()
    
    # Define output files
    experimental_data_file = f"{output_prefix}_embeddings.jsonl"
    analysis_results_file = f"{output_prefix}_analysis.json"
    report_file = f"{output_prefix}_report.md"
    
    start_time = time.time()
    
    # Step 1: Generate experimental embeddings
    print("📝 Step 1: Generating experimental embeddings...")
    print("-" * 40)
    
    if not Path(experimental_data_file).exists():
        process_patents_experimental(
            input_file=input_file,
            output_file=experimental_data_file,
            max_records=max_records,
            model=model
        )
        print(f"✅ Experimental embeddings saved to: {experimental_data_file}")
    else:
        print(f"⚠️ Experimental data file already exists: {experimental_data_file}")
        print("   Skipping embedding generation. Delete file to regenerate.")
    
    print()
    
    # Step 2: Analyze chunking performance
    print("📊 Step 2: Analyzing chunking performance...")
    print("-" * 40)
    
    analyzer = ChunkingAnalyzer(experimental_data_file)
    results = analyzer.run_comprehensive_analysis()
    
    # Save analysis results
    analyzer.save_results(analysis_results_file)
    
    # Generate report
    report = analyzer.generate_performance_report()
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ Analysis results saved to: {analysis_results_file}")
    print(f"✅ Performance report saved to: {report_file}")
    print()
    
    # Step 3: Generate visualizations
    print("📈 Step 3: Generating visualizations...")
    print("-" * 40)
    
    visualizer = ChunkingVisualizer(analysis_results_file)
    visualizer.generate_all_visualizations()
    print("✅ Visualizations generated")
    print()
    
    # Step 4: Summary
    total_time = time.time() - start_time
    print("🎉 EXPERIMENT COMPLETE!")
    print("=" * 60)
    print(f"Total execution time: {total_time/60:.1f} minutes")
    print()
    print("📁 Generated Files:")
    print(f"   📊 Experimental data: {experimental_data_file}")
    print(f"   📋 Analysis results: {analysis_results_file}")
    print(f"   📝 Performance report: {report_file}")
    print(f"   📈 Visualizations: chunking_analysis_visualizations/")
    print()
    
    # Display key findings
    clustering_results = results.get('clustering_performance', {})
    if clustering_results:
        best_strategy = None
        best_score = -1
        
        for strategy_name, strategy_data in clustering_results.items():
            if 'best_clustering' in strategy_data:
                score = strategy_data['best_clustering']['silhouette_score']
                if score > best_score:
                    best_score = score
                    best_strategy = strategy_name
        
        if best_strategy:
            print("🏆 KEY FINDINGS:")
            print(f"   Best performing strategy: {best_strategy}")
            print(f"   Silhouette score: {best_score:.3f}")
            
            quality = ("🟢 Excellent" if best_score > 0.7 else 
                      "🟡 Good" if best_score > 0.5 else 
                      "🟠 Fair" if best_score > 0.3 else "🔴 Poor")
            print(f"   Quality assessment: {quality}")
            print()
    
    print("📖 Next Steps:")
    print(f"   1. Review the report: {report_file}")
    print("   2. Open visualizations in your browser")
    print("   3. Use best performing strategy for production embeddings")
    print()


def create_experiment_summary(results_file: str) -> str:
    """Create a concise experiment summary."""
    
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    summary = []
    summary.append("# Chunking Strategy Experiment Summary")
    summary.append(f"\nGenerated: {results['dataset_info']['analysis_timestamp']}")
    summary.append(f"Patents analyzed: {results['dataset_info']['total_patents']}")
    summary.append(f"Strategies tested: {len(results['dataset_info']['strategies_analyzed'])}")
    
    # Top 3 strategies
    clustering_results = results.get('clustering_performance', {})
    strategy_scores = []
    
    for strategy_name, strategy_data in clustering_results.items():
        if 'best_clustering' in strategy_data:
            score = strategy_data['best_clustering']['silhouette_score']
            strategy_scores.append((strategy_name, score))
    
    strategy_scores.sort(key=lambda x: x[1], reverse=True)
    
    summary.append("\n## 🏆 Top 3 Strategies")
    for i, (strategy, score) in enumerate(strategy_scores[:3], 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
        summary.append(f"{medal} **{strategy}**: {score:.3f}")
    
    # Recommendations
    if strategy_scores:
        best_strategy, best_score = strategy_scores[0]
        summary.append(f"\n## 💡 Recommendation")
        summary.append(f"Use **{best_strategy}** for production (score: {best_score:.3f})")
        
        if best_score > 0.7:
            summary.append("✅ Excellent clustering quality - ready for deployment")
        elif best_score > 0.5:
            summary.append("✅ Good clustering quality - suitable for most use cases")
        else:
            summary.append("⚠️ Consider parameter tuning for better performance")
    
    return "\n".join(summary)


def main() -> None:
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive chunking strategy experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on first 100 patents for quick testing
  python run_chunking_experiment.py patent_abstracts.jsonl --max-records 100
  
  # Run on diverse dataset
  python run_chunking_experiment.py patent_abstracts_10k_diverse.jsonl --prefix diverse_experiment
  
  # Full analysis with custom model
  python run_chunking_experiment.py patent_abstracts.jsonl --model embeddinggemma --prefix full_analysis
        """
    )
    
    parser.add_argument("input_file", help="Input JSONL file with patent data")
    parser.add_argument("--max-records", type=int, help="Maximum number of records to process")
    parser.add_argument("--model", default="embeddinggemma", help="Embedding model to use")
    parser.add_argument("--prefix", default="chunking_experiment", help="Output file prefix")
    parser.add_argument("--summary-only", action="store_true", 
                       help="Generate summary from existing results (skip processing)")
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input_file).exists():
        print(f"Error: Input file {args.input_file} not found")
        return
    
    # Run experiment or generate summary
    if args.summary_only:
        results_file = f"{args.prefix}_analysis.json"
        if not Path(results_file).exists():
            print(f"Error: Results file {results_file} not found")
            print("Run the experiment first without --summary-only")
            return
        
        summary = create_experiment_summary(results_file)
        summary_file = f"{args.prefix}_summary.md"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(summary)
        print(f"\nSummary saved to: {summary_file}")
    else:
        run_full_chunking_experiment(
            input_file=args.input_file,
            max_records=args.max_records,
            model=args.model,
            output_prefix=args.prefix
        )


if __name__ == "__main__":
    main()