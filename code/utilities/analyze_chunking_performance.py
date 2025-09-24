"""Analyze and compare chunking strategy performance."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP


class ChunkingAnalyzer:
    """Comprehensive analysis of chunking strategy performance."""
    
    def __init__(self, experimental_data_file: str):
        self.data_file = experimental_data_file
        self.data = self.load_experimental_data()
        self.results = {}
        
    def load_experimental_data(self) -> List[Dict[str, Any]]:
        """Load experimental embedding data."""
        data = []
        with open(self.data_file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        
        print(f"Loaded experimental data for {len(data)} patents")
        return data
    
    def extract_embeddings_by_strategy(self) -> Dict[str, List[np.ndarray]]:
        """Extract embeddings for each strategy."""
        strategies = {}
        
        for patent in self.data:
            embeddings = patent.get('embeddings', {})
            
            # Original embeddings
            if 'original' in embeddings:
                if 'original' not in strategies:
                    strategies['original'] = []
                strategies['original'].append(np.array(embeddings['original']['embedding']))
            
            # Chunking strategy embeddings
            if 'chunking_strategies' in embeddings:
                for strategy_name, strategy_data in embeddings['chunking_strategies'].items():
                    if 'error' in strategy_data:
                        continue
                        
                    # For each aggregation method
                    aggregations = strategy_data.get('aggregations', {})
                    for agg_method, agg_embedding in aggregations.items():
                        key = f"{strategy_name}_{agg_method}"
                        if key not in strategies:
                            strategies[key] = []
                        strategies[key].append(np.array(agg_embedding))
        
        print(f"Extracted embeddings for {len(strategies)} strategies")
        for strategy, embeds in strategies.items():
            print(f"  {strategy}: {len(embeds)} embeddings")
        
        return strategies
    
    def calculate_clustering_performance(self, embeddings_dict: Dict[str, List[np.ndarray]]) -> Dict[str, Dict]:
        """Calculate clustering performance for each embedding strategy."""
        results = {}
        
        # Get classification labels for evaluation
        classifications = [patent.get('classification', 'unknown') for patent in self.data]
        
        print("\nEvaluating clustering performance...")
        
        for strategy_name, embeddings_list in embeddings_dict.items():
            if len(embeddings_list) < 10:  # Skip if too few samples
                continue
                
            print(f"  Analyzing {strategy_name}...")
            embeddings = np.array(embeddings_list)
            
            strategy_results = {
                'strategy': strategy_name,
                'num_samples': len(embeddings),
                'embedding_dim': embeddings.shape[1] if embeddings.ndim > 1 else 0
            }
            
            # K-means clustering
            kmeans_results = {}
            for k in [3, 5, 7, 10]:
                if k >= len(embeddings):
                    continue
                    
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans_labels = kmeans.fit_predict(embeddings)
                silhouette_avg = silhouette_score(embeddings, kmeans_labels)
                
                kmeans_results[f'kmeans_{k}'] = {
                    'n_clusters': k,
                    'silhouette_score': float(silhouette_avg),
                    'inertia': float(kmeans.inertia_)
                }
            
            # DBSCAN clustering
            dbscan_results = {}
            for eps in [0.3, 0.5, 0.7]:
                dbscan = DBSCAN(eps=eps, min_samples=5)
                dbscan_labels = dbscan.fit_predict(embeddings)
                
                n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
                n_noise = list(dbscan_labels).count(-1)
                
                if n_clusters > 1:
                    non_noise_mask = dbscan_labels != -1
                    if np.sum(non_noise_mask) > 1:
                        silhouette_avg = silhouette_score(
                            embeddings[non_noise_mask], 
                            dbscan_labels[non_noise_mask]
                        )
                    else:
                        silhouette_avg = -1
                else:
                    silhouette_avg = -1
                
                dbscan_results[f'dbscan_{eps}'] = {
                    'eps': eps,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'silhouette_score': float(silhouette_avg)
                }
            
            strategy_results['kmeans'] = kmeans_results
            strategy_results['dbscan'] = dbscan_results
            
            # Best performing clustering
            all_scores = []
            for method_results in [kmeans_results, dbscan_results]:
                for result in method_results.values():
                    score = result.get('silhouette_score', -1)
                    if score > -1:
                        all_scores.append((score, result))
            
            if all_scores:
                best_score, best_config = max(all_scores, key=lambda x: x[0])
                strategy_results['best_clustering'] = {
                    'silhouette_score': best_score,
                    'config': best_config
                }
            
            results[strategy_name] = strategy_results
        
        return results
    
    def calculate_semantic_similarity_preservation(self, embeddings_dict: Dict[str, List[np.ndarray]]) -> Dict[str, float]:
        """Calculate how well chunked embeddings preserve semantic similarity."""
        print("\nEvaluating semantic similarity preservation...")
        
        if 'original' not in embeddings_dict:
            print("  No original embeddings found for comparison")
            return {}
        
        original_embeddings = np.array(embeddings_dict['original'])
        
        # Calculate pairwise similarities for original embeddings
        original_similarities = cosine_similarity(original_embeddings)
        
        results = {}
        
        for strategy_name, embeddings_list in embeddings_dict.items():
            if strategy_name == 'original' or len(embeddings_list) != len(original_embeddings):
                continue
            
            print(f"  Comparing {strategy_name} to original...")
            
            strategy_embeddings = np.array(embeddings_list)
            strategy_similarities = cosine_similarity(strategy_embeddings)
            
            # Calculate correlation between similarity matrices
            original_flat = original_similarities[np.triu_indices(len(original_similarities), k=1)]
            strategy_flat = strategy_similarities[np.triu_indices(len(strategy_similarities), k=1)]
            
            correlation = np.corrcoef(original_flat, strategy_flat)[0, 1]
            results[strategy_name] = float(correlation)
        
        return results
    
    def analyze_processing_efficiency(self) -> Dict[str, Any]:
        """Analyze processing time and efficiency metrics."""
        print("\nAnalyzing processing efficiency...")
        
        efficiency_stats = {}
        
        # Overall processing times
        total_times = []
        chunking_times = []
        
        for patent in self.data:
            embeddings = patent.get('embeddings', {})
            
            # Original processing time
            if 'original' in embeddings:
                original_time = embeddings['original'].get('processing_time', 0)
                total_times.append(original_time)
            
            # Chunking processing times
            if 'chunking_strategies' in embeddings:
                for strategy_name, strategy_data in embeddings['chunking_strategies'].items():
                    if 'error' in strategy_data:
                        continue
                    
                    strategy_time = strategy_data.get('total_processing_time', 0)
                    chunking_times.append((strategy_name, strategy_time))
        
        # Calculate statistics
        if total_times:
            efficiency_stats['original_embedding'] = {
                'mean_time': np.mean(total_times),
                'median_time': np.median(total_times),
                'std_time': np.std(total_times)
            }
        
        # Group chunking times by strategy
        strategy_times = defaultdict(list)
        for strategy_name, processing_time in chunking_times:
            strategy_times[strategy_name].append(processing_time)
        
        efficiency_stats['chunking_strategies'] = {}
        for strategy_name, times in strategy_times.items():
            efficiency_stats['chunking_strategies'][strategy_name] = {
                'mean_time': np.mean(times),
                'median_time': np.median(times),
                'std_time': np.std(times),
                'samples': len(times)
            }
        
        return efficiency_stats
    
    def analyze_chunk_statistics(self) -> Dict[str, Any]:
        """Analyze chunking patterns and statistics."""
        print("\nAnalyzing chunk statistics...")
        
        chunk_stats = {}
        
        for patent in self.data:
            embeddings = patent.get('embeddings', {})
            
            if 'chunking_strategies' not in embeddings:
                continue
            
            for strategy_name, strategy_data in embeddings['chunking_strategies'].items():
                if 'error' in strategy_data:
                    continue
                
                if strategy_name not in chunk_stats:
                    chunk_stats[strategy_name] = {
                        'num_chunks': [],
                        'chunk_sizes': [],
                        'token_counts': []
                    }
                
                num_chunks = strategy_data.get('num_chunks', 0)
                chunk_stats[strategy_name]['num_chunks'].append(num_chunks)
                
                # Analyze individual chunks
                chunks = strategy_data.get('chunks', [])
                for chunk in chunks:
                    chunk_size = len(chunk.get('text', ''))
                    token_count = chunk.get('token_count', 0)
                    
                    chunk_stats[strategy_name]['chunk_sizes'].append(chunk_size)
                    chunk_stats[strategy_name]['token_counts'].append(token_count)
        
        # Calculate summary statistics
        summary_stats = {}
        for strategy_name, stats in chunk_stats.items():
            summary_stats[strategy_name] = {
                'avg_num_chunks': np.mean(stats['num_chunks']),
                'avg_chunk_size': np.mean(stats['chunk_sizes']),
                'avg_token_count': np.mean(stats['token_counts']),
                'chunk_size_std': np.std(stats['chunk_sizes']),
                'samples': len(stats['num_chunks'])
            }
        
        return summary_stats
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run complete analysis and return results."""
        print("Starting comprehensive chunking analysis...")
        print("=" * 60)
        
        # Extract embeddings by strategy
        embeddings_dict = self.extract_embeddings_by_strategy()
        
        # Run all analyses
        analysis_results = {
            'clustering_performance': self.calculate_clustering_performance(embeddings_dict),
            'similarity_preservation': self.calculate_semantic_similarity_preservation(embeddings_dict),
            'processing_efficiency': self.analyze_processing_efficiency(),
            'chunk_statistics': self.analyze_chunk_statistics(),
            'dataset_info': {
                'total_patents': len(self.data),
                'strategies_analyzed': list(embeddings_dict.keys()),
                'analysis_timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        self.results = analysis_results
        return analysis_results
    
    def generate_performance_report(self) -> str:
        """Generate a comprehensive performance report."""
        if not self.results:
            self.run_comprehensive_analysis()
        
        report = []
        report.append("# Chunking Strategy Performance Analysis Report")
        report.append(f"\nGenerated on: {self.results['dataset_info']['analysis_timestamp']}")
        report.append(f"Dataset: {self.results['dataset_info']['total_patents']} patents")
        report.append("\n" + "=" * 60)
        
        # Clustering Performance Summary
        report.append("\n## 🎯 Clustering Performance Rankings")
        
        clustering_results = self.results['clustering_performance']
        strategy_scores = []
        
        for strategy_name, strategy_data in clustering_results.items():
            if 'best_clustering' in strategy_data:
                score = strategy_data['best_clustering']['silhouette_score']
                config = strategy_data['best_clustering']['config']
                strategy_scores.append((strategy_name, score, config))
        
        # Sort by silhouette score
        strategy_scores.sort(key=lambda x: x[1], reverse=True)
        
        report.append(f"\n{'Rank':<5} {'Strategy':<30} {'Silhouette Score':<15} {'Best Config'}")
        report.append("-" * 80)
        
        for i, (strategy, score, config) in enumerate(strategy_scores[:10], 1):
            rank_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:2d}"
            quality = "🟢 Excellent" if score > 0.7 else "🟡 Good" if score > 0.5 else "🟠 Fair" if score > 0.3 else "🔴 Poor"
            
            config_str = f"{config.get('n_clusters', config.get('eps', 'N/A'))}"
            report.append(f"{rank_emoji:<5} {strategy:<30} {score:<15.3f} {config_str} {quality}")
        
        # Similarity Preservation
        if 'similarity_preservation' in self.results:
            report.append("\n## 📊 Semantic Similarity Preservation")
            
            similarity_results = self.results['similarity_preservation']
            sorted_similarities = sorted(similarity_results.items(), key=lambda x: x[1], reverse=True)
            
            report.append(f"\n{'Strategy':<35} {'Correlation with Original':<25} {'Quality'}")
            report.append("-" * 80)
            
            for strategy, correlation in sorted_similarities:
                quality = "🟢 Excellent" if correlation > 0.9 else "🟡 Good" if correlation > 0.8 else "🟠 Fair" if correlation > 0.7 else "🔴 Poor"
                report.append(f"{strategy:<35} {correlation:<25.3f} {quality}")
        
        # Processing Efficiency
        if 'processing_efficiency' in self.results:
            report.append("\n## ⚡ Processing Efficiency")
            
            efficiency = self.results['processing_efficiency']
            
            if 'original_embedding' in efficiency:
                orig = efficiency['original_embedding']
                report.append(f"\n**Original Embedding**: {orig['mean_time']:.2f}s avg (±{orig['std_time']:.2f}s)")
            
            if 'chunking_strategies' in efficiency:
                report.append("\n**Chunking Strategies**:")
                
                strategies = efficiency['chunking_strategies']
                sorted_strategies = sorted(strategies.items(), key=lambda x: x[1]['mean_time'])
                
                for strategy, stats in sorted_strategies:
                    overhead = stats['mean_time'] / orig['mean_time'] if 'original_embedding' in efficiency else 1
                    report.append(f"  {strategy}: {stats['mean_time']:.2f}s avg ({overhead:.1f}x overhead)")
        
        # Chunk Statistics
        if 'chunk_statistics' in self.results:
            report.append("\n## 📏 Chunk Statistics")
            
            chunk_stats = self.results['chunk_statistics']
            
            report.append(f"\n{'Strategy':<25} {'Avg Chunks':<12} {'Avg Size':<12} {'Avg Tokens':<12}")
            report.append("-" * 70)
            
            for strategy, stats in chunk_stats.items():
                report.append(
                    f"{strategy:<25} {stats['avg_num_chunks']:<12.1f} "
                    f"{stats['avg_chunk_size']:<12.0f} {stats['avg_token_count']:<12.0f}"
                )
        
        # Recommendations
        report.append("\n## 🎯 Recommendations")
        
        if strategy_scores:
            best_strategy, best_score, best_config = strategy_scores[0]
            report.append(f"\n**Best Overall Performance**: {best_strategy}")
            report.append(f"- Silhouette Score: {best_score:.3f}")
            report.append(f"- Configuration: {best_config}")
            
            if best_score > 0.7:
                report.append("- ✅ Excellent clustering quality")
            elif best_score > 0.5:
                report.append("- ✅ Good clustering quality")
            else:
                report.append("- ⚠️ Consider parameter tuning")
        
        if 'similarity_preservation' in self.results and similarity_results:
            best_similarity_strategy, best_correlation = max(similarity_results.items(), key=lambda x: x[1])
            report.append(f"\n**Best Similarity Preservation**: {best_similarity_strategy}")
            report.append(f"- Correlation: {best_correlation:.3f}")
        
        return "\n".join(report)
    
    def save_results(self, output_file: str) -> None:
        """Save analysis results to JSON file."""
        if not self.results:
            self.run_comprehensive_analysis()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Analysis results saved to: {output_file}")


def main() -> None:
    """Main function for chunking analysis."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_chunking_performance.py <experimental_data_file> [output_file]")
        print("Example: python analyze_chunking_performance.py patent_experimental_embeddings.jsonl")
        return
    
    experimental_file = sys.argv[1]
    
    if not Path(experimental_file).exists():
        print(f"Error: Experimental data file {experimental_file} not found")
        return
    
    # Run analysis
    analyzer = ChunkingAnalyzer(experimental_file)
    results = analyzer.run_comprehensive_analysis()
    
    # Generate report
    report = analyzer.generate_performance_report()
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(report)
    
    # Save results if output file specified
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
        analyzer.save_results(output_file)
        
        # Save report as markdown
        report_file = output_file.replace('.json', '_report.md')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report saved to: {report_file}")


if __name__ == "__main__":
    main()