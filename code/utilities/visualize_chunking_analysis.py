"""Visualization tools for chunking strategy analysis."""

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class ChunkingVisualizer:
    """Create visualizations for chunking strategy analysis."""
    
    def __init__(self, results_file: str):
        self.results_file = results_file
        self.results = self.load_results()
        self.output_dir = Path("chunking_analysis_visualizations")
        self.output_dir.mkdir(exist_ok=True)
        
    def load_results(self) -> Dict[str, Any]:
        """Load analysis results from JSON file."""
        with open(self.results_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def create_clustering_performance_chart(self) -> None:
        """Create clustering performance comparison chart."""
        clustering_results = self.results.get('clustering_performance', {})
        
        # Extract data for visualization
        strategies = []
        scores = []
        configs = []
        
        for strategy_name, strategy_data in clustering_results.items():
            if 'best_clustering' in strategy_data:
                strategies.append(strategy_name.replace('_', ' ').title())
                scores.append(strategy_data['best_clustering']['silhouette_score'])
                config = strategy_data['best_clustering']['config']
                configs.append(f"{config.get('n_clusters', config.get('eps', 'N/A'))}")
        
        # Sort by score
        data = list(zip(strategies, scores, configs))
        data.sort(key=lambda x: x[1], reverse=True)
        strategies, scores, configs = zip(*data) if data else ([], [], [])
        
        # Create Plotly bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=scores,
                y=strategies,
                orientation='h',
                text=[f"{score:.3f}" for score in scores],
                textposition='inside',
                marker=dict(
                    color=scores,
                    colorscale='RdYlGn',
                    cmin=0,
                    cmax=1
                )
            )
        ])
        
        fig.update_layout(
            title="Chunking Strategy Clustering Performance",
            xaxis_title="Silhouette Score",
            yaxis_title="Strategy",
            height=max(400, len(strategies) * 30),
            showlegend=False
        )
        
        # Add quality thresholds
        fig.add_vline(x=0.7, line_dash="dash", line_color="green", 
                     annotation_text="Excellent (>0.7)")
        fig.add_vline(x=0.5, line_dash="dash", line_color="orange", 
                     annotation_text="Good (>0.5)")
        fig.add_vline(x=0.3, line_dash="dash", line_color="red", 
                     annotation_text="Fair (>0.3)")
        
        output_file = self.output_dir / "clustering_performance.html"
        fig.write_html(str(output_file))
        print(f"Clustering performance chart saved to: {output_file}")
    
    def create_similarity_preservation_chart(self) -> None:
        """Create semantic similarity preservation comparison chart."""
        similarity_results = self.results.get('similarity_preservation', {})
        
        if not similarity_results:
            print("No similarity preservation data found")
            return
        
        strategies = [s.replace('_', ' ').title() for s in similarity_results.keys()]
        correlations = list(similarity_results.values())
        
        # Sort by correlation
        data = list(zip(strategies, correlations))
        data.sort(key=lambda x: x[1], reverse=True)
        strategies, correlations = zip(*data)
        
        # Create Plotly bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=correlations,
                y=strategies,
                orientation='h',
                text=[f"{corr:.3f}" for corr in correlations],
                textposition='inside',
                marker=dict(
                    color=correlations,
                    colorscale='RdYlGn',
                    cmin=0,
                    cmax=1
                )
            )
        ])
        
        fig.update_layout(
            title="Semantic Similarity Preservation",
            xaxis_title="Correlation with Original Embeddings",
            yaxis_title="Strategy",
            height=max(400, len(strategies) * 30),
            showlegend=False
        )
        
        # Add quality thresholds
        fig.add_vline(x=0.9, line_dash="dash", line_color="green", 
                     annotation_text="Excellent (>0.9)")
        fig.add_vline(x=0.8, line_dash="dash", line_color="orange", 
                     annotation_text="Good (>0.8)")
        fig.add_vline(x=0.7, line_dash="dash", line_color="red", 
                     annotation_text="Fair (>0.7)")
        
        output_file = self.output_dir / "similarity_preservation.html"
        fig.write_html(str(output_file))
        print(f"Similarity preservation chart saved to: {output_file}")
    
    def create_processing_efficiency_chart(self) -> None:
        """Create processing efficiency comparison chart."""
        efficiency = self.results.get('processing_efficiency', {})
        
        if 'chunking_strategies' not in efficiency:
            print("No processing efficiency data found")
            return
        
        strategies = []
        mean_times = []
        std_times = []
        
        # Add original embedding as baseline
        if 'original_embedding' in efficiency:
            strategies.append('Original (Baseline)')
            mean_times.append(efficiency['original_embedding']['mean_time'])
            std_times.append(efficiency['original_embedding']['std_time'])
        
        # Add chunking strategies
        for strategy_name, stats in efficiency['chunking_strategies'].items():
            strategies.append(strategy_name.replace('_', ' ').title())
            mean_times.append(stats['mean_time'])
            std_times.append(stats['std_time'])
        
        # Create Plotly bar chart with error bars
        fig = go.Figure(data=[
            go.Bar(
                x=strategies,
                y=mean_times,
                error_y=dict(type='data', array=std_times),
                text=[f"{time:.2f}s" for time in mean_times],
                textposition='outside',
                marker_color='lightblue'
            )
        ])
        
        fig.update_layout(
            title="Processing Time by Strategy",
            xaxis_title="Strategy",
            yaxis_title="Processing Time (seconds)",
            height=500,
            xaxis_tickangle=-45
        )
        
        output_file = self.output_dir / "processing_efficiency.html"
        fig.write_html(str(output_file))
        print(f"Processing efficiency chart saved to: {output_file}")
    
    def create_chunk_statistics_chart(self) -> None:
        """Create chunk statistics visualization."""
        chunk_stats = self.results.get('chunk_statistics', {})
        
        if not chunk_stats:
            print("No chunk statistics data found")
            return
        
        # Prepare data
        strategies = []
        avg_chunks = []
        avg_sizes = []
        avg_tokens = []
        
        for strategy_name, stats in chunk_stats.items():
            strategies.append(strategy_name.replace('_', ' ').title())
            avg_chunks.append(stats['avg_num_chunks'])
            avg_sizes.append(stats['avg_chunk_size'])
            avg_tokens.append(stats['avg_token_count'])
        
        # Create subplot with three charts
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=("Average Number of Chunks", "Average Chunk Size (chars)", "Average Token Count"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Average number of chunks
        fig.add_trace(
            go.Bar(x=strategies, y=avg_chunks, name="Avg Chunks", marker_color='lightcoral'),
            row=1, col=1
        )
        
        # Average chunk size
        fig.add_trace(
            go.Bar(x=strategies, y=avg_sizes, name="Avg Size", marker_color='lightblue'),
            row=1, col=2
        )
        
        # Average token count
        fig.add_trace(
            go.Bar(x=strategies, y=avg_tokens, name="Avg Tokens", marker_color='lightgreen'),
            row=1, col=3
        )
        
        fig.update_layout(
            title="Chunk Statistics by Strategy",
            height=500,
            showlegend=False
        )
        
        # Rotate x-axis labels
        fig.update_xaxes(tickangle=-45)
        
        output_file = self.output_dir / "chunk_statistics.html"
        fig.write_html(str(output_file))
        print(f"Chunk statistics chart saved to: {output_file}")
    
    def create_performance_heatmap(self) -> None:
        """Create performance heatmap comparing all metrics."""
        clustering_results = self.results.get('clustering_performance', {})
        similarity_results = self.results.get('similarity_preservation', {})
        efficiency_results = self.results.get('processing_efficiency', {}).get('chunking_strategies', {})
        
        # Get common strategies
        common_strategies = set()
        if clustering_results:
            common_strategies.update(clustering_results.keys())
        if similarity_results:
            common_strategies.update(similarity_results.keys())
        if efficiency_results:
            common_strategies.update(efficiency_results.keys())
        
        # Prepare data matrix
        strategies_list = sorted(list(common_strategies))
        metrics = ['Clustering Score', 'Similarity Preservation', 'Processing Efficiency']
        
        data_matrix = []
        
        for strategy in strategies_list:
            row = []
            
            # Clustering score (silhouette score)
            if strategy in clustering_results and 'best_clustering' in clustering_results[strategy]:
                clustering_score = clustering_results[strategy]['best_clustering']['silhouette_score']
            else:
                clustering_score = 0
            row.append(clustering_score)
            
            # Similarity preservation (correlation)
            similarity_score = similarity_results.get(strategy, 0)
            row.append(similarity_score)
            
            # Processing efficiency (inverse of time, normalized)
            if strategy in efficiency_results:
                proc_time = efficiency_results[strategy]['mean_time']
                # Normalize to 0-1 scale (inverse and scale)
                max_time = max([stats['mean_time'] for stats in efficiency_results.values()])
                efficiency_score = 1 - (proc_time / max_time)
            else:
                efficiency_score = 0
            row.append(efficiency_score)
            
            data_matrix.append(row)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=data_matrix,
            x=metrics,
            y=[s.replace('_', ' ').title() for s in strategies_list],
            colorscale='RdYlGn',
            text=[[f"{val:.3f}" for val in row] for row in data_matrix],
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title="Performance Heatmap: All Metrics",
            height=max(400, len(strategies_list) * 30),
            width=600
        )
        
        output_file = self.output_dir / "performance_heatmap.html"
        fig.write_html(str(output_file))
        print(f"Performance heatmap saved to: {output_file}")
    
    def create_strategy_comparison_dashboard(self) -> None:
        """Create comprehensive dashboard with all visualizations."""
        # Create a dashboard with all charts
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Clustering Performance",
                "Similarity Preservation", 
                "Processing Efficiency",
                "Performance Summary"
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "heatmap"}]
            ]
        )
        
        # Get data for each chart
        clustering_results = self.results.get('clustering_performance', {})
        similarity_results = self.results.get('similarity_preservation', {})
        efficiency_results = self.results.get('processing_efficiency', {}).get('chunking_strategies', {})
        
        # Clustering performance
        if clustering_results:
            strategies = []
            scores = []
            for strategy_name, strategy_data in clustering_results.items():
                if 'best_clustering' in strategy_data:
                    strategies.append(strategy_name.replace('_', ' ')[:20])  # Truncate long names
                    scores.append(strategy_data['best_clustering']['silhouette_score'])
            
            fig.add_trace(
                go.Bar(x=strategies, y=scores, name="Clustering", marker_color='lightcoral'),
                row=1, col=1
            )
        
        # Similarity preservation
        if similarity_results:
            strategies = [s.replace('_', ' ')[:20] for s in similarity_results.keys()]
            correlations = list(similarity_results.values())
            
            fig.add_trace(
                go.Bar(x=strategies, y=correlations, name="Similarity", marker_color='lightblue'),
                row=1, col=2
            )
        
        # Processing efficiency
        if efficiency_results:
            strategies = [s.replace('_', ' ')[:20] for s in efficiency_results.keys()]
            times = [stats['mean_time'] for stats in efficiency_results.values()]
            
            fig.add_trace(
                go.Bar(x=strategies, y=times, name="Efficiency", marker_color='lightgreen'),
                row=2, col=1
            )
        
        fig.update_layout(
            title="Chunking Strategy Analysis Dashboard",
            height=800,
            showlegend=False
        )
        
        # Update axis labels
        fig.update_xaxes(tickangle=-45)
        fig.update_yaxes(title_text="Silhouette Score", row=1, col=1)
        fig.update_yaxes(title_text="Correlation", row=1, col=2)
        fig.update_yaxes(title_text="Time (seconds)", row=2, col=1)
        
        output_file = self.output_dir / "analysis_dashboard.html"
        fig.write_html(str(output_file))
        print(f"Analysis dashboard saved to: {output_file}")
    
    def generate_all_visualizations(self) -> None:
        """Generate all visualization charts."""
        print("Generating chunking analysis visualizations...")
        print("=" * 50)
        
        self.create_clustering_performance_chart()
        self.create_similarity_preservation_chart()
        self.create_processing_efficiency_chart()
        self.create_chunk_statistics_chart()
        self.create_performance_heatmap()
        self.create_strategy_comparison_dashboard()
        
        print("\n" + "=" * 50)
        print(f"All visualizations saved to: {self.output_dir}")
        print("\nGenerated files:")
        for file_path in self.output_dir.glob("*.html"):
            print(f"  - {file_path.name}")


def main() -> None:
    """Main function for visualization generation."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python visualize_chunking_analysis.py <results_file.json>")
        print("Example: python visualize_chunking_analysis.py chunking_analysis_results.json")
        return
    
    results_file = sys.argv[1]
    
    if not Path(results_file).exists():
        print(f"Error: Results file {results_file} not found")
        return
    
    # Generate visualizations
    visualizer = ChunkingVisualizer(results_file)
    visualizer.generate_all_visualizations()


if __name__ == "__main__":
    main()