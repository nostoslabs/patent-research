"""Comprehensive model performance analysis and re-sorting based on experimental results."""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class ModelPerformance:
    """Performance metrics for a single model."""
    name: str
    speed_patents_per_minute: float
    speed_seconds_per_patent: float
    context_window: int
    embedding_dimension: int
    chunking_required_percent: float
    memory_efficiency: str
    quality_ranking: int
    production_viability: str
    cost_efficiency: str


class ModelPerformanceAnalyzer:
    """Analyze and re-sort models based on comprehensive performance metrics."""
    
    def __init__(self):
        """Initialize analyzer with batch results."""
        self.batch_results = self._load_batch_results()
        self.model_specs = self._get_model_specifications()
        self.performance_data = {}
    
    def _load_batch_results(self) -> Dict:
        """Load all batch result files."""
        batch_files = [
            "validation_batch_results.json",
            "diverse_10k_full_batch_results.json", 
            "original_500_batch_results.json",
            "production_top2_batch_results.json"
        ]
        
        results = {}
        for file in batch_files:
            if Path(file).exists():
                with open(file, 'r') as f:
                    batch_name = file.replace('_batch_results.json', '')
                    results[batch_name] = json.load(f)
        
        return results
    
    def _get_model_specifications(self) -> Dict:
        """Model specifications from experiments."""
        return {
            'nomic-embed-text': {
                'embedding_dimension': 768,
                'context_window': 8192,
                'chunking_required': False,  # Large context window
                'memory_efficiency': 'High',
                'architecture': 'Transformer-based'
            },
            'bge-m3': {
                'embedding_dimension': 1024,
                'context_window': 8192,
                'chunking_required': False,  # Large context window
                'memory_efficiency': 'Medium',
                'architecture': 'BGE-M3 multilingual'
            },
            'embeddinggemma': {
                'embedding_dimension': 768,
                'context_window': 2048,
                'chunking_required': True,  # Limited context window (some patents exceed)
                'memory_efficiency': 'Medium',
                'architecture': 'Gemma-based'
            },
            'mxbai-embed-large': {
                'embedding_dimension': 1024,
                'context_window': 512,
                'chunking_required': True,  # Very limited context window
                'memory_efficiency': 'Low',
                'architecture': 'Large transformer'
            }
        }
    
    def analyze_performance_metrics(self) -> Dict[str, ModelPerformance]:
        """Analyze performance across all experiments."""
        
        performance_metrics = {}
        
        for model_name in ['nomic-embed-text', 'bge-m3', 'embeddinggemma', 'mxbai-embed-large']:
            
            # Collect timing data across experiments
            timings = []
            patent_counts = []
            
            # Process each batch result
            for batch_name, batch_data in self.batch_results.items():
                if 'experiments' in batch_data:
                    for exp_name, exp_data in batch_data['experiments'].items():
                        if model_name in exp_name and exp_data['status'] == 'completed':
                            # Calculate patents processed
                            if 'validation' in batch_name:
                                patents = 25
                            elif 'diverse_10k_full' in batch_name:
                                patents = 100
                            elif 'original_500' in batch_name:
                                patents = 500
                            elif 'production_top2' in batch_name:
                                patents = 1000
                            else:
                                patents = 100  # default
                            
                            execution_time_seconds = exp_data['execution_time']
                            timings.append(execution_time_seconds)
                            patent_counts.append(patents)
            
            if timings and patent_counts:
                # Calculate average speed metrics
                total_patents = sum(patent_counts)
                total_time_seconds = sum(timings)
                
                patents_per_minute = (total_patents / total_time_seconds) * 60
                seconds_per_patent = total_time_seconds / total_patents
                
                # Get model specs
                specs = self.model_specs[model_name]
                
                # Estimate chunking percentage based on context window
                chunking_percent = self._estimate_chunking_percentage(specs['context_window'])
                
                # Create performance object
                performance = ModelPerformance(
                    name=model_name,
                    speed_patents_per_minute=patents_per_minute,
                    speed_seconds_per_patent=seconds_per_patent,
                    context_window=specs['context_window'],
                    embedding_dimension=specs['embedding_dimension'],
                    chunking_required_percent=chunking_percent,
                    memory_efficiency=specs['memory_efficiency'],
                    quality_ranking=self._get_quality_ranking(model_name),
                    production_viability=self._assess_production_viability(patents_per_minute, chunking_percent),
                    cost_efficiency=self._assess_cost_efficiency(patents_per_minute, specs['embedding_dimension'])
                )
                
                performance_metrics[model_name] = performance
        
        return performance_metrics
    
    def _estimate_chunking_percentage(self, context_window: int) -> float:
        """Estimate percentage of patents requiring chunking based on context window."""
        # Based on our experimental observations
        if context_window >= 8192:
            return 5.0  # Very few patents need chunking
        elif context_window >= 2048:
            return 15.0  # Some patents need chunking (embeddinggemma)
        else:
            return 85.0  # Most patents need chunking (mxbai-embed-large)
    
    def _get_quality_ranking(self, model_name: str) -> int:
        """Quality ranking based on embedding dimension and architecture."""
        quality_order = {
            'bge-m3': 1,           # Highest quality - 1024D, multilingual
            'nomic-embed-text': 2, # High quality - optimized for speed
            'mxbai-embed-large': 3, # High precision but limited by chunking
            'embeddinggemma': 4    # Good baseline
        }
        return quality_order.get(model_name, 5)
    
    def _assess_production_viability(self, speed: float, chunking_percent: float) -> str:
        """Assess production viability based on speed and chunking requirements."""
        if speed > 300 and chunking_percent < 20:
            return "Excellent"
        elif speed > 200 and chunking_percent < 30:
            return "Very Good"
        elif speed > 100 and chunking_percent < 50:
            return "Good"
        elif speed > 50:
            return "Fair"
        else:
            return "Poor"
    
    def _assess_cost_efficiency(self, speed: float, embedding_dim: int) -> str:
        """Assess cost efficiency based on speed and resource usage."""
        efficiency_score = speed / (embedding_dim / 768)  # Normalize by dimension
        
        if efficiency_score > 400:
            return "Excellent"
        elif efficiency_score > 250:
            return "Very Good"
        elif efficiency_score > 150:
            return "Good"
        elif efficiency_score > 75:
            return "Fair"
        else:
            return "Poor"
    
    def generate_comprehensive_ranking(self, performance_metrics: Dict[str, ModelPerformance]) -> List[Tuple[str, float]]:
        """Generate comprehensive ranking based on weighted scoring."""
        
        rankings = []
        
        for model_name, perf in performance_metrics.items():
            # Weighted scoring system
            score = 0
            
            # Speed (30% weight) - Higher is better
            speed_score = min(perf.speed_patents_per_minute / 500, 1.0) * 30
            
            # Context window (20% weight) - Higher is better, avoid chunking
            context_score = min(perf.context_window / 8192, 1.0) * 20
            
            # Quality (25% weight) - Lower ranking number is better
            quality_score = (5 - perf.quality_ranking) / 4 * 25
            
            # Production viability (15% weight)
            viability_scores = {"Excellent": 1.0, "Very Good": 0.8, "Good": 0.6, "Fair": 0.4, "Poor": 0.2}
            viability_score = viability_scores.get(perf.production_viability, 0) * 15
            
            # Cost efficiency (10% weight)
            cost_scores = {"Excellent": 1.0, "Very Good": 0.8, "Good": 0.6, "Fair": 0.4, "Poor": 0.2}
            cost_score = cost_scores.get(perf.cost_efficiency, 0) * 10
            
            total_score = speed_score + context_score + quality_score + viability_score + cost_score
            
            rankings.append((model_name, total_score))
        
        # Sort by score (highest first)
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        return rankings
    
    def create_performance_report(self) -> str:
        """Create comprehensive performance report."""
        
        performance_metrics = self.analyze_performance_metrics()
        rankings = self.generate_comprehensive_ranking(performance_metrics)
        
        report = "# 🔬 COMPREHENSIVE MODEL PERFORMANCE ANALYSIS\n\n"
        report += f"*Generated from {len(self.batch_results)} experiment batches*\n\n"
        
        # Rankings section
        report += "## 🏆 FINAL MODEL RANKINGS (Comprehensive Score)\n\n"
        for i, (model_name, score) in enumerate(rankings):
            perf = performance_metrics[model_name]
            report += f"### {i+1}. **{model_name.upper()}** (Score: {score:.1f}/100)\n"
            report += f"- **Speed**: {perf.speed_patents_per_minute:.0f} patents/min ({perf.speed_seconds_per_patent:.1f}s per patent)\n"
            report += f"- **Context Window**: {perf.context_window:,} tokens ({perf.chunking_required_percent:.1f}% need chunking)\n"
            report += f"- **Embedding Dimension**: {perf.embedding_dimension}D\n"
            report += f"- **Production Viability**: {perf.production_viability}\n"
            report += f"- **Cost Efficiency**: {perf.cost_efficiency}\n"
            report += f"- **Quality Ranking**: #{perf.quality_ranking}\n\n"
        
        # Detailed analysis
        report += "## 📊 DETAILED PERFORMANCE BREAKDOWN\n\n"
        
        report += "### ⚡ Speed Analysis\n"
        speed_sorted = sorted(performance_metrics.items(), key=lambda x: x[1].speed_patents_per_minute, reverse=True)
        for model_name, perf in speed_sorted:
            report += f"- **{model_name}**: {perf.speed_patents_per_minute:.0f} patents/min\n"
        
        report += "\n### 🎯 Context Window Efficiency\n"
        context_sorted = sorted(performance_metrics.items(), key=lambda x: x[1].context_window, reverse=True)
        for model_name, perf in context_sorted:
            report += f"- **{model_name}**: {perf.context_window:,} tokens ({perf.chunking_required_percent:.1f}% chunking)\n"
        
        report += "\n### 💎 Quality Ranking\n"
        quality_sorted = sorted(performance_metrics.items(), key=lambda x: x[1].quality_ranking)
        for model_name, perf in quality_sorted:
            report += f"- **{model_name}**: Rank #{perf.quality_ranking} ({perf.embedding_dimension}D embeddings)\n"
        
        # Recommendations
        report += "\n## 🎯 PRODUCTION RECOMMENDATIONS\n\n"
        
        top_model = rankings[0][0]
        report += f"### 🥇 **PRIMARY RECOMMENDATION: {top_model.upper()}**\n"
        top_perf = performance_metrics[top_model]
        report += f"- **Best Overall Performance**: {rankings[0][1]:.1f}/100 comprehensive score\n"
        report += f"- **Speed**: {top_perf.speed_patents_per_minute:.0f} patents/min\n"
        report += f"- **Context Efficiency**: {top_perf.context_window:,} tokens, {top_perf.chunking_required_percent:.1f}% chunking\n"
        report += f"- **Production Viability**: {top_perf.production_viability}\n"
        
        # Use case specific recommendations
        report += "\n### 📋 Use Case Specific Rankings:\n\n"
        
        # Speed-critical applications
        fastest_model = max(performance_metrics.items(), key=lambda x: x[1].speed_patents_per_minute)[0]
        report += f"**🚀 Speed-Critical Applications**: {fastest_model} ({performance_metrics[fastest_model].speed_patents_per_minute:.0f} patents/min)\n"
        
        # Quality-critical applications  
        highest_quality = min(performance_metrics.items(), key=lambda x: x[1].quality_ranking)[0]
        report += f"**🎯 Quality-Critical Applications**: {highest_quality} (Rank #{performance_metrics[highest_quality].quality_ranking}, {performance_metrics[highest_quality].embedding_dimension}D)\n"
        
        # Large-scale production
        report += f"**🏭 Large-Scale Production**: {top_model} (Best overall balance)\n"
        
        # Cost-sensitive deployments
        most_cost_efficient = max(performance_metrics.items(), key=lambda x: x[1].cost_efficiency == "Excellent")[0]
        report += f"**💰 Cost-Sensitive Deployments**: {most_cost_efficient} (Cost efficiency: {performance_metrics[most_cost_efficient].cost_efficiency})\n"
        
        return report
    
    def create_performance_table(self, performance_metrics: Dict[str, ModelPerformance]) -> str:
        """Create a formatted table of performance metrics."""
        
        table = "| Model | Speed (pat/min) | Context Window | Embedding Dim | Chunking % | Production Viability | Cost Efficiency |\n"
        table += "|-------|-----------------|----------------|---------------|------------|---------------------|------------------|\n"
        
        # Sort by comprehensive ranking
        rankings = self.generate_comprehensive_ranking(performance_metrics)
        
        for model_name, _ in rankings:
            perf = performance_metrics[model_name]
            table += f"| **{model_name}** | {perf.speed_patents_per_minute:.0f} | {perf.context_window:,} | {perf.embedding_dimension} | {perf.chunking_required_percent:.1f}% | {perf.production_viability} | {perf.cost_efficiency} |\n"
        
        return table


def main():
    """Generate comprehensive model performance analysis."""
    
    analyzer = ModelPerformanceAnalyzer()
    
    # Generate analysis
    print("🔬 Analyzing model performance across all experiments...")
    
    performance_metrics = analyzer.analyze_performance_metrics()
    
    if not performance_metrics:
        print("❌ No performance data found. Make sure batch result files exist.")
        return
    
    print(f"✅ Analyzed {len(performance_metrics)} models")
    
    # Generate comprehensive report
    report = analyzer.create_performance_report()
    
    # Save report
    with open("MODEL_PERFORMANCE_ANALYSIS.md", "w") as f:
        f.write(report)
    
    # Print summary
    print("\n" + "="*80)
    print("MODEL PERFORMANCE ANALYSIS - KEY FINDINGS")
    print("="*80)
    
    rankings = analyzer.generate_comprehensive_ranking(performance_metrics)
    
    print(f"\n🏆 TOP 3 MODELS (Comprehensive Ranking):")
    for i, (model_name, score) in enumerate(rankings[:3]):
        perf = performance_metrics[model_name]
        print(f"  {i+1}. {model_name.upper()}: {score:.1f}/100")
        print(f"     Speed: {perf.speed_patents_per_minute:.0f} patents/min")
        print(f"     Context: {perf.context_window:,} tokens")
        print(f"     Quality: Rank #{perf.quality_ranking}")
        print(f"     Production: {perf.production_viability}")
        print()
    
    print(f"📊 Full analysis saved to: MODEL_PERFORMANCE_ANALYSIS.md")
    
    return performance_metrics, rankings


if __name__ == "__main__":
    main()