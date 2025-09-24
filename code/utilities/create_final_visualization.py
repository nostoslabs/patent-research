#!/usr/bin/env python3
"""Create final visualization of accurate token rates."""

def create_performance_chart():
    """Create comprehensive performance visualization."""

    # Data from accurate testing
    llm_data = {
        'smollm2:latest': {'tokens_per_sec': 34.20, 'warm_up': 2.77},
        'llama3.2:latest': {'tokens_per_sec': 24.18, 'warm_up': 3.72},
        'gemma3:latest': {'tokens_per_sec': 12.25, 'warm_up': 8.28},
        'llama3.1:8b': {'tokens_per_sec': 5.74, 'warm_up': 9.66}
    }

    embedding_data = {
        'nomic-embed-text:latest': {'emb_per_sec': 46.33, 'warm_up': 6.70, 'dims': 768},
        'mxbai-embed-large:latest': {'emb_per_sec': 29.82, 'warm_up': 7.10, 'dims': 1024},
        'embeddinggemma:latest': {'emb_per_sec': 8.00, 'warm_up': 13.42, 'dims': 768},
        'bge-m3:latest': {'emb_per_sec': 6.63, 'warm_up': 2.32, 'dims': 1024}
    }

    print("OLLAMA ACCURATE TOKEN RATE ANALYSIS")
    print("=" * 50)
    print("✅ Results after proper model warm-up")
    print()

    # LLM Performance Chart
    print("🚀 LLM TOKEN GENERATION RATES")
    print("=" * 35)
    print()

    max_tokens = max(data['tokens_per_sec'] for data in llm_data.values())

    for i, (model, data) in enumerate(sorted(llm_data.items(), key=lambda x: x[1]['tokens_per_sec'], reverse=True), 1):
        bar_length = int((data['tokens_per_sec'] / max_tokens) * 40)
        bar = "█" * bar_length
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f" {i}"

        print(f"{medal} {model:<25}")
        print(f"   {bar:<40} {data['tokens_per_sec']:>6.1f} tok/s")
        print(f"   Warm-up: {data['warm_up']:.1f}s")
        print()

    # Embedding Performance Chart
    print("⚡ EMBEDDING GENERATION RATES")
    print("=" * 35)
    print()

    max_emb = max(data['emb_per_sec'] for data in embedding_data.values())

    for i, (model, data) in enumerate(sorted(embedding_data.items(), key=lambda x: x[1]['emb_per_sec'], reverse=True), 1):
        bar_length = int((data['emb_per_sec'] / max_emb) * 40)
        bar = "█" * bar_length
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f" {i}"

        print(f"{medal} {model:<25}")
        print(f"   {bar:<40} {data['emb_per_sec']:>6.1f} emb/s")
        print(f"   Dimensions: {data['dims']}, Warm-up: {data['warm_up']:.1f}s")
        print()

    # Performance Comparison
    print("📊 PERFORMANCE COMPARISON")
    print("=" * 30)
    print()

    print("Speed Tiers:")
    print("🔥 Ultra-Fast:  30+ tokens/sec  → smollm2")
    print("⚡ Fast:        20+ tokens/sec  → llama3.2")
    print("🟢 Good:        10+ tokens/sec  → gemma3")
    print("🟡 Moderate:    5+ tokens/sec   → llama3.1:8b")
    print()

    print("Warm-up Speed:")
    print("🚀 Instant:     <5 seconds      → smollm2, llama3.2")
    print("⏱️  Quick:       5-10 seconds    → gemma3, llama3.1")
    print("⏳ Moderate:    10+ seconds     → Large models")
    print()

    # Recommendations
    print("🎯 RECOMMENDATIONS")
    print("=" * 20)
    print()
    print("🏆 Best Overall:     smollm2:latest")
    print("   • 34.20 tokens/sec after 2.77s warm-up")
    print("   • Perfect for real-time applications")
    print()
    print("⚖️  Best Balance:     llama3.2:latest")
    print("   • 24.18 tokens/sec after 3.72s warm-up")
    print("   • Excellent speed + quality trade-off")
    print()
    print("📝 Best Quality:     gemma3:latest")
    print("   • 12.25 tokens/sec, detailed responses")
    print("   • Choose when quality > speed")
    print()
    print("🔍 Best Embedding:   nomic-embed-text:latest")
    print("   • 46.33 embeddings/sec, 768 dimensions")
    print("   • Fastest embedding processing")
    print()

    # Technical insights
    print("🔬 TECHNICAL INSIGHTS")
    print("=" * 22)
    print()
    print("• Model warm-up is CRITICAL - 5-10x performance difference")
    print("• Smaller models (1-3B) often outperform larger ones in speed")
    print("• Embedding models are much faster than expected after warm-up")
    print("• gpt-oss models have loading issues requiring extended timeouts")
    print("• Token rates are highly consistent after warm-up period")

def main():
    create_performance_chart()

    print("\n" + "=" * 60)
    print("📁 COMPLETE BENCHMARK RESULTS AVAILABLE:")
    print("   • comprehensive_token_rate_report.md")
    print("   • quick_accurate_results.json")
    print("   • All benchmark scripts for reproduction")
    print("=" * 60)

if __name__ == "__main__":
    main()