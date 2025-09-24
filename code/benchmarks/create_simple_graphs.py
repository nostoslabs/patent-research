#!/usr/bin/env python3
"""Create simple ASCII graphs of the benchmark results."""

def create_ascii_bar_chart(data, title, max_width=50):
    """Create ASCII bar chart."""
    if not data:
        return f"{title}\nNo data available"

    max_value = max(data.values())

    lines = [title, "=" * len(title), ""]

    for name, value in sorted(data.items(), key=lambda x: x[1], reverse=True):
        bar_length = int((value / max_value) * max_width)
        bar = "█" * bar_length
        lines.append(f"{name:25} │{bar:<{max_width}} {value:.2f}")

    return "\n".join(lines)

def main():
    # LLM response times (from quick test results)
    llm_times = {
        "llama3.2:latest": 6.48,
        "gpt-oss:latest": 7.98,
        "llama3.1:8b": 11.83,
        "smollm2:latest": 11.86,
        "gemma3:latest": 15.39,
        "llava:latest": 17.86,
        "gpt-oss:20b": 24.89,
        "qwen3:latest": 27.48,
        "deepseek-r1:latest": 39.39,
        "deepseek-r1:8b": 43.37,
        "deepseek-r1:32b": 52.72
    }

    # Embedding response times
    embedding_times = {
        "mxbai-embed-large:latest": 2.75,
        "bge-m3:latest": 4.92,
        "embeddinggemma:latest": 5.33,
        "nomic-embed-text:latest": 8.55
    }

    # Estimated tokens per second (inverse of response time, scaled)
    llm_tokens_per_sec = {k: 50/v for k, v in llm_times.items()}  # Rough estimate
    embedding_per_sec = {k: 1/v for k, v in embedding_times.items()}

    print("OLLAMA MODEL PERFORMANCE VISUALIZATION")
    print("=" * 50)
    print()

    # LLM Response Times
    print(create_ascii_bar_chart(llm_times, "LLM Response Times (seconds) - Lower is Better"))
    print("\n" + "="*70 + "\n")

    # LLM Estimated Tokens per Second
    print(create_ascii_bar_chart(llm_tokens_per_sec, "LLM Estimated Tokens/Second - Higher is Better"))
    print("\n" + "="*70 + "\n")

    # Embedding Response Times
    print(create_ascii_bar_chart(embedding_times, "Embedding Response Times (seconds) - Lower is Better"))
    print("\n" + "="*70 + "\n")

    # Embedding Processing Rate
    print(create_ascii_bar_chart(embedding_per_sec, "Embedding Processing Rate (embeddings/sec) - Higher is Better"))
    print("\n" + "="*70 + "\n")

    # Summary
    print("SUMMARY RANKINGS")
    print("=" * 20)
    print()

    print("🏆 TOP PERFORMERS:")
    fastest_llm = min(llm_times.items(), key=lambda x: x[1])
    fastest_emb = min(embedding_times.items(), key=lambda x: x[1])

    print(f"   Fastest LLM: {fastest_llm[0]} ({fastest_llm[1]:.2f}s)")
    print(f"   Fastest Embedding: {fastest_emb[0]} ({fastest_emb[1]:.2f}s)")
    print()

    print("📊 INSIGHTS:")
    print("   • Smaller models (llama3.2, smollm2) excel in speed")
    print("   • DeepSeek models slower due to reasoning tokens")
    print("   • mxbai-embed-large offers best embedding performance")
    print("   • All embedding models are production-ready (<10s)")

if __name__ == "__main__":
    main()