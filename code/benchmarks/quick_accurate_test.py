#!/usr/bin/env python3
"""Quick but accurate token test with warm-up for top models."""

import json
import time
import urllib.request
import statistics

def make_request(url, data=None, timeout=120):
    if data:
        data = json.dumps(data).encode('utf-8')
    req = urllib.request.Request(url, data=data)
    req.add_header('Content-Type', 'application/json')
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode('utf-8'))

def warm_up_and_test(model_name):
    """Warm up model and test token rate."""
    print(f"\nTesting {model_name}:")

    # Warm-up call
    print("  Warming up...")
    try:
        start = time.time()
        make_request("http://localhost:11434/api/generate", {
            "model": model_name,
            "prompt": "Hello",
            "stream": False,
            "options": {"max_tokens": 5}
        }, timeout=180)
        warm_time = time.time() - start
        print(f"  Warm-up: {warm_time:.2f}s")
        time.sleep(2)  # Let model settle
    except Exception as e:
        print(f"  Warm-up failed: {e}")
        return None

    # Actual test calls
    prompt = "Explain the key technical innovations in this patent abstract: A novel battery management system for electric vehicles that uses machine learning algorithms to optimize charging patterns and extend battery life through predictive maintenance and thermal regulation."

    times = []
    token_rates = []

    for i in range(3):
        try:
            start = time.time()
            response = make_request("http://localhost:11434/api/generate", {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"max_tokens": 100, "temperature": 0.7}
            }, timeout=120)
            end = time.time()

            response_time = end - start
            generated_text = response.get('response', '')
            tokens = len(generated_text.split())
            token_rate = tokens / response_time if response_time > 0 else 0

            times.append(response_time)
            token_rates.append(token_rate)

            print(f"  Test {i+1}: {response_time:.2f}s, {tokens} tokens, {token_rate:.2f} tok/s")

        except Exception as e:
            print(f"  Test {i+1} failed: {e}")

    if token_rates:
        avg_rate = statistics.mean(token_rates)
        avg_time = statistics.mean(times)
        return {
            'model': model_name,
            'warm_up_time': warm_time,
            'avg_token_rate': avg_rate,
            'avg_response_time': avg_time,
            'successful_tests': len(token_rates)
        }
    return None

def test_embedding(model_name):
    """Test embedding model."""
    print(f"\nTesting embedding {model_name}:")

    try:
        # Warm-up
        start = time.time()
        make_request("http://localhost:11434/api/embeddings", {
            "model": model_name, "prompt": "test"
        }, timeout=60)
        warm_time = time.time() - start
        print(f"  Warm-up: {warm_time:.2f}s")

        # Test calls
        times = []
        text = "A pharmaceutical composition for treating inflammatory diseases using targeted drug delivery."

        for i in range(3):
            start = time.time()
            response = make_request("http://localhost:11434/api/embeddings", {
                "model": model_name, "prompt": text
            }, timeout=60)
            end = time.time()

            response_time = end - start
            times.append(response_time)
            print(f"  Test {i+1}: {response_time:.2f}s")

        if times:
            avg_time = statistics.mean(times)
            embedding = response.get('embedding', [])
            dimensions = len(embedding)
            return {
                'model': model_name,
                'warm_up_time': warm_time,
                'avg_response_time': avg_time,
                'embeddings_per_second': 1/avg_time,
                'dimensions': dimensions
            }
    except Exception as e:
        print(f"  Failed: {e}")
    return None

def main():
    print("QUICK ACCURATE TOKEN RATE TEST")
    print("="*40)

    # Test top performing models
    llm_models = [
        "llama3.2:latest", "gpt-oss:latest", "llama3.1:8b",
        "smollm2:latest", "gemma3:latest"
    ]

    embedding_models = [
        "mxbai-embed-large:latest", "bge-m3:latest",
        "embeddinggemma:latest", "nomic-embed-text:latest"
    ]

    results = {}

    # Test LLMs
    print("\nLLM MODELS:")
    for model in llm_models:
        result = warm_up_and_test(model)
        if result:
            results[model] = result

    # Test embeddings
    print("\nEMBEDDING MODELS:")
    for model in embedding_models:
        result = test_embedding(model)
        if result:
            results[model] = result

    # Print summary
    print("\n" + "="*50)
    print("SUMMARY RESULTS")
    print("="*50)

    llm_results = {k: v for k, v in results.items() if 'avg_token_rate' in v}
    emb_results = {k: v for k, v in results.items() if 'embeddings_per_second' in v}

    if llm_results:
        print("\nLLM Token Rates (after warm-up):")
        sorted_llm = sorted(llm_results.items(), key=lambda x: x[1]['avg_token_rate'], reverse=True)
        for i, (model, data) in enumerate(sorted_llm, 1):
            print(f"  {i}. {model}: {data['avg_token_rate']:.2f} tokens/sec")

    if emb_results:
        print("\nEmbedding Rates (after warm-up):")
        sorted_emb = sorted(emb_results.items(), key=lambda x: x[1]['embeddings_per_second'], reverse=True)
        for i, (model, data) in enumerate(sorted_emb, 1):
            print(f"  {i}. {model}: {data['embeddings_per_second']:.2f} emb/sec ({data['dimensions']} dims)")

    # Save results
    import os
    os.makedirs("/home/c/projects/patent_research/ollama_benchmark_results", exist_ok=True)
    with open("/home/c/projects/patent_research/ollama_benchmark_results/quick_accurate_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    return results

if __name__ == "__main__":
    main()