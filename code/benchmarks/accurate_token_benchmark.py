#!/usr/bin/env python3
"""
Accurate Token Rate Benchmark for Ollama Models
Includes proper model warm-up and detailed token rate analysis.
"""

import json
import time
import urllib.request
import urllib.parse
import statistics
import random
import os
from datetime import datetime

def make_request(url, data=None, timeout=120):
    """Make HTTP request with proper error handling."""
    if data:
        data = json.dumps(data).encode('utf-8')
    req = urllib.request.Request(url, data=data)
    req.add_header('Content-Type', 'application/json')

    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode('utf-8'))

def warm_up_model(model_name, base_url="http://localhost:11434"):
    """Warm up a model by making a simple request to load it into memory."""
    print(f"Warming up model: {model_name}")

    try:
        start_time = time.time()
        response = make_request(
            f"{base_url}/api/generate",
            {
                "model": model_name,
                "prompt": "Hello",
                "stream": False,
                "options": {"max_tokens": 5}
            },
            timeout=300  # 5 minutes for large models
        )
        end_time = time.time()

        warm_up_time = end_time - start_time
        print(f"  Warm-up completed in {warm_up_time:.2f}s")

        # Wait a moment to ensure model is settled
        time.sleep(2)

        return warm_up_time, True
    except Exception as e:
        print(f"  Warm-up failed: {e}")
        return 0, False

def count_tokens_accurately(text):
    """More accurate token counting approximation."""
    # Simple approximation: tokens ≈ words + punctuation/4
    words = len(text.split())
    punctuation = sum(1 for c in text if not c.isalnum() and not c.isspace())
    return words + punctuation // 4

def benchmark_llm_tokens(model_name, patent_data, num_tests=5, base_url="http://localhost:11434"):
    """Benchmark LLM token generation rate after warm-up."""
    print(f"\nBenchmarking LLM token rate: {model_name}")

    # Warm up the model first
    warm_up_time, warm_up_success = warm_up_model(model_name, base_url)
    if not warm_up_success:
        return None

    # Test prompts of varying lengths
    test_prompts = [
        "Summarize this patent abstract in 2-3 sentences: ",
        "What are the key technical innovations described in this patent: ",
        "Explain the commercial potential and technical advantages of this invention: ",
        "Provide a detailed analysis of the technical approach used in this patent: ",
    ]

    results = {
        'model': model_name,
        'warm_up_time': warm_up_time,
        'tests': [],
        'response_times': [],
        'prompt_tokens': [],
        'generated_tokens': [],
        'total_tokens': [],
        'tokens_per_second': [],
        'success_count': 0
    }

    for i in range(num_tests):
        try:
            # Select random prompt and patent
            prompt = random.choice(test_prompts)
            abstract = random.choice(patent_data)
            full_prompt = f"{prompt}{abstract}"

            prompt_tokens = count_tokens_accurately(full_prompt)

            print(f"  Test {i+1}/{num_tests}: Prompt tokens: {prompt_tokens}")

            start_time = time.time()
            response = make_request(
                f"{base_url}/api/generate",
                {
                    "model": model_name,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "max_tokens": 200  # Longer responses for better token rate measurement
                    }
                },
                timeout=180
            )
            end_time = time.time()

            response_time = end_time - start_time
            generated_text = response.get('response', '')
            generated_tokens = count_tokens_accurately(generated_text)
            total_tokens = prompt_tokens + generated_tokens
            tokens_per_second = generated_tokens / response_time if response_time > 0 else 0

            test_result = {
                'test_number': i + 1,
                'response_time': response_time,
                'prompt_tokens': prompt_tokens,
                'generated_tokens': generated_tokens,
                'total_tokens': total_tokens,
                'tokens_per_second': tokens_per_second,
                'response_preview': generated_text[:100] + "..." if len(generated_text) > 100 else generated_text
            }

            results['tests'].append(test_result)
            results['response_times'].append(response_time)
            results['prompt_tokens'].append(prompt_tokens)
            results['generated_tokens'].append(generated_tokens)
            results['total_tokens'].append(total_tokens)
            results['tokens_per_second'].append(tokens_per_second)
            results['success_count'] += 1

            print(f"    Response time: {response_time:.2f}s, Generated: {generated_tokens} tokens, Rate: {tokens_per_second:.2f} tok/s")

        except Exception as e:
            print(f"    Test {i+1} failed: {e}")

    # Calculate statistics
    if results['tokens_per_second']:
        results['avg_tokens_per_second'] = statistics.mean(results['tokens_per_second'])
        results['median_tokens_per_second'] = statistics.median(results['tokens_per_second'])
        results['max_tokens_per_second'] = max(results['tokens_per_second'])
        results['min_tokens_per_second'] = min(results['tokens_per_second'])
        results['avg_response_time'] = statistics.mean(results['response_times'])
        results['avg_generated_tokens'] = statistics.mean(results['generated_tokens'])
        results['success_rate'] = results['success_count'] / num_tests
    else:
        results['avg_tokens_per_second'] = 0
        results['median_tokens_per_second'] = 0
        results['max_tokens_per_second'] = 0
        results['min_tokens_per_second'] = 0
        results['avg_response_time'] = 0
        results['avg_generated_tokens'] = 0
        results['success_rate'] = 0

    return results

def benchmark_embedding_rate(model_name, patent_data, num_tests=8, base_url="http://localhost:11434"):
    """Benchmark embedding generation rate after warm-up."""
    print(f"\nBenchmarking embedding rate: {model_name}")

    # Warm up with a simple embedding
    print(f"Warming up embedding model: {model_name}")
    try:
        start_time = time.time()
        make_request(
            f"{base_url}/api/embeddings",
            {"model": model_name, "prompt": "test"},
            timeout=60
        )
        end_time = time.time()
        warm_up_time = end_time - start_time
        print(f"  Warm-up completed in {warm_up_time:.2f}s")
        time.sleep(1)
    except Exception as e:
        print(f"  Warm-up failed: {e}")
        return None

    results = {
        'model': model_name,
        'warm_up_time': warm_up_time,
        'tests': [],
        'response_times': [],
        'embeddings_per_second': [],
        'success_count': 0,
        'embedding_dimension': None
    }

    for i in range(num_tests):
        try:
            text = random.choice(patent_data)

            start_time = time.time()
            response = make_request(
                f"{base_url}/api/embeddings",
                {"model": model_name, "prompt": text},
                timeout=60
            )
            end_time = time.time()

            response_time = end_time - start_time
            embeddings_per_second = 1 / response_time if response_time > 0 else 0

            # Get embedding dimension
            if results['embedding_dimension'] is None:
                embedding = response.get('embedding', [])
                results['embedding_dimension'] = len(embedding)

            test_result = {
                'test_number': i + 1,
                'response_time': response_time,
                'embeddings_per_second': embeddings_per_second,
                'text_length': len(text)
            }

            results['tests'].append(test_result)
            results['response_times'].append(response_time)
            results['embeddings_per_second'].append(embeddings_per_second)
            results['success_count'] += 1

            print(f"  Test {i+1}/{num_tests}: {response_time:.2f}s, {embeddings_per_second:.2f} emb/s")

        except Exception as e:
            print(f"  Test {i+1} failed: {e}")

    # Calculate statistics
    if results['embeddings_per_second']:
        results['avg_embeddings_per_second'] = statistics.mean(results['embeddings_per_second'])
        results['median_embeddings_per_second'] = statistics.median(results['embeddings_per_second'])
        results['avg_response_time'] = statistics.mean(results['response_times'])
        results['success_rate'] = results['success_count'] / num_tests
    else:
        results['avg_embeddings_per_second'] = 0
        results['median_embeddings_per_second'] = 0
        results['avg_response_time'] = 0
        results['success_rate'] = 0

    return results

def load_patent_data(file_path, sample_size=30):
    """Load patent abstracts for testing."""
    try:
        with open(file_path, 'r') as f:
            data = [json.loads(line) for line in f]

        abstracts = [item.get('abstract', '')[:400] for item in data if item.get('abstract') and len(item.get('abstract', '')) > 50]
        patent_data = random.sample(abstracts, min(sample_size, len(abstracts)))
        print(f"Loaded {len(patent_data)} patent abstracts for testing")
        return patent_data
    except Exception as e:
        print(f"Error loading patent data: {e}")
        return [
            "A novel method for improving battery efficiency in electric vehicles through advanced thermal management systems and intelligent power distribution algorithms.",
            "A pharmaceutical composition comprising targeted drug delivery mechanisms for treating inflammatory diseases with reduced side effects.",
            "Machine learning algorithm for predicting equipment failure in industrial systems using sensor data and pattern recognition techniques.",
            "An improved solar panel design with enhanced light absorption capabilities and increased energy conversion efficiency through nanostructured surfaces.",
            "A surgical instrument with haptic feedback capabilities for minimally invasive procedures, providing enhanced precision and control."
        ] * 6

def main():
    """Main benchmarking function."""
    print("ACCURATE OLLAMA TOKEN RATE BENCHMARK")
    print("="*50)
    print("This benchmark includes proper model warm-up for accurate token rate measurement.")
    print()

    # Load patent data
    patent_data = load_patent_data("/home/c/projects/patent_research/data/raw/patent_abstracts.jsonl")

    # Define working models (excluding timeout models)
    llm_models = [
        "llama3.2:latest", "gpt-oss:latest", "llama3.1:8b", "smollm2:latest",
        "gemma3:latest", "llava:latest", "gpt-oss:20b", "qwen3:latest",
        "deepseek-r1:latest", "deepseek-r1:8b", "deepseek-r1:32b"
    ]

    embedding_models = [
        "mxbai-embed-large:latest", "bge-m3:latest",
        "embeddinggemma:latest", "nomic-embed-text:latest"
    ]

    all_results = {}

    # Benchmark LLM models
    print("\n" + "="*60)
    print("BENCHMARKING LLM MODELS")
    print("="*60)

    for model in llm_models:
        try:
            result = benchmark_llm_tokens(model, patent_data, num_tests=4)
            if result:
                all_results[model] = result
                print(f"✓ {model}: {result['avg_tokens_per_second']:.2f} tokens/sec (avg)")
        except Exception as e:
            print(f"✗ {model}: Failed - {e}")

    # Benchmark embedding models
    print("\n" + "="*60)
    print("BENCHMARKING EMBEDDING MODELS")
    print("="*60)

    for model in embedding_models:
        try:
            result = benchmark_embedding_rate(model, patent_data, num_tests=6)
            if result:
                all_results[model] = result
                result['type'] = 'embedding'
                print(f"✓ {model}: {result['avg_embeddings_per_second']:.2f} embeddings/sec (avg)")
        except Exception as e:
            print(f"✗ {model}: Failed - {e}")

    # Generate comprehensive report
    output_dir = "/home/c/projects/patent_research/ollama_benchmark_results"
    os.makedirs(output_dir, exist_ok=True)

    # Save detailed results
    results_file = os.path.join(output_dir, 'detailed_token_benchmark.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Detailed results saved to: {results_file}")

    return all_results

if __name__ == "__main__":
    results = main()