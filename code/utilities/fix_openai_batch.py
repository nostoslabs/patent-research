#!/usr/bin/env python3
"""Fix OpenAI batch requests by truncating long patent texts."""

import json
import tiktoken

def truncate_text_for_embedding(text, max_tokens=8000):
    """Truncate text to fit within token limits for embedding models."""
    # Use cl100k_base encoding (used by text-embedding-3-small)
    encoding = tiktoken.get_encoding("cl100k_base")

    # Encode the text
    tokens = encoding.encode(text)

    # If within limit, return original
    if len(tokens) <= max_tokens:
        return text

    # Truncate to max_tokens and decode back
    truncated_tokens = tokens[:max_tokens]
    truncated_text = encoding.decode(truncated_tokens)

    return truncated_text

def fix_batch_file(input_file, output_file, max_tokens=7500):
    """Fix batch file by truncating long texts."""
    print(f"Processing {input_file} -> {output_file}")
    print(f"Max tokens per text: {max_tokens}")

    truncated_count = 0
    total_count = 0

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            total_count += 1
            data = json.loads(line.strip())

            original_text = data['body']['input']
            truncated_text = truncate_text_for_embedding(original_text, max_tokens)

            if len(truncated_text) < len(original_text):
                truncated_count += 1
                data['body']['input'] = truncated_text

            outfile.write(json.dumps(data) + '\n')

            if total_count % 1000 == 0:
                print(f"Processed {total_count} requests...")

    print(f"Completed! Processed {total_count} requests")
    print(f"Truncated {truncated_count} long texts ({truncated_count/total_count*100:.1f}%)")

if __name__ == "__main__":
    fix_batch_file("openai_batch_requests.jsonl", "openai_batch_requests_fixed.jsonl")