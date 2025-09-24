#!/usr/bin/env python3
"""Filter OpenAI batch requests to exclude patents exceeding token limits."""

import json

def filter_batch_file(input_file, output_file, max_chars=30000):
    """Filter batch file to exclude long texts that would exceed token limits.

    Using conservative estimate: 30,000 chars max (well under 8,191 token limit)
    This provides a safety margin for tokenization variations.
    """
    print(f"Filtering {input_file} -> {output_file}")
    print(f"Maximum character limit: {max_chars:,} chars")
    print(f"Estimated token limit: ~{max_chars/4:.0f} tokens (with safety margin)")
    print()

    excluded_patents = []
    included_count = 0
    total_count = 0

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            total_count += 1
            data = json.loads(line.strip())

            text_length = len(data['body']['input'])
            patent_id = data['custom_id'].replace('embedding-', '')

            if text_length <= max_chars:
                outfile.write(json.dumps(data) + '\n')
                included_count += 1
            else:
                excluded_patents.append({
                    'patent_id': patent_id,
                    'char_count': text_length,
                    'estimated_tokens': text_length / 4
                })

            if total_count % 1000 == 0:
                print(f"Processed {total_count} requests...")

    print(f"\n=== FILTERING COMPLETE ===")
    print(f"Total requests processed: {total_count}")
    print(f"Requests included: {included_count}")
    print(f"Requests excluded: {len(excluded_patents)}")
    print(f"Inclusion rate: {included_count/total_count*100:.2f}%")

    if excluded_patents:
        print(f"\n=== EXCLUDED PATENTS ===")
        for patent in excluded_patents:
            print(f"  {patent['patent_id']}: {patent['char_count']:,} chars (~{patent['estimated_tokens']:.0f} tokens)")

    # Save exclusion report
    report_file = output_file.replace('.jsonl', '_exclusions.json')
    with open(report_file, 'w') as f:
        json.dump({
            'max_chars': max_chars,
            'total_requests': total_count,
            'included': included_count,
            'excluded': len(excluded_patents),
            'excluded_patents': excluded_patents
        }, f, indent=2)

    print(f"\nExclusion report saved to: {report_file}")

    return included_count, excluded_patents

if __name__ == "__main__":
    included, excluded = filter_batch_file(
        "openai_batch_requests.jsonl",
        "openai_batch_requests_filtered.jsonl",
        max_chars=30000  # Conservative limit with safety margin
    )

    print(f"\n✅ Ready to submit filtered batch with {included} requests")