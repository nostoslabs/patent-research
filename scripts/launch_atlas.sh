#!/bin/bash
# Launch Apple Embedding Atlas for Patent Research
# Updated for consolidated data format

echo "Launching Apple Embedding Atlas..."
echo "Dataset: openai_all_three_atlas.parquet"
echo "Data directory: data_v2/atlas_data"
echo ""
echo "Available datasets:"
echo "  - openai_all_three_atlas.parquet (OpenAI embeddings, patents with all 3 models)"
echo "  - nomic_sample_atlas.parquet (nomic embeddings, broad sample)"
echo "  - model_comparison_atlas.parquet (All models side by side)"
echo ""
echo "Atlas will open in your default browser"
echo "Use Ctrl+C to stop the server when done"
echo ""

cd "data_v2/atlas_data" || exit 1

# Launch with optimal parameters for patent data
embedding-atlas \
    --data "openai_all_three_atlas.parquet" \
    --text "text" \
    --x "umap_x" \
    --y "umap_y" \
    --color "classification" \
    --host "localhost" \
    --port 8000

echo ""
echo "Atlas server stopped"