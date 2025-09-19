#!/bin/bash
echo "Launching Apple Embedding Atlas..."
echo "Dataset: openai_all_three_atlas.parquet (3,431 patents with all 3 models)"
echo ""
echo "Atlas will open in your default browser at http://localhost:8000"
echo "Use Ctrl+C to stop the server when done"
echo ""

cd data_v2/atlas_data || exit 1

embedding-atlas \
    --data "openai_all_three_atlas.parquet" \
    --text "text" \
    --x "umap_x" \
    --y "umap_y" \
    --color "classification" \
    --host "localhost" \
    --port 8000
