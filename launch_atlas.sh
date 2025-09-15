#!/bin/bash
# Launch Apple Embedding Atlas for Patent Research

echo "Launching Apple Embedding Atlas..."
echo "Dataset: patent_atlas_enhanced.parquet"
echo ""
echo "Atlas will open in your default browser"
echo "Use Ctrl+C to stop the server when done"
echo ""

# Launch with optimal parameters for patent data
embedding-atlas \
    --data "patent_atlas_enhanced.parquet" \
    --text "text" \
    --embedding "embedding" \
    --x "umap_x" \
    --y "umap_y" \
    --host "localhost" \
    --port 8000

echo ""
echo "Atlas server stopped"
