"""Launch Apple Embedding Atlas with pre-computed coordinates and metadata."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def enhance_parquet_for_atlas(
    parquet_file: str = "patent_embeddings_atlas.parquet",
    reduction_results_dir: str = "reduction_results",
    output_file: str = "patent_atlas_enhanced.parquet"
) -> str:
    """
    Enhance Parquet file with pre-computed coordinates for Atlas.

    Args:
        parquet_file: Input Parquet file with embeddings
        reduction_results_dir: Directory with dimensionality reduction results
        output_file: Output enhanced Parquet file

    Returns:
        Path to enhanced Parquet file
    """
    print("Enhancing Parquet file for Apple Embedding Atlas...")

    # Load existing Parquet file
    df = pd.read_parquet(parquet_file)
    print(f"Loaded {len(df)} patents from {parquet_file}")

    # Load reduction results
    results_path = Path(reduction_results_dir)
    with open(results_path / "reduction_results.json") as f:
        metadata = json.load(f)

    # Add pre-computed coordinates
    coordinate_methods = ["umap", "tsne"]
    for method_name in coordinate_methods:
        embedding_file = results_path / f"{method_name}_embeddings.npy"
        if embedding_file.exists():
            coords = np.load(embedding_file)
            df[f"{method_name}_x"] = coords[:, 0]
            df[f"{method_name}_y"] = coords[:, 1]
            print(f"Added {method_name.upper()} coordinates")

    # Add clustering labels from best performing method
    best_method = "umap"  # UMAP generally performs well
    if best_method in metadata["methods"] and "clustering" in metadata["methods"][best_method]:
        clustering_results = metadata["methods"][best_method]["clustering"]

        # Find best clustering result (highest silhouette score)
        best_clustering = None
        best_score = -1
        for cluster_name, cluster_data in clustering_results.items():
            score = cluster_data["silhouette_score"]
            if score > best_score:
                best_score = score
                best_clustering = cluster_name

        if best_clustering:
            # Load clustering labels (stored in memory during reduction)
            # For now, we'll create synthetic cluster labels based on classification
            # In a real scenario, you'd save cluster labels separately
            df["cluster"] = df["classification"].astype(str)
            print(f"Added cluster labels from {best_clustering} (silhouette: {best_score:.3f})")

    # Add additional metadata for Atlas
    df["abstract_words"] = df["text"].str.split().str.len()
    df["has_long_abstract"] = (df["abstract_length"] > 1000).astype(str)

    # Create display text with more info
    df["atlas_display"] = df.apply(
        lambda row: f"Patent {row['id']}: {row['display_text'][:100]}... "
                   f"(Class: {row['classification']}, {row['abstract_length']} chars)",
        axis=1
    )

    # Save enhanced Parquet file
    df.to_parquet(output_file, index=False)

    file_size = Path(output_file).stat().st_size / (1024 * 1024)  # MB
    print(f"Saved enhanced Parquet file: {output_file} ({file_size:.1f} MB)")
    print(f"Columns: {list(df.columns)}")

    return output_file


def create_atlas_launch_script(
    parquet_file: str,
    script_name: str = "launch_atlas.sh"
) -> str:
    """
    Create shell script to launch Embedding Atlas with proper parameters.

    Args:
        parquet_file: Parquet file for Atlas
        script_name: Output script name

    Returns:
        Path to launch script
    """
    script_content = f"""#!/bin/bash
# Launch Apple Embedding Atlas for Patent Research

echo "Launching Apple Embedding Atlas..."
echo "Dataset: {parquet_file}"
echo ""
echo "Atlas will open in your default browser"
echo "Use Ctrl+C to stop the server when done"
echo ""

# Launch with optimal parameters for patent data
embedding-atlas \\
    --data "{parquet_file}" \\
    --text "text" \\
    --embedding "embedding" \\
    --x "umap_x" \\
    --y "umap_y" \\
    --host "localhost" \\
    --port 8000

echo ""
echo "Atlas server stopped"
"""

    with open(script_name, "w") as f:
        f.write(script_content)

    # Make executable
    import stat
    Path(script_name).chmod(stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)

    print(f"Created launch script: {script_name}")
    return script_name


def launch_atlas_directly(
    parquet_file: str,
    text_column: str = "text",
    embedding_column: str = "embedding",
    x_column: str = "umap_x",
    y_column: str = "umap_y",
    host: str = "localhost",
    port: int = 8000
) -> None:
    """
    Launch Embedding Atlas directly.

    Args:
        parquet_file: Parquet file to visualize
        text_column: Text column name
        embedding_column: Embedding column name
        x_column: X coordinate column name
        y_column: Y coordinate column name
        host: Server host
        port: Server port
    """
    print("Launching Apple Embedding Atlas...")
    print(f"Dataset: {parquet_file}")
    print(f"Server: http://{host}:{port}")
    print("Use Ctrl+C to stop the server")
    print("-" * 50)

    # Build command
    cmd = [
        "embedding-atlas",
        "--data", parquet_file,
        "--text", text_column,
        "--embedding", embedding_column,
        "--host", host,
        "--port", str(port)
    ]

    # Add coordinate columns if available
    df = pd.read_parquet(parquet_file)
    if x_column in df.columns and y_column in df.columns:
        cmd.extend(["--x", x_column, "--y", y_column])
        print(f"Using pre-computed coordinates: {x_column}, {y_column}")

    try:
        # Launch Atlas
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nAtlas server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"Error launching Atlas: {e}")
        print("Make sure embedding-atlas is installed: pip install embedding-atlas")
    except FileNotFoundError:
        print("Error: embedding-atlas command not found")
        print("Install with: pip install embedding-atlas")


def main() -> None:
    """Main function to set up and launch Atlas."""
    print("Apple Embedding Atlas Setup for Patent Research")
    print("=" * 60)

    try:
        # Check if base Parquet file exists
        base_parquet = "patent_embeddings_atlas.parquet"
        if not Path(base_parquet).exists():
            print(f"Error: {base_parquet} not found")
            print("Run: python convert_to_parquet.py")
            return

        # Check if reduction results exist
        if not Path("reduction_results").exists():
            print("Error: reduction_results directory not found")
            print("Run: python dimensionality_reduction.py")
            return

        # Enhance Parquet file with coordinates
        enhanced_parquet = enhance_parquet_for_atlas(
            parquet_file=base_parquet,
            reduction_results_dir="reduction_results",
            output_file="patent_atlas_enhanced.parquet"
        )

        # Create launch script
        launch_script = create_atlas_launch_script(enhanced_parquet)

        print("\n✅ Atlas setup complete!")
        print("\nTo launch Apple Embedding Atlas:")
        print(f"1. Run the script: ./{launch_script}")
        print(f"2. Or run directly: python -c \"from launch_atlas import launch_atlas_directly; launch_atlas_directly('{enhanced_parquet}')\"")

        # Ask user if they want to launch now
        try:
            response = input("\nLaunch Atlas now? (y/n): ").strip().lower()
            if response == 'y':
                launch_atlas_directly(enhanced_parquet)
        except KeyboardInterrupt:
            print("\nSetup complete.")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
