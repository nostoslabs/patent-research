"""Create 2D/3D visualizations of patent embeddings with clustering."""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_reduction_results(results_dir: str = "reduction_results") -> Dict[str, Any]:
    """
    Load dimensionality reduction results from directory.

    Args:
        results_dir: Directory containing reduction results

    Returns:
        Dictionary with loaded results
    """
    results_path = Path(results_dir)
    
    # Load metadata
    with open(results_path / "reduction_results.json") as f:
        metadata = json.load(f)
    
    # Load embedding matrices
    methods = {}
    for method_name in ["pca", "tsne", "umap", "mds"]:
        embedding_file = results_path / f"{method_name}_embeddings.npy"
        if embedding_file.exists():
            methods[method_name] = {
                "embeddings": np.load(embedding_file),
                "info": metadata["methods"][method_name]
            }
    
    return {
        "metadata": metadata,
        "methods": methods,
        "patent_ids": metadata["patent_ids"],
        "classifications": metadata["classifications"]
    }


def create_2d_scatter_plot(
    embeddings: np.ndarray,
    labels: np.ndarray,
    title: str,
    color_column: str = "classification",
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Create interactive 2D scatter plot with Plotly.

    Args:
        embeddings: 2D embeddings (n_samples x 2)
        labels: Labels for coloring points
        title: Plot title
        color_column: Name of color column
        save_path: Optional path to save plot

    Returns:
        Plotly figure
    """
    df = pd.DataFrame({
        "x": embeddings[:, 0],
        "y": embeddings[:, 1],
        color_column: labels
    })
    
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color=color_column,
        title=title,
        width=800,
        height=600,
        hover_data=[color_column]
    )
    
    fig.update_layout(
        showlegend=True,
        xaxis_title=f"{title.split()[0]} Component 1",
        yaxis_title=f"{title.split()[0]} Component 2"
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"Saved plot to {save_path}")
    
    return fig


def create_3d_scatter_plot(
    embeddings: np.ndarray,
    labels: np.ndarray,
    title: str,
    color_column: str = "classification",
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Create interactive 3D scatter plot with Plotly.

    Args:
        embeddings: 3D embeddings (n_samples x 3)
        labels: Labels for coloring points
        title: Plot title
        color_column: Name of color column
        save_path: Optional path to save plot

    Returns:
        Plotly figure
    """
    if embeddings.shape[1] < 3:
        raise ValueError("Need at least 3 dimensions for 3D plot")
    
    df = pd.DataFrame({
        "x": embeddings[:, 0],
        "y": embeddings[:, 1],
        "z": embeddings[:, 2],
        color_column: labels
    })
    
    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        color=color_column,
        title=title,
        width=800,
        height=600,
        hover_data=[color_column]
    )
    
    fig.update_layout(
        scene=dict(
            xaxis_title=f"{title.split()[0]} Component 1",
            yaxis_title=f"{title.split()[0]} Component 2",
            zaxis_title=f"{title.split()[0]} Component 3"
        )
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"Saved plot to {save_path}")
    
    return fig


def create_clustering_comparison_plot(
    embeddings: np.ndarray,
    clustering_results: Dict[str, Any],
    title: str,
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Create subplot comparison of different clustering methods.

    Args:
        embeddings: 2D embeddings
        clustering_results: Dictionary with clustering results
        title: Plot title
        save_path: Optional path to save plot

    Returns:
        Plotly figure with subplots
    """
    n_methods = len(clustering_results)
    cols = min(3, n_methods)
    rows = (n_methods + cols - 1) // cols
    
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=list(clustering_results.keys()),
        specs=[[{"type": "scatter"} for _ in range(cols)] for _ in range(rows)]
    )
    
    for i, (method_name, cluster_data) in enumerate(clustering_results.items()):
        row = i // cols + 1
        col = i % cols + 1
        
        labels = cluster_data["labels"]
        silhouette = cluster_data["silhouette_score"]
        
        # Create scatter trace
        scatter = go.Scatter(
            x=embeddings[:, 0],
            y=embeddings[:, 1],
            mode="markers",
            marker=dict(
                color=labels,
                colorscale="viridis",
                size=4
            ),
            name=f"{method_name} (sil={silhouette:.3f})",
            showlegend=False
        )
        
        fig.add_trace(scatter, row=row, col=col)
    
    fig.update_layout(
        title_text=f"{title} - Clustering Comparison",
        showlegend=False,
        height=400 * rows
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"Saved clustering comparison to {save_path}")
    
    return fig


def create_similarity_heatmap(
    embeddings: np.ndarray,
    patent_ids: List[str],
    max_patents: int = 100,
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Create similarity heatmap for subset of patents.

    Args:
        embeddings: High-dimensional embeddings
        patent_ids: Patent IDs
        max_patents: Maximum number of patents for heatmap
        save_path: Optional path to save plot

    Returns:
        Plotly heatmap figure
    """
    # Use subset for computational efficiency
    n_patents = min(len(embeddings), max_patents)
    indices = np.random.choice(len(embeddings), n_patents, replace=False)
    
    subset_embeddings = embeddings[indices]
    subset_ids = [patent_ids[i] for i in indices]
    
    # Compute cosine similarity matrix
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(subset_embeddings)
    
    fig = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        x=subset_ids,
        y=subset_ids,
        colorscale='RdYlBu_r',
        zmid=0
    ))
    
    fig.update_layout(
        title=f"Patent Similarity Heatmap ({n_patents} patents)",
        width=800,
        height=800,
        xaxis_title="Patents",
        yaxis_title="Patents"
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"Saved similarity heatmap to {save_path}")
    
    return fig


def create_pca_explained_variance_plot(
    pca_info: Dict[str, Any],
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Create PCA explained variance plot.

    Args:
        pca_info: PCA information dictionary
        save_path: Optional path to save plot

    Returns:
        Plotly figure
    """
    # Handle case where arrays might be stored as strings in JSON
    if isinstance(pca_info["explained_variance_ratio"], str):
        explained_var = np.fromstring(
            pca_info["explained_variance_ratio"].strip("[]"), 
            sep=' '
        )
        cumulative_var = np.fromstring(
            pca_info["cumulative_variance"].strip("[]"), 
            sep=' '
        )
    else:
        explained_var = np.array(pca_info["explained_variance_ratio"])
        cumulative_var = np.array(pca_info["cumulative_variance"])
    
    n_components = len(explained_var)
    components = list(range(1, n_components + 1))
    
    fig = go.Figure()
    
    # Individual explained variance
    fig.add_trace(go.Bar(
        x=components,
        y=explained_var.tolist(),
        name="Individual",
        opacity=0.7
    ))
    
    # Cumulative explained variance
    fig.add_trace(go.Scatter(
        x=components,
        y=cumulative_var.tolist(),
        mode="lines+markers",
        name="Cumulative",
        yaxis="y2",
        line=dict(color="red")
    ))
    
    fig.update_layout(
        title="PCA Explained Variance",
        xaxis_title="Principal Component",
        yaxis_title="Explained Variance Ratio",
        yaxis2=dict(
            title="Cumulative Explained Variance",
            overlaying="y",
            side="right"
        ),
        width=800,
        height=500
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"Saved PCA variance plot to {save_path}")
    
    return fig


def create_classification_distribution_plot(
    classifications: List[str],
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Create classification distribution plot.

    Args:
        classifications: List of patent classifications
        save_path: Optional path to save plot

    Returns:
        Plotly figure
    """
    # Count classifications
    from collections import Counter
    class_counts = Counter(classifications)
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(class_counts.keys()),
            y=list(class_counts.values()),
            text=list(class_counts.values()),
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Patent Classification Distribution",
        xaxis_title="Classification",
        yaxis_title="Number of Patents",
        width=800,
        height=500
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"Saved classification distribution to {save_path}")
    
    return fig


def generate_all_visualizations(
    results_dir: str = "reduction_results",
    output_dir: str = "visualizations"
) -> None:
    """
    Generate all visualization plots.

    Args:
        results_dir: Directory with reduction results
        output_dir: Output directory for plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("Loading reduction results...")
    results = load_reduction_results(results_dir)
    
    patent_ids = results["patent_ids"]
    classifications = results["classifications"]
    methods = results["methods"]
    
    print(f"Generating visualizations for {len(patent_ids)} patents...")
    
    # 1. Classification distribution
    print("\n1. Creating classification distribution plot...")
    create_classification_distribution_plot(
        classifications,
        save_path=str(output_path / "classification_distribution.html")
    )
    
    # 2. PCA explained variance
    if "pca" in methods:
        print("2. Creating PCA explained variance plot...")
        create_pca_explained_variance_plot(
            methods["pca"]["info"],
            save_path=str(output_path / "pca_explained_variance.html")
        )
    
    # 3. 2D scatter plots for each method
    print("3. Creating 2D scatter plots...")
    for method_name, method_data in methods.items():
        if method_name == "pca":  # Skip PCA (too many dimensions)
            continue
        
        embeddings = method_data["embeddings"]
        if embeddings.shape[1] >= 2:
            # Handle MDS which might have used a subset
            if method_name == "mds" and "sample_indices" in method_data["info"]:
                # sample_indices might be stored as string in JSON, convert to numpy array
                sample_indices_str = method_data["info"]["sample_indices"]
                if isinstance(sample_indices_str, str):
                    sample_indices = np.fromstring(
                        sample_indices_str.strip("[]"), 
                        sep=' ',
                        dtype=int
                    )
                else:
                    sample_indices = np.array(sample_indices_str)
                subset_classifications = [classifications[i] for i in sample_indices]
                labels_to_use = subset_classifications
            else:
                labels_to_use = classifications
            
            # Plot by classification
            create_2d_scatter_plot(
                embeddings,
                labels_to_use,
                f"{method_name.upper()} Visualization by Classification",
                color_column="classification",
                save_path=str(output_path / f"{method_name}_by_classification.html")
            )
    
    # 4. 3D plots for PCA
    if "pca" in methods and methods["pca"]["embeddings"].shape[1] >= 3:
        print("4. Creating 3D PCA plot...")
        create_3d_scatter_plot(
            methods["pca"]["embeddings"],
            classifications,
            "PCA 3D Visualization by Classification",
            color_column="classification",
            save_path=str(output_path / "pca_3d_by_classification.html")
        )
    
    # 5. Clustering comparison plots
    print("5. Creating clustering comparison plots...")
    for method_name, method_data in methods.items():
        if method_name == "pca" or "clustering" not in method_data:
            continue
        
        embeddings = method_data["embeddings"]
        clustering_results = method_data["clustering"]
        
        create_clustering_comparison_plot(
            embeddings,
            clustering_results,
            f"{method_name.upper()}",
            save_path=str(output_path / f"{method_name}_clustering_comparison.html")
        )
    
    # 6. Similarity heatmap
    print("6. Creating similarity heatmap...")
    # Use original embeddings for similarity (more meaningful)
    if "pca" in methods:
        create_similarity_heatmap(
            methods["pca"]["embeddings"],
            patent_ids,
            max_patents=50,  # Small subset for readability
            save_path=str(output_path / "patent_similarity_heatmap.html")
        )
    
    print(f"\n✅ All visualizations saved to {output_path}/")
    print(f"Generated plots:")
    for html_file in output_path.glob("*.html"):
        print(f"  - {html_file.name}")


def main() -> None:
    """Generate all embedding visualizations."""
    print("Patent Embedding Visualization Suite")
    print("=" * 50)
    
    try:
        generate_all_visualizations(
            results_dir="reduction_results",
            output_dir="visualizations"
        )
        
        print("\n🎨 Visualization complete!")
        print("\nOpen the HTML files in visualizations/ to view interactive plots")
        print("\nRecommended viewing order:")
        print("1. classification_distribution.html - Overview of data")
        print("2. pca_explained_variance.html - PCA analysis")
        print("3. umap_by_classification.html - UMAP clustering")
        print("4. tsne_by_classification.html - t-SNE clustering")
        print("5. *_clustering_comparison.html - Compare clustering methods")
        print("6. patent_similarity_heatmap.html - Patent relationships")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()