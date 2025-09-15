"""Multi-method dimensionality reduction for patent embeddings."""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import umap


def load_embeddings_from_jsonl(
    file_path: str, model_name: str = "embeddinggemma"
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Load embeddings from JSONL file.

    Args:
        file_path: Path to JSONL file with embeddings
        model_name: Embedding model name to extract

    Returns:
        Tuple of (embeddings_matrix, patent_ids, classifications)
    """
    embeddings = []
    patent_ids = []
    classifications = []

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            
            record = json.loads(line)
            if "embeddings" not in record:
                continue

            # Find the embedding for the specified model
            embedding_data = None
            for emb in record.get("embeddings", []):
                if emb.get("model") == model_name:
                    embedding_data = emb
                    break

            if embedding_data is None:
                continue

            embedding = embedding_data.get("embedding", [])
            if not embedding:
                continue

            embeddings.append(embedding)
            patent_ids.append(record.get("id", ""))
            classifications.append(record.get("classification", ""))

    embeddings_matrix = np.array(embeddings, dtype=np.float32)
    print(f"Loaded {embeddings_matrix.shape[0]} embeddings with dimension {embeddings_matrix.shape[1]}")
    
    return embeddings_matrix, patent_ids, classifications


def apply_pca(
    embeddings: np.ndarray, n_components: int = 50
) -> Tuple[np.ndarray, PCA, Dict[str, Any]]:
    """
    Apply PCA dimensionality reduction.

    Args:
        embeddings: High-dimensional embeddings
        n_components: Number of PCA components

    Returns:
        Tuple of (reduced_embeddings, pca_model, info_dict)
    """
    print(f"Applying PCA with {n_components} components...")
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embeddings)
    
    explained_var_ratio = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var_ratio)
    
    info = {
        "method": "PCA",
        "n_components": n_components,
        "explained_variance_ratio": explained_var_ratio,
        "cumulative_variance": cumulative_var,
        "total_variance_explained": cumulative_var[-1],
    }
    
    print(f"PCA: Explained {info['total_variance_explained']:.3f} of total variance")
    
    return reduced, pca, info


def apply_tsne(
    embeddings: np.ndarray, 
    n_components: int = 2,
    perplexity: float = 30.0,
    random_state: int = 42
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply t-SNE dimensionality reduction.

    Args:
        embeddings: High-dimensional embeddings
        n_components: Number of output dimensions
        perplexity: t-SNE perplexity parameter
        random_state: Random seed

    Returns:
        Tuple of (reduced_embeddings, info_dict)
    """
    print(f"Applying t-SNE with perplexity={perplexity}...")
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        max_iter=1000,
        verbose=1
    )
    reduced = tsne.fit_transform(embeddings)
    
    info = {
        "method": "t-SNE",
        "n_components": n_components,
        "perplexity": perplexity,
        "random_state": random_state,
        "kl_divergence": tsne.kl_divergence_,
    }
    
    print(f"t-SNE completed with KL divergence: {info['kl_divergence']:.3f}")
    
    return reduced, info


def apply_umap(
    embeddings: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply UMAP dimensionality reduction.

    Args:
        embeddings: High-dimensional embeddings
        n_components: Number of output dimensions
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        random_state: Random seed

    Returns:
        Tuple of (reduced_embeddings, info_dict)
    """
    print(f"Applying UMAP with n_neighbors={n_neighbors}, min_dist={min_dist}...")
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        verbose=True
    )
    reduced = reducer.fit_transform(embeddings)
    
    info = {
        "method": "UMAP",
        "n_components": n_components,
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "random_state": random_state,
    }
    
    print(f"UMAP completed")
    
    return reduced, info


def apply_mds(
    embeddings: np.ndarray,
    n_components: int = 2,
    random_state: int = 42,
    max_samples: int = 1000
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply MDS dimensionality reduction.

    Args:
        embeddings: High-dimensional embeddings
        n_components: Number of output dimensions
        random_state: Random seed
        max_samples: Max samples for MDS (due to computational complexity)

    Returns:
        Tuple of (reduced_embeddings, info_dict)
    """
    # MDS is computationally expensive, so limit the sample size
    n_samples = min(len(embeddings), max_samples)
    if n_samples < len(embeddings):
        print(f"MDS: Using subset of {n_samples} samples (MDS is computationally expensive)")
        np.random.seed(random_state)
        indices = np.random.choice(len(embeddings), n_samples, replace=False)
        sample_embeddings = embeddings[indices]
    else:
        sample_embeddings = embeddings
        indices = np.arange(len(embeddings))
    
    print(f"Applying MDS with {n_components} components...")
    mds = MDS(
        n_components=n_components,
        random_state=random_state,
        dissimilarity='euclidean',
        n_jobs=-1
    )
    reduced = mds.fit_transform(sample_embeddings)
    
    info = {
        "method": "MDS",
        "n_components": n_components,
        "random_state": random_state,
        "stress": mds.stress_,
        "n_samples_used": n_samples,
        "sample_indices": indices,
    }
    
    print(f"MDS completed with stress: {info['stress']:.3f}")
    
    return reduced, info


def perform_clustering(
    reduced_embeddings: np.ndarray,
    method: str = "kmeans",
    n_clusters: int = 5
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Perform clustering on reduced embeddings.

    Args:
        reduced_embeddings: Dimensionality-reduced embeddings
        method: Clustering method ("kmeans" or "dbscan")
        n_clusters: Number of clusters (for k-means)

    Returns:
        Tuple of (cluster_labels, silhouette_score, info_dict)
    """
    print(f"Performing {method} clustering...")
    
    if method == "kmeans":
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = clusterer.fit_predict(reduced_embeddings)
        
        info = {
            "method": "k-means",
            "n_clusters": n_clusters,
            "inertia": clusterer.inertia_,
        }
        
    elif method == "dbscan":
        clusterer = DBSCAN(eps=0.5, min_samples=5)
        cluster_labels = clusterer.fit_predict(reduced_embeddings)
        
        unique_labels = np.unique(cluster_labels)
        n_clusters_found = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        info = {
            "method": "DBSCAN",
            "n_clusters_found": n_clusters_found,
            "n_noise_points": np.sum(cluster_labels == -1),
        }
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    # Calculate silhouette score
    if len(np.unique(cluster_labels)) > 1:
        silhouette = silhouette_score(reduced_embeddings, cluster_labels)
    else:
        silhouette = -1.0  # Invalid score if only one cluster
    
    print(f"{method} clustering: {len(np.unique(cluster_labels))} clusters, "
          f"silhouette score: {silhouette:.3f}")
    
    return cluster_labels, silhouette, info


def run_dimensionality_reduction_suite(
    input_file: str = "patent_abstracts_with_embeddings.jsonl",
    output_dir: str = "reduction_results",
    model_name: str = "embeddinggemma"
) -> Dict[str, Any]:
    """
    Run comprehensive dimensionality reduction analysis.

    Args:
        input_file: Input JSONL file with embeddings
        output_dir: Output directory for results
        model_name: Embedding model name

    Returns:
        Dictionary with all results
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("Loading embeddings...")
    embeddings, patent_ids, classifications = load_embeddings_from_jsonl(input_file, model_name)
    
    if len(embeddings) == 0:
        raise ValueError("No embeddings found")
    
    results = {
        "n_patents": len(embeddings),
        "embedding_dim": embeddings.shape[1],
        "patent_ids": patent_ids,
        "classifications": classifications,
        "methods": {}
    }
    
    # Apply different reduction methods
    print("\n" + "="*50)
    print("DIMENSIONALITY REDUCTION ANALYSIS")
    print("="*50)
    
    # 1. PCA (for preprocessing and analysis)
    pca_reduced, pca_model, pca_info = apply_pca(embeddings, n_components=50)
    results["methods"]["pca"] = {
        "reduced_embeddings": pca_reduced,
        "model": pca_model,
        "info": pca_info
    }
    
    # 2. t-SNE (on PCA-reduced data for efficiency)
    print(f"\nApplying t-SNE on PCA-reduced data...")
    tsne_reduced, tsne_info = apply_tsne(pca_reduced[:, :20])  # Use first 20 PCA components
    results["methods"]["tsne"] = {
        "reduced_embeddings": tsne_reduced,
        "info": tsne_info
    }
    
    # 3. UMAP (on original data)
    umap_reduced, umap_info = apply_umap(embeddings)
    results["methods"]["umap"] = {
        "reduced_embeddings": umap_reduced,
        "info": umap_info
    }
    
    # 4. MDS (on subset due to computational cost)
    mds_reduced, mds_info = apply_mds(embeddings, max_samples=500)
    results["methods"]["mds"] = {
        "reduced_embeddings": mds_reduced,
        "info": mds_info
    }
    
    # Clustering analysis for each method
    print("\n" + "="*30)
    print("CLUSTERING ANALYSIS")
    print("="*30)
    
    clustering_methods = ["kmeans", "dbscan"]
    for method_name, method_data in results["methods"].items():
        if method_name == "pca":  # Skip PCA for clustering (too many dimensions)
            continue
            
        reduced_emb = method_data["reduced_embeddings"]
        method_data["clustering"] = {}
        
        for cluster_method in clustering_methods:
            if cluster_method == "kmeans":
                # Try different numbers of clusters
                for n_clusters in [3, 5, 7, 10]:
                    labels, silhouette, cluster_info = perform_clustering(
                        reduced_emb, cluster_method, n_clusters
                    )
                    method_data["clustering"][f"{cluster_method}_{n_clusters}"] = {
                        "labels": labels,
                        "silhouette_score": silhouette,
                        "info": cluster_info
                    }
            else:
                labels, silhouette, cluster_info = perform_clustering(reduced_emb, cluster_method)
                method_data["clustering"][cluster_method] = {
                    "labels": labels,
                    "silhouette_score": silhouette,
                    "info": cluster_info
                }
    
    # Save results
    print(f"\nSaving results to {output_path}...")
    
    # Save embedding matrices and metadata
    for method_name, method_data in results["methods"].items():
        if "reduced_embeddings" in method_data:
            np.save(
                output_path / f"{method_name}_embeddings.npy",
                method_data["reduced_embeddings"]
            )
    
    # Save metadata
    metadata = {
        "n_patents": results["n_patents"],
        "embedding_dim": results["embedding_dim"],
        "patent_ids": results["patent_ids"],
        "classifications": results["classifications"],
    }
    
    # Save method info and clustering results
    methods_info = {}
    for method_name, method_data in results["methods"].items():
        info_data = method_data["info"].copy()
        if "clustering" in method_data:
            info_data["clustering"] = {}
            for cluster_name, cluster_data in method_data["clustering"].items():
                info_data["clustering"][cluster_name] = {
                    "silhouette_score": cluster_data["silhouette_score"],
                    "info": cluster_data["info"]
                }
        methods_info[method_name] = info_data
    
    metadata["methods"] = methods_info
    
    with open(output_path / "reduction_results.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"Results saved to {output_path}")
    
    return results


def main() -> None:
    """Run the dimensionality reduction suite."""
    print("Patent Embedding Dimensionality Reduction Suite")
    print("=" * 60)
    
    try:
        results = run_dimensionality_reduction_suite(
            input_file="patent_abstracts_with_embeddings.jsonl",
            output_dir="reduction_results"
        )
        
        print(f"\n✅ Analysis complete!")
        print(f"Processed {results['n_patents']} patents")
        print(f"Original embedding dimension: {results['embedding_dim']}")
        
        print(f"\nMethods applied:")
        for method_name, method_data in results["methods"].items():
            if "reduced_embeddings" in method_data:
                shape = method_data["reduced_embeddings"].shape
                print(f"  {method_name.upper()}: {shape[1]}D ({shape[0]} samples)")
        
        print(f"\nNext steps:")
        print(f"  1. Run visualization scripts")
        print(f"  2. Use 'embedding-atlas patent_embeddings_atlas.parquet' for interactive exploration")
        print(f"  3. Check reduction_results/ directory for detailed analysis")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()