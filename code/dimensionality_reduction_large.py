"""Enhanced dimensionality reduction and clustering for large patent datasets."""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from umap import UMAP


def load_large_patent_embeddings(
    filename: str,
    max_records: int | None = None,
    embedding_key: str = "embeddinggemma"
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Load patent embeddings from JSONL file, optimized for large datasets."""
    print(f"Loading patent embeddings from {filename}...")
    
    embeddings = []
    records = []
    
    with open(filename, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_records and i >= max_records:
                break
                
            if i % 1000 == 0:
                print(f"Loaded {i} records...")
            
            try:
                record = json.loads(line.strip())
                
                # Extract embedding
                if "embeddings" in record and embedding_key in record["embeddings"]:
                    embedding_data = record["embeddings"][embedding_key]
                    embedding = np.array(embedding_data["embedding"])
                    embeddings.append(embedding)
                    
                    # Keep essential metadata only
                    clean_record = {
                        "id": record["id"],
                        "classification": record["classification"],
                        "abstract_length": record.get("abstract_length", len(record.get("abstract", ""))),
                        "text": record.get("abstract", "")[:200] + "..." if len(record.get("abstract", "")) > 200 else record.get("abstract", "")
                    }
                    records.append(clean_record)
                    
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error processing record {i}: {e}")
                continue
    
    embeddings_array = np.array(embeddings)
    print(f"Loaded {len(embeddings_array)} embeddings with shape {embeddings_array.shape}")
    
    return embeddings_array, records


def efficient_dimensionality_reduction(
    embeddings: np.ndarray,
    methods: list[str] = ["pca", "umap", "tsne"],
    sample_for_tsne: int = 5000,
    sample_for_mds: int = 1000,
    n_components_pca: int = 50,
    save_dir: str = "reduction_results_large"
) -> dict[str, Any]:
    """Perform dimensionality reduction optimized for large datasets."""
    
    results_dir = Path(save_dir)
    results_dir.mkdir(exist_ok=True)
    
    print(f"Running dimensionality reduction on {len(embeddings)} embeddings...")
    print(f"Available methods: {methods}")
    print(f"Results will be saved to: {results_dir}")
    
    results = {"methods": {}, "processing_info": {}}
    
    # Standardize embeddings for better performance
    print("Standardizing embeddings...")
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # PCA - Always do this first as preprocessing for other methods
    if "pca" in methods or len(embeddings) > 10000:  # Always use PCA for large datasets
        print(f"\\nRunning PCA (768D -> {n_components_pca}D)...")
        start_time = time.time()
        
        pca = PCA(n_components=n_components_pca, random_state=42)
        pca_embeddings = pca.fit_transform(embeddings_scaled)
        
        pca_time = time.time() - start_time
        variance_explained = pca.explained_variance_ratio_.cumsum()[-1]
        
        # Save PCA results
        np.save(results_dir / "pca_embeddings.npy", pca_embeddings)
        
        results["methods"]["pca"] = {
            "embeddings_shape": pca_embeddings.shape,
            "variance_explained": float(variance_explained),
            "processing_time": pca_time,
            "info": {
                "n_components": n_components_pca,
                "explained_variance_ratios": pca.explained_variance_ratio_.tolist()
            }
        }
        
        print(f"PCA completed in {pca_time:.2f}s - Variance explained: {variance_explained:.3f}")
    else:
        pca_embeddings = embeddings_scaled
    
    # UMAP - Generally faster than t-SNE and handles large datasets well
    if "umap" in methods:
        print("\\nRunning UMAP (768D -> 2D)...")
        start_time = time.time()
        
        # Use PCA preprocessed data for efficiency
        input_data = pca_embeddings if len(embeddings) > 5000 else embeddings_scaled
        
        umap = UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )
        umap_embeddings = umap.fit_transform(input_data)
        
        umap_time = time.time() - start_time
        
        # Save UMAP results
        np.save(results_dir / "umap_embeddings.npy", umap_embeddings)
        
        results["methods"]["umap"] = {
            "embeddings_shape": umap_embeddings.shape,
            "processing_time": umap_time,
            "info": {
                "n_neighbors": 15,
                "min_dist": 0.1,
                "input_preprocessing": "pca" if len(embeddings) > 5000 else "standardized"
            }
        }
        
        print(f"UMAP completed in {umap_time:.2f}s")
    
    # t-SNE - Sample for large datasets due to O(n²) complexity
    if "tsne" in methods:
        tsne_sample_size = min(sample_for_tsne, len(embeddings))
        print(f"\\nRunning t-SNE on {tsne_sample_size} samples (20D -> 2D)...")
        
        # Sample randomly for t-SNE
        if tsne_sample_size < len(embeddings):
            sample_indices = np.random.choice(len(embeddings), tsne_sample_size, replace=False)
            tsne_input = pca_embeddings[sample_indices]
        else:
            sample_indices = np.arange(len(embeddings))
            tsne_input = pca_embeddings
        
        # Use first 20 PCA components for t-SNE
        tsne_input_reduced = tsne_input[:, :20] if tsne_input.shape[1] > 20 else tsne_input
        
        start_time = time.time()
        tsne = TSNE(
            n_components=2,
            perplexity=min(30, tsne_sample_size // 4),
            random_state=42,
            max_iter=1000,
            n_jobs=-1
        )
        tsne_embeddings = tsne.fit_transform(tsne_input_reduced)
        
        tsne_time = time.time() - start_time
        
        # Save t-SNE results
        np.save(results_dir / "tsne_embeddings.npy", tsne_embeddings)
        
        results["methods"]["tsne"] = {
            "embeddings_shape": tsne_embeddings.shape,
            "processing_time": tsne_time,
            "sample_size": tsne_sample_size,
            "kl_divergence": float(tsne.kl_divergence_) if hasattr(tsne, 'kl_divergence_') else None,
            "info": {
                "perplexity": min(30, tsne_sample_size // 4),
                "sample_indices": sample_indices.tolist() if tsne_sample_size < len(embeddings) else None,
                "input_preprocessing": "pca_20d"
            }
        }
        
        print(f"t-SNE completed in {tsne_time:.2f}s")
    
    # MDS - Only for small samples due to computational complexity
    if "mds" in methods and len(embeddings) <= 2000:
        mds_sample_size = min(sample_for_mds, len(embeddings))
        print(f"\\nRunning MDS on {mds_sample_size} samples...")
        
        if mds_sample_size < len(embeddings):
            sample_indices = np.random.choice(len(embeddings), mds_sample_size, replace=False)
            mds_input = embeddings_scaled[sample_indices]
        else:
            sample_indices = np.arange(len(embeddings))
            mds_input = embeddings_scaled
        
        start_time = time.time()
        mds = MDS(n_components=2, random_state=42, n_jobs=-1)
        mds_embeddings = mds.fit_transform(mds_input)
        
        mds_time = time.time() - start_time
        
        # Save MDS results
        np.save(results_dir / "mds_embeddings.npy", mds_embeddings)
        
        results["methods"]["mds"] = {
            "embeddings_shape": mds_embeddings.shape,
            "processing_time": mds_time,
            "sample_size": mds_sample_size,
            "stress": float(mds.stress_),
            "info": {
                "sample_indices": sample_indices.tolist() if mds_sample_size < len(embeddings) else None
            }
        }
        
        print(f"MDS completed in {mds_time:.2f}s")
    elif "mds" in methods:
        print(f"\\nSkipping MDS for large dataset ({len(embeddings)} samples > 2000)")
    
    return results


def efficient_clustering_analysis(
    results: dict[str, Any],
    embeddings: np.ndarray,
    max_samples_for_clustering: int = 10000,
    save_dir: str = "reduction_results_large"
) -> dict[str, Any]:
    """Perform clustering analysis optimized for large datasets."""
    
    print(f"\\nRunning clustering analysis...")
    results_dir = Path(save_dir)
    
    # Sample for clustering if dataset is too large
    if len(embeddings) > max_samples_for_clustering:
        print(f"Sampling {max_samples_for_clustering} embeddings for clustering analysis...")
        cluster_indices = np.random.choice(len(embeddings), max_samples_for_clustering, replace=False)
    else:
        cluster_indices = np.arange(len(embeddings))
    
    for method_name, method_data in results["methods"].items():
        if method_name == "pca":
            continue  # Skip PCA for clustering
        
        print(f"\\nClustering analysis for {method_name.upper()}...")
        
        # Load reduced embeddings
        embedding_file = results_dir / f"{method_name}_embeddings.npy"
        if not embedding_file.exists():
            print(f"Embeddings file {embedding_file} not found, skipping...")
            continue
        
        reduced_embeddings = np.load(embedding_file)
        
        # Handle sampling for t-SNE and MDS
        if method_name in ["tsne", "mds"] and "sample_indices" in method_data.get("info", {}):
            sample_indices = method_data["info"]["sample_indices"]
            if sample_indices:
                # Further sample if needed
                if len(sample_indices) > max_samples_for_clustering:
                    sub_indices = np.random.choice(len(sample_indices), max_samples_for_clustering, replace=False)
                    clustering_data = reduced_embeddings[sub_indices]
                    used_indices = [sample_indices[i] for i in sub_indices]
                else:
                    clustering_data = reduced_embeddings
                    used_indices = sample_indices
            else:
                clustering_data = reduced_embeddings[cluster_indices]
                used_indices = cluster_indices.tolist()
        else:
            clustering_data = reduced_embeddings[cluster_indices]
            used_indices = cluster_indices.tolist()
        
        method_data["clustering"] = {}
        
        # K-means clustering with different k values
        k_values = [3, 5, 7, 10] if len(clustering_data) > 50 else [min(3, len(clustering_data)//2)]
        
        for k in k_values:
            if k >= len(clustering_data):
                continue
                
            print(f"  K-means (k={k})...")
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans_labels = kmeans.fit_predict(clustering_data)
            
            silhouette_avg = silhouette_score(clustering_data, kmeans_labels)
            
            method_data["clustering"][f"kmeans_{k}"] = {
                "algorithm": "kmeans",
                "n_clusters": k,
                "silhouette_score": float(silhouette_avg),
                "inertia": float(kmeans.inertia_),
                "sample_size": len(clustering_data)
            }
            
            print(f"    Silhouette score: {silhouette_avg:.3f}")
        
        # DBSCAN clustering
        if len(clustering_data) <= 5000:  # Only for reasonable sizes
            print(f"  DBSCAN...")
            eps_values = [0.3, 0.5, 0.7]
            
            for eps in eps_values:
                dbscan = DBSCAN(eps=eps, min_samples=5, n_jobs=-1)
                dbscan_labels = dbscan.fit_predict(clustering_data)
                
                n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
                n_noise = list(dbscan_labels).count(-1)
                
                if n_clusters > 1:  # Only calculate silhouette if we have clusters
                    # Exclude noise points for silhouette calculation
                    non_noise_mask = dbscan_labels != -1
                    if np.sum(non_noise_mask) > 1:
                        silhouette_avg = silhouette_score(
                            clustering_data[non_noise_mask], 
                            dbscan_labels[non_noise_mask]
                        )
                    else:
                        silhouette_avg = -1
                else:
                    silhouette_avg = -1
                
                method_data["clustering"][f"dbscan_{eps}"] = {
                    "algorithm": "dbscan",
                    "eps": eps,
                    "n_clusters": n_clusters,
                    "n_noise": n_noise,
                    "silhouette_score": float(silhouette_avg),
                    "sample_size": len(clustering_data)
                }
                
                print(f"    eps={eps}: {n_clusters} clusters, silhouette: {silhouette_avg:.3f}")
    
    return results


def save_large_results(results: dict[str, Any], save_dir: str = "reduction_results_large") -> None:
    """Save results to JSON file."""
    results_path = Path(save_dir) / "reduction_results.json"
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\\nResults saved to {results_path}")


def print_large_analysis_summary(results: dict[str, Any]) -> None:
    """Print comprehensive analysis summary."""
    print("\\n" + "="*80)
    print("LARGE DATASET DIMENSIONALITY REDUCTION & CLUSTERING ANALYSIS")
    print("="*80)
    
    # Processing summary
    print("\\n📊 PROCESSING SUMMARY:")
    for method_name, method_data in results["methods"].items():
        shape = method_data["embeddings_shape"]
        time_taken = method_data["processing_time"]
        print(f"  {method_name.upper()}: {shape[0]} samples → {shape[1]}D in {time_taken:.1f}s")
        
        if method_name == "pca":
            variance = method_data["variance_explained"]
            print(f"    └─ Variance explained: {variance:.1%}")
    
    # Clustering results
    print("\\n🎯 CLUSTERING PERFORMANCE (Silhouette Scores):")
    
    clustering_results = []
    for method_name, method_data in results["methods"].items():
        if "clustering" not in method_data:
            continue
            
        for cluster_name, cluster_data in method_data["clustering"].items():
            score = cluster_data["silhouette_score"]
            clustering_results.append({
                "method": method_name.upper(),
                "algorithm": cluster_name,
                "score": score,
                "sample_size": cluster_data["sample_size"]
            })
    
    # Sort by silhouette score
    clustering_results.sort(key=lambda x: x["score"], reverse=True)
    
    print(f"{'Method':<8} {'Algorithm':<15} {'Score':<8} {'Sample Size':<12} {'Quality'}")
    print("-" * 60)
    
    for result in clustering_results:
        if result["score"] > 0.7:
            quality = "🟢 Excellent"
        elif result["score"] > 0.5:
            quality = "🟡 Good"
        elif result["score"] > 0.3:
            quality = "🟠 Fair"
        else:
            quality = "🔴 Poor"
            
        print(f"{result['method']:<8} {result['algorithm']:<15} {result['score']:<8.3f} "
              f"{result['sample_size']:<12} {quality}")
    
    # Best performance
    if clustering_results:
        best = clustering_results[0]
        print(f"\\n🏆 BEST PERFORMANCE: {best['method']} + {best['algorithm']} "
              f"(Silhouette: {best['score']:.3f})")
    
    print("\\n" + "="*80)


def main() -> None:
    """Main function for large dataset analysis."""
    import sys
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python dimensionality_reduction_large.py <embedding_file> [max_records]")
        print("Example: python dimensionality_reduction_large.py patent_abstracts_with_embeddings_large.jsonl 50000")
        return
    
    embedding_file = sys.argv[1]
    max_records = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    if not Path(embedding_file).exists():
        print(f"Error: File {embedding_file} not found")
        return
    
    print(f"Large Dataset Analysis Pipeline")
    print(f"Input file: {embedding_file}")
    print(f"Max records: {max_records or 'All'}")
    print("="*60)
    
    # Load embeddings
    embeddings, records = load_large_patent_embeddings(embedding_file, max_records)
    
    if len(embeddings) == 0:
        print("No embeddings found in file!")
        return
    
    # Run dimensionality reduction
    results = efficient_dimensionality_reduction(embeddings)
    
    # Run clustering analysis
    results = efficient_clustering_analysis(results, embeddings)
    
    # Save results
    save_large_results(results)
    
    # Print summary
    print_large_analysis_summary(results)
    
    print(f"\\n✅ Analysis complete! Check 'reduction_results_large/' for detailed results.")


if __name__ == "__main__":
    main()