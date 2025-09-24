"""Semantic search and analysis tools for patent embeddings."""

import json
from typing import Any

import numpy as np
import ollama
from sklearn.metrics.pairwise import cosine_similarity


def load_patent_embeddings(
    file_path: str = "patent_abstracts_with_embeddings.jsonl",
    model_name: str = "embeddinggemma"
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """
    Load patent embeddings and metadata.

    Args:
        file_path: Path to JSONL file with embeddings
        model_name: Embedding model name

    Returns:
        Tuple of (embeddings_matrix, patent_records)
    """
    embeddings = []
    records = []

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
            records.append(record)

    embeddings_matrix = np.array(embeddings, dtype=np.float32)
    print(f"Loaded {len(records)} patents with {embeddings_matrix.shape[1]}D embeddings")

    return embeddings_matrix, records


class PatentSemanticSearch:
    """Semantic search engine for patents."""

    def __init__(
        self,
        embeddings: np.ndarray,
        records: list[dict[str, Any]],
        model_name: str = "embeddinggemma"
    ):
        """
        Initialize semantic search engine.

        Args:
            embeddings: Patent embeddings matrix
            records: Patent record data
            model_name: Embedding model name for generating query embeddings
        """
        self.embeddings = embeddings
        self.records = records
        self.model_name = model_name
        self.ollama_client = ollama.Client()

        # Pre-compute norms for efficiency
        self.embedding_norms = np.linalg.norm(embeddings, axis=1)

        print(f"Initialized semantic search with {len(records)} patents")

    def generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate embedding for search query.

        Args:
            query: Search query text

        Returns:
            Query embedding vector
        """
        try:
            response = self.ollama_client.embeddings(model=self.model_name, prompt=query)
            return np.array(response["embedding"], dtype=np.float32)
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            raise

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_similarity: float = 0.0
    ) -> list[dict[str, Any]]:
        """
        Perform semantic search.

        Args:
            query: Search query
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of search results with similarity scores
        """
        print(f"Searching for: '{query}'")

        # Generate query embedding
        query_embedding = self.generate_query_embedding(query)

        # Compute similarities
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]

        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            similarity = similarities[idx]
            if similarity < min_similarity:
                break

            record = self.records[idx].copy()
            result = {
                "patent_id": record.get("id", ""),
                "similarity": float(similarity),
                "abstract": record.get("abstract", ""),
                "classification": record.get("classification", ""),
                "full_text": record.get("full_text", "")[:500] + "...",  # Truncated
            }
            results.append(result)

        print(f"Found {len(results)} results")
        return results

    def find_similar_patents(
        self,
        patent_id: str,
        top_k: int = 10,
        exclude_self: bool = True
    ) -> list[dict[str, Any]]:
        """
        Find patents similar to a given patent.

        Args:
            patent_id: ID of reference patent
            top_k: Number of similar patents to find
            exclude_self: Whether to exclude the reference patent

        Returns:
            List of similar patents with similarity scores
        """
        # Find the reference patent
        ref_idx = None
        for i, record in enumerate(self.records):
            if record.get("id") == patent_id:
                ref_idx = i
                break

        if ref_idx is None:
            raise ValueError(f"Patent {patent_id} not found")

        print(f"Finding patents similar to {patent_id}")

        # Get reference embedding
        ref_embedding = self.embeddings[ref_idx]

        # Compute similarities
        similarities = cosine_similarity([ref_embedding], self.embeddings)[0]

        # Get top results
        top_indices = np.argsort(similarities)[::-1]

        if exclude_self:
            # Remove the reference patent itself
            top_indices = top_indices[top_indices != ref_idx]

        top_indices = top_indices[:top_k]

        results = []
        for idx in top_indices:
            similarity = similarities[idx]
            record = self.records[idx].copy()

            result = {
                "patent_id": record.get("id", ""),
                "similarity": float(similarity),
                "abstract": record.get("abstract", ""),
                "classification": record.get("classification", ""),
            }
            results.append(result)

        return results

    def analyze_classification_similarities(self) -> dict[str, Any]:
        """
        Analyze similarities within and between patent classifications.

        Returns:
            Analysis results
        """
        print("Analyzing classification similarities...")

        # Group patents by classification
        class_groups = {}
        for i, record in enumerate(self.records):
            classification = record.get("classification", "unknown")
            if classification not in class_groups:
                class_groups[classification] = []
            class_groups[classification].append(i)

        # Compute intra-class and inter-class similarities
        intra_class_sims = {}
        inter_class_sims = {}

        for class_name, indices in class_groups.items():
            if len(indices) < 2:
                continue

            # Intra-class similarities
            class_embeddings = self.embeddings[indices]
            class_similarities = cosine_similarity(class_embeddings)

            # Remove diagonal (self-similarities)
            mask = ~np.eye(class_similarities.shape[0], dtype=bool)
            intra_sims = class_similarities[mask]

            intra_class_sims[class_name] = {
                "mean": float(np.mean(intra_sims)),
                "std": float(np.std(intra_sims)),
                "count": len(indices)
            }

        # Inter-class similarities
        class_names = list(class_groups.keys())
        for i, class_a in enumerate(class_names):
            for j, class_b in enumerate(class_names[i+1:], i+1):
                indices_a = class_groups[class_a]
                indices_b = class_groups[class_b]

                embeddings_a = self.embeddings[indices_a]
                embeddings_b = self.embeddings[indices_b]

                cross_similarities = cosine_similarity(embeddings_a, embeddings_b)

                pair_name = f"{class_a}-{class_b}"
                inter_class_sims[pair_name] = {
                    "mean": float(np.mean(cross_similarities)),
                    "std": float(np.std(cross_similarities)),
                    "count": len(cross_similarities.flat)
                }

        return {
            "intra_class": intra_class_sims,
            "inter_class": inter_class_sims,
            "class_counts": {k: len(v) for k, v in class_groups.items()}
        }


def create_search_interface(search_engine: PatentSemanticSearch) -> None:
    """
    Create interactive search interface.

    Args:
        search_engine: Initialized search engine
    """
    print("\n" + "="*60)
    print("Patent Semantic Search Interface")
    print("="*60)
    print("Commands:")
    print("  search <query>     - Search patents by text")
    print("  similar <patent_id> - Find similar patents")
    print("  analyze            - Analyze classification similarities")
    print("  quit               - Exit")
    print("="*60)

    while True:
        try:
            command = input("\n> ").strip()

            if command == "quit":
                break
            elif command == "analyze":
                results = search_engine.analyze_classification_similarities()

                print("\nIntra-class similarities (within same class):")
                for class_name, stats in results["intra_class"].items():
                    print(f"  {class_name}: {stats['mean']:.3f} ± {stats['std']:.3f} ({stats['count']} patents)")

                print("\nTop inter-class similarities (between different classes):")
                sorted_inter = sorted(
                    results["inter_class"].items(),
                    key=lambda x: x[1]["mean"],
                    reverse=True
                )
                for pair_name, stats in sorted_inter[:10]:
                    print(f"  {pair_name}: {stats['mean']:.3f} ± {stats['std']:.3f}")

            elif command.startswith("search "):
                query = command[7:].strip()
                if not query:
                    print("Please provide a search query")
                    continue

                results = search_engine.search(query, top_k=5)

                print(f"\nTop {len(results)} results:")
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. {result['patent_id']} (similarity: {result['similarity']:.3f})")
                    print(f"   Classification: {result['classification']}")
                    print(f"   Abstract: {result['abstract'][:200]}...")

            elif command.startswith("similar "):
                patent_id = command[8:].strip()
                if not patent_id:
                    print("Please provide a patent ID")
                    continue

                try:
                    results = search_engine.find_similar_patents(patent_id, top_k=5)

                    print(f"\nTop 5 patents similar to {patent_id}:")
                    for i, result in enumerate(results, 1):
                        print(f"\n{i}. {result['patent_id']} (similarity: {result['similarity']:.3f})")
                        print(f"   Classification: {result['classification']}")
                        print(f"   Abstract: {result['abstract'][:200]}...")

                except ValueError as e:
                    print(f"Error: {e}")

            else:
                print("Unknown command. Type 'quit' to exit.")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    print("\nGoodbye!")


def main() -> None:
    """Main function to run semantic search demo."""
    print("Patent Semantic Search Tool")
    print("=" * 50)

    try:
        # Load embeddings and records
        print("Loading patent embeddings...")
        embeddings, records = load_patent_embeddings()

        # Initialize search engine
        search_engine = PatentSemanticSearch(embeddings, records)

        # Demo searches
        print("\n" + "="*50)
        print("DEMO SEARCHES")
        print("="*50)

        demo_queries = [
            "machine learning artificial intelligence",
            "medical device heart surgery",
            "battery energy storage lithium"
        ]

        for query in demo_queries:
            print(f"\nDemo search: '{query}'")
            results = search_engine.search(query, top_k=3)

            for i, result in enumerate(results, 1):
                print(f"  {i}. {result['patent_id']} (sim: {result['similarity']:.3f}) - {result['abstract'][:100]}...")

        # Find similar patents demo
        if records:
            sample_patent = records[0]["id"]
            print(f"\nDemo: Patents similar to {sample_patent}")
            similar_results = search_engine.find_similar_patents(sample_patent, top_k=3)

            for i, result in enumerate(similar_results, 1):
                print(f"  {i}. {result['patent_id']} (sim: {result['similarity']:.3f}) - {result['abstract'][:100]}...")

        # Interactive interface
        create_search_interface(search_engine)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
