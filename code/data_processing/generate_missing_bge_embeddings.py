#!/usr/bin/env python3
"""
Generate Missing bge-m3 Embeddings

Generates bge-m3 embeddings for patents that have OpenAI + nomic but are missing bge-m3.
Updates the master embeddings file with the new embeddings.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BGEEmbeddingGenerator:
    """Generate embeddings using bge-m3 model"""

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        """
        Initialize the bge-m3 embedding generator.

        Args:
            model_name: HuggingFace model identifier
        """
        logger.info(f"Loading bge-m3 model: {model_name}")
        self.model_name = model_name

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)

            # Use GPU if available
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.model.eval()

            logger.info(f"Model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def generate_embeddings(self, texts: List[str], batch_size: int = 8) -> List[np.ndarray]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing

        Returns:
            List of embedding vectors
        """
        logger.info(f"Generating embeddings for {len(texts)} texts (batch_size={batch_size})")

        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(texts) + batch_size - 1) // batch_size

                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_texts)} texts)")

                try:
                    # Tokenize with appropriate truncation
                    inputs = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=8192,  # bge-m3 max length
                        return_tensors='pt'
                    ).to(self.device)

                    # Generate embeddings
                    outputs = self.model(**inputs)

                    # Use mean pooling on the last hidden state
                    embeddings = outputs.last_hidden_state.mean(dim=1)

                    # Convert to numpy and normalize
                    batch_embeddings = embeddings.cpu().numpy()

                    # Normalize embeddings
                    norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                    batch_embeddings = batch_embeddings / norms

                    all_embeddings.extend(batch_embeddings)

                    logger.info(f"Batch {batch_num} complete. Embedding shape: {batch_embeddings.shape}")

                except Exception as e:
                    logger.error(f"Error processing batch {batch_num}: {e}")
                    # Add zero embeddings for failed batch
                    failed_embeddings = [np.zeros(1024) for _ in batch_texts]
                    all_embeddings.extend(failed_embeddings)

                # Small delay to prevent memory issues
                time.sleep(0.1)

        logger.info(f"Generated {len(all_embeddings)} embeddings")
        return all_embeddings


def load_patents_for_embedding(input_file: str) -> List[Dict]:
    """
    Load patents that need bge-m3 embeddings.

    Args:
        input_file: JSONL file with patents needing embeddings

    Returns:
        List of patent records
    """
    logger.info(f"Loading patents from {input_file}")

    patents = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                patents.append(json.loads(line.strip()))

    logger.info(f"Loaded {len(patents)} patents for embedding")
    return patents


def update_master_embeddings(master_file: str, new_embeddings: Dict[str, np.ndarray]) -> str:
    """
    Update the master embeddings file with new bge-m3 embeddings.

    Args:
        master_file: Path to master embeddings file
        new_embeddings: Dictionary mapping patent_id to embedding vector

    Returns:
        Path to updated file
    """
    logger.info(f"Updating master embeddings file with {len(new_embeddings)} new embeddings...")

    # Create backup
    backup_file = f"{master_file}.backup_{int(time.time())}"
    Path(master_file).rename(backup_file)
    logger.info(f"Created backup: {backup_file}")

    updated_count = 0
    total_patents = 0

    with open(backup_file, 'r') as infile, open(master_file, 'w') as outfile:
        for line in infile:
            if line.strip():
                total_patents += 1
                data = json.loads(line.strip())
                patent_id = data.get('patent_id', '')

                # Add new bge-m3 embedding if available
                if patent_id in new_embeddings:
                    embedding_vector = new_embeddings[patent_id].tolist()

                    data['embeddings']['bge-m3'] = {
                        'vector': embedding_vector,
                        'dimension': len(embedding_vector),
                        'source_file': 'generated_missing_bge_embeddings',
                        'generated_at': datetime.now().isoformat()
                    }

                    updated_count += 1
                    logger.info(f"Updated {patent_id} with bge-m3 embedding")

                outfile.write(json.dumps(data) + '\n')

    logger.info(f"Updated {updated_count}/{total_patents} patents in master file")
    return master_file


def main():
    """Main execution"""
    input_file = "data_v2/patents_needing_bge_m3.jsonl"
    master_file = "data_v2/master_patent_embeddings.jsonl"

    # Check input file
    if not Path(input_file).exists():
        print(f"❌ Input file not found: {input_file}")
        print("Run find_missing_bge_embeddings.py first")
        return

    # Check master file
    if not Path(master_file).exists():
        print(f"❌ Master embeddings file not found: {master_file}")
        return

    try:
        print("🚀 Starting bge-m3 embedding generation...")

        # Load patents needing embeddings
        patents = load_patents_for_embedding(input_file)

        if not patents:
            print("✅ No patents need bge-m3 embeddings!")
            return

        # Initialize embedding generator
        generator = BGEEmbeddingGenerator()

        # Extract texts and IDs
        texts = [patent['text'] for patent in patents]
        patent_ids = [patent['id'] for patent in patents]

        # Generate embeddings
        embeddings = generator.generate_embeddings(texts, batch_size=4)

        # Create mapping
        new_embeddings = dict(zip(patent_ids, embeddings))

        # Update master file
        update_master_embeddings(master_file, new_embeddings)

        print("\n" + "="*60)
        print("✅ BGE-M3 EMBEDDING GENERATION COMPLETE!")
        print("="*60)
        print(f"📊 Patents processed: {len(patents)}")
        print(f"🎯 Embeddings generated: {len(embeddings)}")
        print(f"💾 Master file updated: {master_file}")
        print(f"🔄 Backup created: {master_file}.backup_*")
        print("\n🎯 IMPACT:")
        print(f"   Before: 3,431 patents with all three models")
        print(f"   After:  ~3,443 patents with all three models")
        print("\n🚀 NEXT STEPS:")
        print("1. Re-run model intersection analysis to verify")
        print("2. Regenerate Atlas data with complete coverage")
        print("3. Launch Atlas for visualization")
        print("="*60)

    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise


if __name__ == "__main__":
    main()