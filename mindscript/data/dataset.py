"""
MindScript Dataset Handler
==========================
Data loading and preprocessing for cognitive pattern analysis.

Author: [Your Name]
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MindScriptDataset(Dataset):
    """
    Dataset for MindScript training and evaluation.
    
    Handles:
    - Text tokenization
    - Label normalization
    - Data augmentation (optional)
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: np.ndarray,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        dimension_names: Optional[List[str]] = None
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dimension_names = dimension_names or [
            "analytical", "creative", "social", "structured", "emotional"
        ]
        
        logger.info(f"Dataset initialized with {len(texts)} samples")
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        labels = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.float)
        }


class DataProcessor:
    """
    Process raw data for MindScript training.
    """
    
    def __init__(
        self,
        tokenizer_name: str = "distilbert-base-uncased",
        max_length: int = 512
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
        # Column mappings (essays.csv format)
        self.text_column = "TEXT"
        self.label_columns = ["cOPN", "cCON", "cEXT", "cAGR", "cNEU"]
        self.dimension_names = [
            "creative",    # Openness
            "structured",  # Conscientiousness
            "social",      # Extraversion
            "emotional",   # Agreeableness (inverted for our model)
            "analytical"   # Neuroticism (inverted for analytical)
        ]
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load and validate data"""
        logger.info(f"Loading data from {data_path}")
        
        df = pd.read_csv(data_path, encoding="latin-1")
        
        # Validate columns exist
        required_cols = [self.text_column] + self.label_columns
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        # Clean data
        df = df.dropna(subset=required_cols)
        df = df[df[self.text_column].str.len() > 50]  # Minimum text length
        
        logger.info(f"Loaded {len(df)} valid samples")
        
        return df
    
    def normalize_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Normalize labels to 0-1 range"""
        labels = df[self.label_columns].values
        
        # Min-max normalization per column
        labels_min = labels.min(axis=0)
        labels_max = labels.max(axis=0)
        labels_normalized = (labels - labels_min) / (labels_max - labels_min + 1e-8)
        
        return labels_normalized
    
    def prepare_datasets(
        self,
        data_path: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_state: int = 42
    ) -> Tuple[MindScriptDataset, MindScriptDataset, MindScriptDataset]:
        """
        Prepare train, validation, and test datasets.
        """
        # Load data
        df = self.load_data(data_path)
        
        # Extract texts and labels
        texts = df[self.text_column].tolist()
        labels = self.normalize_labels(df)
        
        # Shuffle
        np.random.seed(random_state)
        indices = np.random.permutation(len(texts))
        
        # Split
        n_train = int(len(texts) * train_ratio)
        n_val = int(len(texts) * val_ratio)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        # Create datasets
        train_texts = [texts[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        test_texts = [texts[i] for i in test_idx]
        
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        test_labels = labels[test_idx]
        
        train_dataset = MindScriptDataset(
            train_texts, train_labels, self.tokenizer, self.max_length
        )
        val_dataset = MindScriptDataset(
            val_texts, val_labels, self.tokenizer, self.max_length
        )
        test_dataset = MindScriptDataset(
            test_texts, test_labels, self.tokenizer, self.max_length
        )
        
        logger.info(f"Split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def create_dataloaders(
        self,
        train_dataset: MindScriptDataset,
        val_dataset: MindScriptDataset,
        test_dataset: MindScriptDataset,
        batch_size: int = 16,
        num_workers: int = 4
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create DataLoaders"""
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader