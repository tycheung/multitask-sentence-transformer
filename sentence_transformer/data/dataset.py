import os
import json
import pandas as pd
import tensorflow as tf
from typing import Dict, List

from sentence_transformer.config import Config, DataConfig


class SentenceDataset:
    """Base dataset for sentence transformer training."""
    
    def __init__(self, config: DataConfig):
        """Initialize the dataset.
        
        Args:
            config: Data configuration
        """
        self.config = config
        self.max_seq_length = config.max_seq_length
        self.do_lowercase = config.do_lowercase
    
    def load_samples(self, data_path):
        """Load text samples from the data path.
        
        This method should be implemented by subclasses to load
        data specific to their needs.
        
        Args:
            data_path: Path to the data
            
        Returns:
            List of samples
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def preprocess_text(self, text):
        """Preprocess text before feeding to the model.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        if self.do_lowercase:
            text = text.lower()
        
        return text
    
    def create_tf_dataset(
        self,
        samples: List,
        batch_size: int,
        shuffle: bool = True,
        repeat: bool = False
    ) -> tf.data.Dataset:
        """Create a TensorFlow dataset from the samples.
        
        Args:
            samples: List of samples
            batch_size: Batch size
            shuffle: Whether to shuffle the dataset
            repeat: Whether to repeat the dataset
            
        Returns:
            TensorFlow dataset
        """
        dataset = tf.data.Dataset.from_tensor_slices(samples)
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(samples))
        
        dataset = dataset.batch(batch_size)
        
        if repeat:
            dataset = dataset.repeat()
        
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset


class TripletSentenceDataset(SentenceDataset):
    """Dataset for triplet loss training of the sentence encoder."""
    
    def load_samples(self, data_path):
        """Load triplet samples (anchor, positive, negative) from data path.
        
        Args:
            data_path: Path to the data
            
        Returns:
            Dictionary of anchor, positive, and negative sentences
        """
        # If data_path is a directory, look for triplets.csv in it
        if os.path.isdir(data_path):
            triplets_path = os.path.join(data_path, "triplets.csv")
            if os.path.exists(triplets_path):
                data_path = triplets_path
            else:
                # Try data.csv if triplets.csv doesn't exist
                data_csv_path = os.path.join(data_path, "data.csv")
                if os.path.exists(data_csv_path):
                    data_path = data_csv_path
                else:
                    files = [f for f in os.listdir(data_path) if f.endswith('.csv') or f.endswith('.json')]
                    if files:
                        data_path = os.path.join(data_path, files[0])
                    else:
                        raise FileNotFoundError(f"No suitable data file found in directory: {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        _, ext = os.path.splitext(data_path)
        
        if ext == ".csv":
            df = pd.read_csv(data_path)
            # Check if this is a triplet CSV (has anchor, positive, negative columns)
            if all(col in df.columns for col in ["anchor", "positive", "negative"]):
                anchors = df["anchor"].tolist()
                positives = df["positive"].tolist()
                negatives = df["negative"].tolist()
            else:
                raise ValueError(f"CSV file {data_path} does not contain required triplet columns")
        elif ext == ".json":
            with open(data_path, "r") as f:
                data = json.load(f)
            anchors = [item["anchor"] for item in data]
            positives = [item["positive"] for item in data]
            negatives = [item["negative"] for item in data]
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        anchors = [self.preprocess_text(text) for text in anchors]
        positives = [self.preprocess_text(text) for text in positives]
        negatives = [self.preprocess_text(text) for text in negatives]
        
        return {
            "anchor": anchors,
            "positive": positives,
            "negative": negatives
        }
    
    def create_tf_dataset(
        self,
        samples: Dict[str, List[str]],
        batch_size: int,
        shuffle: bool = True,
        repeat: bool = False
    ) -> tf.data.Dataset:
        """Create a TensorFlow dataset from the triplet samples.
        
        Args:
            samples: Dictionary of anchor, positive, and negative sentences
            batch_size: Batch size
            shuffle: Whether to shuffle the dataset
            repeat: Whether to repeat the dataset
            
        Returns:
            TensorFlow dataset
        """
        anchors = tf.constant(samples["anchor"])
        positives = tf.constant(samples["positive"])
        negatives = tf.constant(samples["negative"])
        
        dataset = tf.data.Dataset.from_tensor_slices(
            (anchors, positives, negatives)
        )
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(samples["anchor"]))
        
        dataset = dataset.batch(batch_size)
        
        if repeat:
            dataset = dataset.repeat()
        
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset


class MultitaskDataset(SentenceDataset):
    """Dataset for multi-task learning."""
    
    def __init__(self, config: DataConfig):
        """Initialize the multi-task dataset.
        
        Args:
            config: Data configuration
        """
        super().__init__(config)
        
        self.class_names = config.class_names
        self.ner_labels = config.ner_labels
        
        self.class_to_id = {
            name: i for i, name in enumerate(self.class_names)
        }
        
        self.ner_label_to_id = {
            label: i for i, label in enumerate(self.ner_labels)
        }
    
    def load_samples(self, data_path, include_contrastive=True):
        """Load multi-task samples from data path.
        
        Args:
            data_path: Path to the data
            include_contrastive: Whether to include contrastive learning data
            
        Returns:
            Dictionary containing text, classification labels, and NER labels
        """
        # If data_path is a directory, look for multitask_data.csv in it
        if os.path.isdir(data_path):
            multitask_path = os.path.join(data_path, "multitask_data.csv")
            if os.path.exists(multitask_path):
                data_path = multitask_path
            else:
                # Try data.csv if multitask_data.csv doesn't exist
                data_csv_path = os.path.join(data_path, "data.csv")
                if os.path.exists(data_csv_path):
                    data_path = data_csv_path
                else:
                    files = [f for f in os.listdir(data_path) if f.endswith('.csv') or f.endswith('.json')]
                    if files:
                        data_path = os.path.join(data_path, files[0])
                    else:
                        raise FileNotFoundError(f"No suitable data file found in directory: {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        _, ext = os.path.splitext(data_path)
        
        texts = []
        class_labels = []
        ner_labels = []
        contrastive_labels = [] if include_contrastive else None
        
        if ext == ".csv":
            df = pd.read_csv(data_path)
            # Check if this file has multitask columns
            required_cols = ["text", "class_label", "ner_labels"]
            if all(col in df.columns for col in required_cols):
                texts = df["text"].tolist()
                class_labels = df["class_label"].tolist()
                ner_labels = df["ner_labels"].tolist()
                
                if include_contrastive and "contrastive_label" in df.columns:
                    contrastive_labels = df["contrastive_label"].tolist()
            else:
                raise ValueError(f"CSV file {data_path} does not contain required multitask columns")
        elif ext == ".json":
            with open(data_path, "r") as f:
                data = json.load(f)
            texts = [item["text"] for item in data]
            class_labels = [item["class_label"] for item in data]
            ner_labels = [item["ner_labels"] for item in data]
            
            if include_contrastive and all("contrastive_label" in item for item in data):
                contrastive_labels = [item["contrastive_label"] for item in data]
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        texts = [self.preprocess_text(text) for text in texts]
        
        class_label_ids = [self.class_to_id[label] for label in class_labels]
        
        ner_label_ids = []
        for ner_sequence in ner_labels:
            if isinstance(ner_sequence, str):
                ner_sequence = json.loads(ner_sequence)
            
            label_ids = [self.ner_label_to_id[label] for label in ner_sequence]
            
            if len(label_ids) > self.max_seq_length:
                label_ids = label_ids[:self.max_seq_length]
            else:
                label_ids += [self.ner_label_to_id["O"]] * (self.max_seq_length - len(label_ids))
            
            ner_label_ids.append(label_ids)
        
        result = {
            "texts": texts,
            "classification_labels": class_label_ids,
            "ner_labels": ner_label_ids
        }
        
        if include_contrastive and contrastive_labels:
            result["contrastive_labels"] = contrastive_labels
        
        return result
    
    def create_contrastive_pairs(self, samples):
        """Create contrastive pairs from samples.
        
        This method creates pairs of similar sentences based on their class labels.
        Sentences with the same class label are considered similar.
        
        Args:
            samples: Dictionary of samples
            
        Returns:
            Dictionary with paired samples for contrastive learning
        """
        texts = samples["texts"]
        class_labels = samples["classification_labels"]
        ner_labels = samples["ner_labels"]
        
        class_groups = {}
        for i, label in enumerate(class_labels):
            if label not in class_groups:
                class_groups[label] = []
            class_groups[label].append(i)

        paired_texts = []
        paired_class_labels = []
        paired_ner_labels = []
        is_pair = []
        
        for label, indices in class_groups.items():
            if len(indices) < 2:
                continue
                
            for i in range(0, len(indices) - 1, 2):
                idx1 = indices[i]
                idx2 = indices[i + 1]
                
                paired_texts.append(texts[idx1])
                paired_texts.append(texts[idx2])
                
                paired_class_labels.append(class_labels[idx1])
                paired_class_labels.append(class_labels[idx2])
                
                paired_ner_labels.append(ner_labels[idx1])
                paired_ner_labels.append(ner_labels[idx2])
                
                is_pair.append(True)
                is_pair.append(True)
        
        return {
            "texts": paired_texts,
            "classification_labels": paired_class_labels,
            "ner_labels": paired_ner_labels,
            "is_pair": is_pair
        }
    
    def create_tf_dataset(
        self,
        samples: Dict,
        batch_size: int,
        shuffle: bool = True,
        repeat: bool = False,
        use_contrastive_pairs: bool = False
    ) -> tf.data.Dataset:
        """Create a TensorFlow dataset from the multi-task samples.
        
        Args:
            samples: Dictionary of texts and labels
            batch_size: Batch size
            shuffle: Whether to shuffle the dataset
            repeat: Whether to repeat the dataset
            use_contrastive_pairs: Whether to create pairs for contrastive learning
            
        Returns:
            TensorFlow dataset
        """
        if use_contrastive_pairs and "contrastive_labels" not in samples:
            samples = self.create_contrastive_pairs(samples)
        
        texts = tf.constant(samples["texts"])
        classification_labels = tf.constant(samples["classification_labels"])
        ner_labels = tf.constant(samples["ner_labels"])
        
        inputs = texts
        targets = {
            "classification_labels": classification_labels,
            "ner_labels": ner_labels
        }
        
        if "contrastive_labels" in samples:
            targets["contrastive_labels"] = tf.constant(samples["contrastive_labels"])
        
        if "is_pair" in samples:
            targets["is_pair"] = tf.constant(samples["is_pair"])
        
        dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(samples["texts"]))
        
        dataset = dataset.batch(batch_size)
        
        if repeat:
            dataset = dataset.repeat()
        
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset


def create_dataset(config: Config, data_path: str, batch_size: int = None, is_multitask: bool = False):
    """Factory function to create appropriate dataset.
    
    Args:
        config: Configuration
        data_path: Path to the data
        batch_size: Batch size (default: from config)
        is_multitask: Whether to create a multi-task dataset
        
    Returns:
        Dataset instance and TensorFlow dataset
    """
    if batch_size is None:
        batch_size = config.training.batch_size
    
    if is_multitask:
        dataset = MultitaskDataset(config.data)
    else:
        dataset = TripletSentenceDataset(config.data)
    
    samples = dataset.load_samples(data_path)
    tf_dataset = dataset.create_tf_dataset(samples, batch_size=batch_size)
    
    return dataset, tf_dataset