import pytest
import tensorflow as tf
import os
import pandas as pd
import tempfile
import shutil

from sentence_transformer.config import Config, DataConfig
from sentence_transformer.data.dataset import (
    TripletSentenceDataset,
    MultitaskDataset,
    create_dataset
)


@pytest.fixture
def data_config():
    """Create a test data configuration."""
    return DataConfig(
        train_data_path="training_data/train",
        eval_data_path="training_data/eval",
        test_data_path="training_data/test",
        do_lowercase=True,
        max_seq_length=128,
        class_names=["positive", "neutral", "negative"],
        ner_labels=["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"],
        use_contrastive_pairs=True
    )


@pytest.fixture
def config(data_config):
    """Create a full test configuration."""
    config = Config()
    config.data = data_config
    return config


@pytest.fixture
def sample_triplet_dir(request):
    """Create sample triplet data directory with Godfather quotes for testing."""
    temp_dir = tempfile.mkdtemp()
    
    data = {
        "anchor": [
            "I'll make him an offer he can't refuse.",
            "Keep your friends close, but your enemies closer."
        ],
        "positive": [
            "I'm gonna make him an offer he can't refuse.",
            "Hold your friends close, but your enemies closer."
        ],
        "negative": [
            "The weather is nice today.",
            "I enjoy programming in Python."
        ]
    }
    
    # Create triplets.csv in the directory
    triplets_file = os.path.join(temp_dir, "triplets.csv")
    pd.DataFrame(data).to_csv(triplets_file, index=False)
    
    # Also create a data.csv as a copy for testing fallback
    data_file = os.path.join(temp_dir, "data.csv")
    pd.DataFrame(data).to_csv(data_file, index=False)
    
    if not hasattr(request.module, 'temp_dirs_to_cleanup'):
        request.module.temp_dirs_to_cleanup = []
    request.module.temp_dirs_to_cleanup.append(temp_dir)
    
    return temp_dir


@pytest.fixture
def sample_triplet_data(sample_triplet_dir):
    """Return the path to triplets.csv in the sample directory."""
    return os.path.join(sample_triplet_dir, "triplets.csv")


@pytest.fixture
def sample_multitask_dir(request):
    """Create sample multi-task data directory with Godfather quotes for testing."""
    temp_dir = tempfile.mkdtemp()
    
    data = {
        "text": [
            "I'll make him an offer he can't refuse.",
            "Keep your friends close, but your enemies closer.",
            "Luca Brasi sleeps with the fishes.",
            "You broke my heart, Fredo. You broke my heart."
        ],
        "class_label": [
            "negative",
            "neutral",
            "negative",
            "negative"
        ],
        "ner_labels": [
            "[]",
            "[]",
            "[\"B-PER\", \"I-PER\", \"O\", \"O\", \"O\", \"O\", \"O\"]",
            "[\"O\", \"O\", \"O\", \"O\", \"B-PER\", \"O\", \"O\", \"O\", \"O\", \"O\"]"
        ],
        "contrastive_label": [
            0,
            1,
            0,
            0
        ]
    }
    
    # Create multitask_data.csv in the directory
    multitask_file = os.path.join(temp_dir, "multitask_data.csv")
    pd.DataFrame(data).to_csv(multitask_file, index=False)
    
    # Also create a data.csv as a copy for testing fallback
    data_file = os.path.join(temp_dir, "data.csv")
    pd.DataFrame(data).to_csv(data_file, index=False)
    
    if not hasattr(request.module, 'temp_dirs_to_cleanup'):
        request.module.temp_dirs_to_cleanup = []
    request.module.temp_dirs_to_cleanup.append(temp_dir)
    
    return temp_dir


@pytest.fixture
def sample_multitask_data(sample_multitask_dir):
    """Return the path to multitask_data.csv in the sample directory."""
    return os.path.join(sample_multitask_dir, "multitask_data.csv")


def test_triplet_dataset_init(data_config):
    """Test triplet dataset initialization."""
    dataset = TripletSentenceDataset(data_config)
    assert dataset is not None
    assert dataset.max_seq_length == data_config.max_seq_length
    assert dataset.do_lowercase == data_config.do_lowercase


def test_triplet_dataset_load(data_config, sample_triplet_data):
    """Test loading triplet data."""
    dataset = TripletSentenceDataset(data_config)
    samples = dataset.load_samples(sample_triplet_data)

    assert "anchor" in samples
    assert "positive" in samples
    assert "negative" in samples

    assert len(samples["anchor"]) == 2
    assert len(samples["positive"]) == 2
    assert len(samples["negative"]) == 2

    if data_config.do_lowercase:
        assert samples["anchor"][0] == samples["anchor"][0].lower()


def test_triplet_dataset_load_directory(data_config, sample_triplet_dir):
    """Test loading triplet data from a directory."""
    dataset = TripletSentenceDataset(data_config)
    samples = dataset.load_samples(sample_triplet_dir)

    assert "anchor" in samples
    assert "positive" in samples
    assert "negative" in samples

    assert len(samples["anchor"]) == 2


def test_triplet_dataset_tf_dataset(data_config, sample_triplet_data):
    """Test creating TensorFlow dataset from triplet data."""
    dataset = TripletSentenceDataset(data_config)
    samples = dataset.load_samples(sample_triplet_data)

    batch_size = 2
    tf_dataset = dataset.create_tf_dataset(samples, batch_size=batch_size)

    assert isinstance(tf_dataset, tf.data.Dataset)

    for batch in tf_dataset.take(1):
        assert len(batch) == 3  # anchor, positive, negative
        anchors, positives, negatives = batch

        assert anchors.shape[0] == batch_size
        assert positives.shape[0] == batch_size
        assert negatives.shape[0] == batch_size


def test_multitask_dataset_init(data_config):
    """Test multi-task dataset initialization."""
    dataset = MultitaskDataset(data_config)
    assert dataset is not None
    assert dataset.max_seq_length == data_config.max_seq_length
    assert dataset.do_lowercase == data_config.do_lowercase

    assert dataset.class_to_id["positive"] == 0
    assert dataset.class_to_id["neutral"] == 1
    assert dataset.class_to_id["negative"] == 2
    
    assert dataset.ner_label_to_id["O"] == 0
    assert dataset.ner_label_to_id["B-PER"] == 1
    assert dataset.ner_label_to_id["I-PER"] == 2


def test_multitask_dataset_load(data_config, sample_multitask_data):
    """Test loading multi-task data."""
    dataset = MultitaskDataset(data_config)
    samples = dataset.load_samples(sample_multitask_data, include_contrastive=True)

    assert "texts" in samples
    assert "classification_labels" in samples
    assert "ner_labels" in samples
    assert "contrastive_labels" in samples

    assert len(samples["texts"]) == 4
    assert len(samples["classification_labels"]) == 4
    assert len(samples["ner_labels"]) == 4
    assert len(samples["contrastive_labels"]) == 4
    
    # Check label conversion
    assert samples["classification_labels"][0] == 2  # "negative" -> 2
    assert samples["classification_labels"][1] == 1  # "neutral" -> 1


def test_multitask_dataset_load_directory(data_config, sample_multitask_dir):
    """Test loading multi-task data from a directory."""
    dataset = MultitaskDataset(data_config)
    samples = dataset.load_samples(sample_multitask_dir, include_contrastive=True)

    assert "texts" in samples
    assert "classification_labels" in samples
    assert "ner_labels" in samples
    assert "contrastive_labels" in samples

    assert len(samples["texts"]) == 4


def test_multitask_dataset_tf_dataset(data_config, sample_multitask_data):
    """Test creating TensorFlow dataset from multi-task data."""
    dataset = MultitaskDataset(data_config)
    samples = dataset.load_samples(sample_multitask_data, include_contrastive=True)

    batch_size = 2
    tf_dataset = dataset.create_tf_dataset(
        samples,
        batch_size=batch_size,
        use_contrastive_pairs=False
    )
    
    assert isinstance(tf_dataset, tf.data.Dataset)
    
    for batch in tf_dataset.take(1):
        assert len(batch) == 2  # inputs, targets
        inputs, targets = batch

        assert inputs.shape[0] == batch_size
        assert targets["classification_labels"].shape[0] == batch_size
        assert targets["ner_labels"].shape[0] == batch_size
        assert targets["contrastive_labels"].shape[0] == batch_size


def test_multitask_dataset_contrastive_pairs(data_config, sample_multitask_data):
    """Test creating contrastive pairs from multi-task data."""
    dataset = MultitaskDataset(data_config)
    samples = dataset.load_samples(sample_multitask_data, include_contrastive=False)

    paired_samples = dataset.create_contrastive_pairs(samples)

    assert "is_pair" in paired_samples
    assert any(paired_samples["is_pair"])

    texts = paired_samples["texts"]
    is_pair = paired_samples["is_pair"]
    classification_labels = paired_samples["classification_labels"]
    
    pair_count = sum(is_pair) // 2  # Each pair has two texts
    assert pair_count > 0
    
    for i in range(0, len(texts) - 1, 2):
        if i+1 < len(is_pair) and is_pair[i] and is_pair[i+1]:
            assert classification_labels[i] == classification_labels[i+1], \
                f"Pair at indexes {i},{i+1} has mismatched labels: {classification_labels[i]} vs {classification_labels[i+1]}"
            
            assert len(texts[i]) > 0, f"Empty text at index {i}"
            assert len(texts[i+1]) > 0, f"Empty text at index {i+1}"


def test_create_dataset_factory(config, sample_triplet_data, sample_multitask_data):
    """Test the dataset factory function."""
    # Test with explicit file paths
    dataset_obj, tf_dataset = create_dataset(
        config,
        sample_triplet_data,
        batch_size=2,
        is_multitask=False
    )
    assert isinstance(dataset_obj, TripletSentenceDataset)
    assert isinstance(tf_dataset, tf.data.Dataset)
    
    dataset_obj, tf_dataset = create_dataset(
        config,
        sample_multitask_data,
        batch_size=2,
        is_multitask=True
    )
    assert isinstance(dataset_obj, MultitaskDataset)
    assert isinstance(tf_dataset, tf.data.Dataset)
    
    # Test with directory paths
    dataset_obj, tf_dataset = create_dataset(
        config,
        os.path.dirname(sample_triplet_data),
        batch_size=2,
        is_multitask=False
    )
    assert isinstance(dataset_obj, TripletSentenceDataset)
    assert isinstance(tf_dataset, tf.data.Dataset)
    
    dataset_obj, tf_dataset = create_dataset(
        config,
        os.path.dirname(sample_multitask_data),
        batch_size=2,
        is_multitask=True
    )
    assert isinstance(dataset_obj, MultitaskDataset)
    assert isinstance(tf_dataset, tf.data.Dataset)


def teardown_module(module):
    """Clean up temporary files after tests."""
    # Clean up files
    temp_files = getattr(module, 'temp_files_to_cleanup', [])
    for filepath in temp_files:
        if os.path.exists(filepath):
            os.unlink(filepath)
    
    # Clean up directories
    temp_dirs = getattr(module, 'temp_dirs_to_cleanup', [])
    for dirpath in temp_dirs:
        if os.path.exists(dirpath):
            shutil.rmtree(dirpath)