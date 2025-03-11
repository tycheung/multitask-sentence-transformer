import pytest
import tensorflow as tf
import numpy as np
import unittest.mock as mock

from sentence_transformer.config import MultitaskConfig
from sentence_transformer.models.multitask import MultitaskModel, create_multitask_model, TaskLoss


@pytest.fixture
def model_config():
    """Create a test multi-task model configuration."""
    return MultitaskConfig(
        embedding_dim=768,
        pooling_strategy="cls",
        dropout_rate=0.1,
        classification_classes=3,
        ner_classes=9,
        classification_hidden_dims=[256],
        ner_hidden_dims=[256],
        use_contrastive_learning=True,
        contrastive_margin=0.5,
        task_weights={
            "classification": 1.0,
            "ner": 1.0,
            "sentence_embedding": 0.5
        }
    )


@pytest.fixture
def test_sentences():
    """Create test sentences (Godfather quotes)."""
    return [
        "I'll make him an offer he can't refuse.",
        "Keep your friends close, but your enemies closer."
    ]


@pytest.fixture
def test_targets():
    """Create test targets for multi-task learning."""
    return {
        "classification_labels": tf.constant([2, 1]),  # negative, neutral
        "ner_labels": tf.constant([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # All "O" tags (padded)
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # All "O" tags (padded)
        ]),
        "contrastive_labels": tf.constant([0, 1])  # Different groups
    }


@pytest.fixture(autouse=True)
def mock_hub_and_encoder(monkeypatch):
    """Mock tensorflow_hub and encoder to avoid loading actual models."""
    from sentence_transformer.models import multitask
    
    class MockEncoder:
        def __init__(self, config):
            self.embedding_dim = config.embedding_dim
            self.bert_model = mock.MagicMock()
            self.bert_model.trainable_variables = []
        
        def __call__(self, inputs, training=False):
            batch_size = 1
            if isinstance(inputs, (list, tuple)) or (hasattr(inputs, "shape") and len(inputs.shape) > 0):
                batch_size = len(inputs)
            
            sentence_embedding = tf.ones([batch_size, 768]) * 0.1
            sequence_output = tf.ones([batch_size, 128, 768]) * 0.1
            input_mask = tf.ones([batch_size, 128], dtype=tf.int32)
            
            return {
                "sentence_embedding": sentence_embedding,
                "sequence_output": sequence_output,
                "input_mask": input_mask
            }

    def mock_create_encoder(config=None):
        return MockEncoder(config)
    
    monkeypatch.setattr(multitask, "create_sentence_encoder", mock_create_encoder)


def test_multitask_init(model_config):
    """Test multi-task model initialization."""
    model = MultitaskModel(model_config)
    assert model is not None

    assert model.encoder is not None
    assert model.encoder.embedding_dim == model_config.embedding_dim

    assert len(model.classification_layers) > 0
    assert model.classification_output is not None
    assert model.classification_output.units == model_config.classification_classes

    assert len(model.ner_layers) > 0
    assert model.ner_output is not None
    assert model.ner_output.units == model_config.ner_classes


def test_multitask_call(model_config, test_sentences):
    """Test multi-task model forward pass."""
    model = MultitaskModel(model_config)
    sentences = tf.constant(test_sentences)
    outputs = model(sentences)

    assert "sentence_embedding" in outputs
    assert "classification_logits" in outputs
    assert "ner_logits" in outputs
    assert "input_mask" in outputs

    batch_size = len(test_sentences)
    assert outputs["sentence_embedding"].shape == (batch_size, model_config.embedding_dim)
    assert outputs["classification_logits"].shape == (batch_size, model_config.classification_classes)
    assert len(outputs["ner_logits"].shape) == 3  # [batch_size, seq_len, num_ner_classes]
    assert outputs["ner_logits"].shape[2] == model_config.ner_classes


def test_create_multitask_model(model_config):
    """Test the factory function for creating multi-task models."""
    model = create_multitask_model(model_config)
    assert isinstance(model, MultitaskModel)


def test_task_loss(model_config, test_sentences, test_targets):
    """Test the task loss computation."""
    model = create_multitask_model(model_config)
    sentences = tf.constant(test_sentences)

    outputs = model(sentences)

    task_loss = TaskLoss(model_config)

    losses = task_loss.compute_total_loss(outputs, test_targets)

    assert "total_loss" in losses
    assert "classification_loss" in losses
    assert "ner_loss" in losses
    assert "sentence_embedding_loss" in losses

    assert losses["total_loss"].numpy() >= 0
    assert losses["classification_loss"].numpy() >= 0
    assert losses["ner_loss"].numpy() >= 0

    weights = model_config.task_weights
    expected_total = (
        weights["classification"] * losses["classification_loss"] +
        weights["ner"] * losses["ner_loss"] +
        weights["sentence_embedding"] * losses["sentence_embedding_loss"]
    )
    
    np.testing.assert_allclose(losses["total_loss"].numpy(), expected_total.numpy(), rtol=1e-5)


def test_contrastive_loss(model_config):
    """Test the contrastive loss computation."""
    task_loss = TaskLoss(model_config)
    
    # Create similar and dissimilar embeddings
    emb1 = tf.constant([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=tf.float32)
    emb2 = tf.constant([[0.9, 0.1, 0.0], [0.0, 0.0, 1.0]], dtype=tf.float32)
    
    # Case 1: With labels
    labels = tf.constant([0, 1])  # First pair similar, second pair different
    loss1 = task_loss.contrastive_loss(emb1, labels=labels)
    assert loss1.numpy() > 0
    
    # Case 2: Without labels (assuming pairs)
    # Create pair structure [anchor1, pos1, anchor2, pos2]
    paired_embs = tf.constant([
        [1.0, 0.0, 0.0],  # anchor1
        [0.9, 0.1, 0.0],  # similar to anchor1
        [0.0, 1.0, 0.0],  # anchor2
        [0.0, 0.0, 1.0]   # dissimilar to anchor2
    ], dtype=tf.float32)
    
    loss2 = task_loss.contrastive_loss(paired_embs)
    assert loss2.numpy() > 0


def test_save_model(tmpdir, model_config):
    """Test model saving functionality."""
    model = create_multitask_model(model_config)

    export_path = str(tmpdir.mkdir("test_multitask_model"))

    model.save_model(export_path)

    assert len(tmpdir.listdir()) > 0