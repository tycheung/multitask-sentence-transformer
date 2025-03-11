import pytest
import tensorflow as tf
import numpy as np
import unittest.mock as mock
import os

from sentence_transformer.config import ModelConfig
from sentence_transformer.models.sentence_encoder import SentenceEncoder, create_sentence_encoder


@pytest.fixture
def model_config():
    """Create a test model configuration."""
    return ModelConfig(
        bert_model_name="bert-base-uncased",
        embedding_dim=768,
        max_seq_length=128,
        pooling_strategy="cls",
        trainable_bert=True
    )


@pytest.fixture
def test_sentences():
    """Create test sentences."""
    return [
        "I'll make him an offer he can't refuse.",
        "Keep your friends close, but your enemies closer."
    ]


@pytest.fixture(autouse=True)
def mock_hf_models(monkeypatch):
    """Mock Hugging Face models to avoid loading actual models."""
    from transformers import BertTokenizer, TFBertModel

    mock_tokenizer = mock.MagicMock(spec=BertTokenizer)
    mock_tokenizer.return_value = mock_tokenizer
    
    def mock_tokenizer_call(texts, padding='max_length', truncation=True, max_length=128, return_tensors="tf"):
        batch_size = 1
        if isinstance(texts, (list, tuple)) or (hasattr(texts, "shape") and len(texts.shape) > 0):
            batch_size = len(texts)

        return {
            "input_ids": tf.ones([batch_size, max_length], dtype=tf.int32),
            "attention_mask": tf.ones([batch_size, max_length], dtype=tf.int32),
            "token_type_ids": tf.zeros([batch_size, max_length], dtype=tf.int32)
        }
    
    mock_tokenizer.__call__ = mock_tokenizer_call
    mock_tokenizer.from_pretrained = mock.MagicMock(return_value=mock_tokenizer)

    mock_outputs = mock.MagicMock()
    mock_outputs.last_hidden_state = tf.ones([1, 128, 768]) * 0.1
    
    mock_bert_model = mock.MagicMock(spec=TFBertModel)
    mock_bert_model.return_value = mock_outputs
    mock_bert_model.from_pretrained = mock.MagicMock(return_value=mock_bert_model)

    monkeypatch.setattr('transformers.BertTokenizer', mock_tokenizer)
    monkeypatch.setattr('transformers.TFBertModel', mock_bert_model)


def test_encoder_init(model_config):
    """Test encoder initialization."""
    model = SentenceEncoder(model_config)
    assert model is not None
    assert model.embedding_dim == 768
    assert model.pooling_strategy == "cls"


def test_encoder_call(model_config, test_sentences):
    """Test encoder forward pass."""
    model = SentenceEncoder(model_config)
    sentences = tf.constant(test_sentences)
    outputs = model(sentences)

    assert "sentence_embedding" in outputs
    assert "sequence_output" in outputs
    assert "input_mask" in outputs

    batch_size = len(test_sentences)
    assert outputs["sentence_embedding"].shape == (batch_size, model_config.embedding_dim)
    assert len(outputs["sequence_output"].shape) == 3  # [batch_size, seq_len, hidden_dim]
    assert len(outputs["input_mask"].shape) == 2       # [batch_size, seq_len]


def test_create_sentence_encoder(model_config):
    """Test the factory function for creating sentence encoders."""
    model = create_sentence_encoder(model_config)
    assert isinstance(model, SentenceEncoder)
    assert model.embedding_dim == model_config.embedding_dim
    assert model.pooling_strategy == model_config.pooling_strategy


def test_embedding_consistency(model_config, test_sentences):
    """Test that the model produces consistent embeddings for the same input."""
    model = create_sentence_encoder(model_config)
    sentences = tf.constant(test_sentences)

    outputs1 = model(sentences)
    outputs2 = model(sentences)
    
    embeddings1 = outputs1["sentence_embedding"].numpy()
    embeddings2 = outputs2["sentence_embedding"].numpy()

    np.testing.assert_allclose(embeddings1, embeddings2, rtol=1e-5)


def test_different_pooling_strategies(test_sentences):
    """Test different pooling strategies."""
    pooling_strategies = ["cls", "mean", "max"]
    
    for strategy in pooling_strategies:
        config = ModelConfig(pooling_strategy=strategy)
        model = create_sentence_encoder(config)
        sentences = tf.constant(test_sentences)
        outputs = model(sentences)

        assert outputs["sentence_embedding"].shape == (len(test_sentences), config.embedding_dim)


def test_get_embedding():
    """Test the get_embedding method."""
    model = create_sentence_encoder()
    sentence = "This is a test sentence."

    embedding = model.get_embedding(sentence)

    assert embedding.shape == (1, model.embedding_dim)


def test_save_model(tmpdir, model_config):
    """Test model saving functionality."""
    model = create_sentence_encoder(model_config)

    export_path = str(tmpdir.mkdir("test_model"))

    model.save_model(export_path)

    assert len(tmpdir.listdir()) > 0

    assert os.path.exists(f"{export_path}/bert_model")
    assert os.path.exists(f"{export_path}/tokenizer")