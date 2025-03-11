import pytest
import tensorflow as tf
import numpy as np
import unittest.mock as mock

from sentence_transformer.config import Config, ModelConfig, MultitaskConfig
from sentence_transformer.models.sentence_encoder import create_sentence_encoder
from sentence_transformer.models.multitask import create_multitask_model
from sentence_transformer.training.trainer import SentenceEncoderTrainer, MultitaskTrainer
from sentence_transformer.training.callbacks import GradualLayerUnfreezing


@pytest.fixture
def config():
    """Create a full test configuration."""
    return Config()


@pytest.fixture
def test_batch():
    """Create a test batch for training."""
    anchors = tf.constant(["I'll make him an offer he can't refuse."])
    positives = tf.constant(["I'm gonna make him an offer he can't refuse."])
    negatives = tf.constant(["The weather is nice today."])
    
    return anchors, positives, negatives


@pytest.fixture
def test_multitask_batch():
    """Create a test batch for multi-task training."""
    inputs = tf.constant(["I'll make him an offer he can't refuse.",
                         "Luca Brasi sleeps with the fishes."])

    targets = {
        "classification_labels": tf.constant([2, 2]),  # Both negative
        "ner_labels": tf.constant([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # All O tags (padded)
            [1, 2, 0, 0, 0, 0, 0, 0, 0, 0]   # B-PER, I-PER, O... (padded)
        ])
    }
    
    return inputs, targets


@pytest.fixture(autouse=True)
def mock_models(monkeypatch):
    """Mock models to avoid loading real ones."""
    def mock_create_encoder(config=None):
        mock_encoder = mock.MagicMock()
        mock_encoder.encoder = mock.MagicMock()
        mock_encoder.bert_model = mock.MagicMock()
        mock_encoder.trainable_variables = [tf.Variable(tf.ones([5, 5]))]

        def mock_call(inputs, training=False):
            batch_size = 1
            if isinstance(inputs, (list, tuple)) or (hasattr(inputs, "shape") and len(inputs.shape) > 0):
                batch_size = len(inputs)
            
            return {
                "sentence_embedding": tf.ones([batch_size, 768]) * 0.1,
                "sequence_output": tf.ones([batch_size, 128, 768]) * 0.1,
                "input_mask": tf.ones([batch_size, 128], dtype=tf.int32)
            }
        
        mock_encoder.side_effect = mock_call
        mock_encoder.__call__ = mock_call

        mock_encoder.save_weights = mock.MagicMock()
        mock_encoder.save_model = mock.MagicMock()
        
        return mock_encoder
    
    def mock_create_multitask(config=None):
        mock_model = mock.MagicMock()
        mock_model.encoder = mock.MagicMock()
        mock_model.encoder.bert_model = mock.MagicMock()
        mock_model.trainable_variables = [tf.Variable(tf.ones([5, 5]))]

        def mock_call(inputs, training=False):
            batch_size = 1
            if isinstance(inputs, (list, tuple)) or (hasattr(inputs, "shape") and len(inputs.shape) > 0):
                batch_size = len(inputs)
            
            return {
                "sentence_embedding": tf.ones([batch_size, 768]) * 0.1,
                "classification_logits": tf.ones([batch_size, 3]) * 0.1,
                "ner_logits": tf.ones([batch_size, 128, 9]) * 0.1,
                "input_mask": tf.ones([batch_size, 128], dtype=tf.int32)
            }
        
        mock_model.side_effect = mock_call
        mock_model.__call__ = mock_call

        mock_model.save_weights = mock.MagicMock()
        mock_model.save_model = mock.MagicMock()
        
        return mock_model

    import sentence_transformer.models.sentence_encoder
    import sentence_transformer.models.multitask
    
    monkeypatch.setattr(sentence_transformer.models.sentence_encoder, "create_sentence_encoder", mock_create_encoder)
    monkeypatch.setattr(sentence_transformer.models.multitask, "create_multitask_model", mock_create_multitask)


def test_sentence_encoder_trainer_init(config):
    """Test sentence encoder trainer initialization."""
    model = create_sentence_encoder()
    trainer = SentenceEncoderTrainer(model, config)
    
    assert trainer.model is not None
    assert trainer.config is config
    assert trainer.optimizer is not None
    assert trainer.train_loss is not None
    assert trainer.val_loss is not None


def test_sentence_encoder_train_step(config, test_batch):
    """Test training step for sentence encoder trainer."""
    model = create_sentence_encoder()
    trainer = SentenceEncoderTrainer(model, config)

    loss = trainer.train_step(test_batch)

    assert tf.is_tensor(loss)
    assert loss.shape == ()  # scalar
    assert loss.numpy() >= 0  # loss should be non-negative


def test_sentence_encoder_evaluate_step(config, test_batch):
    """Test evaluation step for sentence encoder trainer."""
    model = create_sentence_encoder()
    trainer = SentenceEncoderTrainer(model, config)

    loss = trainer.evaluate_step(test_batch)

    assert tf.is_tensor(loss)
    assert loss.shape == ()  # scalar
    assert loss.numpy() >= 0  # loss should be non-negative


def test_multitask_trainer_init(config):
    """Test multi-task trainer initialization."""
    config.model = MultitaskConfig()
    model = create_multitask_model(config.model)
    trainer = MultitaskTrainer(model, config)
    
    assert trainer.model is not None
    assert trainer.config is config
    assert trainer.optimizer is not None
    assert trainer.train_loss is not None
    assert trainer.val_loss is not None

    assert trainer.train_classification_loss is not None
    assert trainer.train_ner_loss is not None
    assert trainer.train_classification_accuracy is not None
    assert trainer.train_ner_accuracy is not None


def test_multitask_train_step(config, test_multitask_batch):
    """Test training step for multi-task trainer."""
    config.model = MultitaskConfig()
    model = create_multitask_model(config.model)
    trainer = MultitaskTrainer(model, config)

    loss = trainer.train_step(test_multitask_batch)

    assert tf.is_tensor(loss)
    assert loss.shape == ()  # scalar
    assert loss.numpy() >= 0  # loss should be non-negative


def test_multitask_evaluate_step(config, test_multitask_batch):
    """Test evaluation step for multi-task trainer."""
    config.model = MultitaskConfig()
    model = create_multitask_model(config.model)
    trainer = MultitaskTrainer(model, config)

    loss = trainer.evaluate_step(test_multitask_batch)

    assert tf.is_tensor(loss)
    assert loss.shape == ()  # scalar
    assert loss.numpy() >= 0  # loss should be non-negative


def test_gradual_unfreezing_callback():
    """Test the gradual unfreezing callback with a BERT-like model structure."""
    class EncoderLayer(tf.keras.layers.Layer):
        def __init__(self, index, **kwargs):
            super().__init__(name=f"encoder/layer_{index}", **kwargs)
            self.attention = tf.keras.layers.MultiHeadAttention(
                num_heads=2, key_dim=32, name="attention"
            )
            self.dense = tf.keras.layers.Dense(32, name="output/dense")
        
        def call(self, inputs):
            attn_output = self.attention(inputs, inputs)
            return self.dense(attn_output)

    inputs = tf.keras.Input(shape=(10, 32))
    x = inputs

    encoder_layers = []
    for i in range(3):  # Creating 3 encoder layers
        encoder_layer = EncoderLayer(i)
        encoder_layers.append(encoder_layer)
        x = encoder_layer(x)

    outputs = tf.keras.layers.Dense(1, name="outputs")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    sample_input = tf.random.normal((2, 10, 32))
    _ = model(sample_input)

    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

    for layer in encoder_layers:
        layer.trainable = False

    callback = GradualLayerUnfreezing(
        bert_model=model,
        unfreeze_epochs=[0, 1, 2],
        layer_learning_rates=[0.001, 0.0005, 0.0001],
        optimizer=optimizer
    )

    trainable_counts = []
    
    for epoch in range(3):
        callback.on_epoch_begin(epoch)

        trainable_vars = sum(1 for v in model.trainable_variables)
        trainable_counts.append(trainable_vars)

        current_lr = optimizer.learning_rate.numpy()
        expected_lr = [0.001, 0.0005, 0.0001][min(epoch, 2)]
        assert abs(current_lr - expected_lr) < 1e-6

    assert trainable_counts[-1] >= trainable_counts[0], "No layers were unfrozen"