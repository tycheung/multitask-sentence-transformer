import pytest
from sentence_transformer.config import (
    ModelConfig,
    MultitaskConfig,
    DataConfig,
    TrainingConfig,
    Config
)

def test_model_config_defaults():
    """Test default values for ModelConfig."""
    config = ModelConfig()
   
    assert config.bert_model_name == "bert-base-uncased"
    assert config.embedding_dim == 768
    assert config.max_seq_length == 128
    assert config.pooling_strategy == "cls"
    assert config.dropout_rate == 0.1
    assert config.trainable_bert is True

def test_model_config_custom():
    """Test custom values for ModelConfig."""
    config = ModelConfig(
        bert_model_name="bert-large-uncased",
        embedding_dim=1024,
        max_seq_length=64,
        pooling_strategy="mean",
        dropout_rate=0.2,
        trainable_bert=False
    )
   
    assert config.bert_model_name == "bert-large-uncased"
    assert config.embedding_dim == 1024
    assert config.max_seq_length == 64
    assert config.pooling_strategy == "mean"
    assert config.dropout_rate == 0.2
    assert config.trainable_bert is False

def test_multitask_config_defaults():
    """Test default values for MultitaskConfig."""
    config = MultitaskConfig()
   
    # Inherits from ModelConfig
    assert config.embedding_dim == 768
    assert config.max_seq_length == 128
   
    # MultitaskConfig specific
    assert config.classification_classes == 5
    assert config.classification_hidden_dims == [256]
    assert config.ner_classes == 9
    assert config.ner_hidden_dims == [256]
    assert config.use_contrastive_learning is True
    assert config.contrastive_margin == 0.5
   
    # Task weights
    assert "classification" in config.task_weights
    assert "ner" in config.task_weights
    assert "sentence_embedding" in config.task_weights

def test_multitask_config_custom():
    """Test custom values for MultitaskConfig."""
    config = MultitaskConfig(
        embedding_dim=1024,
        classification_classes=3,
        ner_classes=5,
        use_contrastive_learning=False,
        task_weights={
            "classification": 2.0,
            "ner": 0.5,
            "sentence_embedding": 0.1
        }
    )
   
    assert config.embedding_dim == 1024
    assert config.classification_classes == 3
    assert config.ner_classes == 5
    assert config.use_contrastive_learning is False
    assert config.task_weights["classification"] == 2.0
    assert config.task_weights["ner"] == 0.5
    assert config.task_weights["sentence_embedding"] == 0.1

def test_data_config_defaults():
    """Test default values for DataConfig."""
    config = DataConfig()
   
    assert config.train_data_path == "./training_data/train"
    assert config.eval_data_path == "./training_data/eval"
    assert config.test_data_path == "./training_data/test"
    assert config.do_lowercase is True
    assert config.max_seq_length == 128
    assert config.use_contrastive_pairs is True
   
    # Auto-generated values
    assert config.class_names is not None
    assert len(config.class_names) == 5
    assert config.ner_labels is not None
    assert len(config.ner_labels) == 9
    assert "O" in config.ner_labels
    assert "B-PER" in config.ner_labels

def test_training_config():
    """Test default values for TrainingConfig."""
    config = TrainingConfig()
    
    assert config.learning_rate == 2e-5
    assert config.num_epochs == 10
    assert config.batch_size == 32
    assert config.checkpoint_path == "./checkpoints"
    assert config.save_best_only is True
    assert config.max_grad_norm == 1.0
    assert config.weight_decay == 0.01
    assert config.warmup_proportion == 0.1
    assert config.mixed_precision is False
    assert config.early_stopping is True
    assert config.patience == 3
    
    assert config.use_gradual_unfreezing is True
    assert config.layer_unfreeze_epochs == [0, 2, 4, 6, 8, 10]
    assert len(config.layer_learning_rates) == 6
    assert config.lr_reduce_factor == 0.5
    assert config.lr_reduce_patience == 2
    assert config.min_lr == 1e-6
    
    # Test custom values
    custom_config = TrainingConfig(
        learning_rate=3e-5,
        num_epochs=5,
        batch_size=16,
        layer_unfreeze_epochs=[0, 3, 6],
        mixed_precision=True
    )
    
    assert custom_config.learning_rate == 3e-5
    assert custom_config.num_epochs == 5
    assert custom_config.batch_size == 16
    assert custom_config.layer_unfreeze_epochs == [0, 3, 6]
    assert len(custom_config.layer_learning_rates) == 3
    assert custom_config.mixed_precision is True

def test_main_config():
    """Test the main Config class."""
    config = Config()

    assert isinstance(config.model, ModelConfig)
    assert isinstance(config.data, DataConfig)
    assert isinstance(config.training, TrainingConfig)

    model_config = MultitaskConfig(embedding_dim=1024)
    data_config = DataConfig(do_lowercase=False)
    training_config = TrainingConfig(batch_size=64)
    
    custom_config = Config(
        model=model_config,
        data=data_config,
        training=training_config
    )

    assert isinstance(custom_config.model, MultitaskConfig)
    assert custom_config.model.embedding_dim == 1024
    assert custom_config.data.do_lowercase is False
    assert custom_config.training.batch_size == 64

    assert custom_config.model.classification_classes == 5
    assert custom_config.model.ner_classes == 9
    assert custom_config.model.use_contrastive_learning is True