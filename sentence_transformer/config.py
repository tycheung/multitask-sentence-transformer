from dataclasses import dataclass
from typing import List, Union


@dataclass
class ModelConfig:
    """Model configuration settings."""
    bert_model_name: str = "bert-base-uncased"
    embedding_dim: int = 768
    max_seq_length: int = 128
    pooling_strategy: str = "cls"  # supported: cls, mean, max
    dropout_rate: float = 0.1
    trainable_bert: bool = True


@dataclass
class MultitaskConfig(ModelConfig):
    """Multi-task model configuration settings."""
    classification_classes: int = 5
    classification_hidden_dims: List[int] = None
    
    ner_classes: int = 9  # B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC, O
    ner_hidden_dims: List[int] = None
    
    use_contrastive_learning: bool = True
    contrastive_margin: float = 0.5
    
    task_weights: dict = None
    
    def __post_init__(self):
        if self.classification_hidden_dims is None:
            self.classification_hidden_dims = [256]
        if self.ner_hidden_dims is None:
            self.ner_hidden_dims = [256]
        if self.task_weights is None:
            self.task_weights = {
                "sentence_embedding": 1.0,
                "classification": 1.0,
                "ner": 1.0
            }


@dataclass
class TrainingConfig:
    """Training configuration settings."""
    learning_rate: float = 2e-5
    num_epochs: int = 10
    batch_size: int = 32

    checkpoint_path: str = "./checkpoints"
    save_best_only: bool = True

    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    warmup_proportion: float = 0.1
    mixed_precision: bool = False

    early_stopping: bool = True
    patience: int = 3
    
    use_gradual_unfreezing: bool = True
    layer_unfreeze_epochs: List[int] = None
    layer_learning_rates: List[float] = None

    lr_reduce_factor: float = 0.5
    lr_reduce_patience: int = 2
    min_lr: float = 1e-6
    
    def __post_init__(self):
        if self.layer_unfreeze_epochs is None:
            self.layer_unfreeze_epochs = list(range(0, 12, 2))
        
        if self.layer_learning_rates is None:
            self.layer_learning_rates = [
                self.learning_rate * (0.9 ** i) for i in range(len(self.layer_unfreeze_epochs))
            ]


@dataclass
class DataConfig:
    """Data configuration settings."""
    train_data_path: str = "./training_data/train"
    eval_data_path: str = "./training_data/eval"
    test_data_path: str = "./training_data/test"
    
    do_lowercase: bool = True
    max_seq_length: int = 128
    
    class_names: List[str] = None
    
    ner_labels: List[str] = None
    
    use_contrastive_pairs: bool = True
    
    def __post_init__(self):
        if self.class_names is None:
            self.class_names = ["class_1", "class_2", "class_3", "class_4", "class_5"]
        
        if self.ner_labels is None:
            self.ner_labels = [
                "O", "B-PER", "I-PER", "B-ORG", "I-ORG", 
                "B-LOC", "I-LOC", "B-MISC", "I-MISC"
            ]


@dataclass
class Config:
    """Main configuration container."""
    model: Union[ModelConfig, MultitaskConfig] = None
    data: DataConfig = None
    training: TrainingConfig = None
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.training is None:
            self.training = TrainingConfig()