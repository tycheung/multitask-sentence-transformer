import argparse
import os
import json
import numpy as np
import tensorflow as tf

from sentence_transformer.config import Config, ModelConfig, MultitaskConfig
from sentence_transformer.data.dataset import TripletSentenceDataset, MultitaskDataset
from sentence_transformer.models.sentence_encoder import create_sentence_encoder
from sentence_transformer.models.multitask import create_multitask_model
from sentence_transformer.training.trainer import SentenceEncoderTrainer, MultitaskTrainer
from sentence_transformer.training.callbacks import GradualLayerUnfreezing


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate sentence transformer models"
    )
    
    parser.add_argument(
        "--mode", 
        type=str, 
        default="train", 
        choices=["train", "evaluate", "export"],
        help="Mode to run in"
    )
    
    parser.add_argument(
        "--model_type",
        type=str,
        default="encoder",
        choices=["encoder", "multitask"],
        help="Type of model to use"
    )
    
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to the configuration file"
    )
    
    parser.add_argument(
        "--train_data",
        type=str,
        default=None,
        help="Path to the training data"
    )
    
    parser.add_argument(
        "--val_data",
        type=str,
        default=None,
        help="Path to the validation data"
    )
    
    parser.add_argument(
        "--test_data",
        type=str,
        default=None,
        help="Path to the test data"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models",
        help="Directory to save models to"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for training and evaluation"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs to train for"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate for training"
    )
    
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to model checkpoint to load"
    )
    
    parser.add_argument(
        "--export_path",
        type=str,
        default=None,
        help="Path to export the model to"
    )
    
    parser.add_argument(
        "--use_contrastive",
        action="store_true",
        help="Use contrastive learning for multitask model"
    )
    
    return parser.parse_args()


def load_config(args):
    """Load configuration from file or defaults."""
    config = Config()

    if args.config_path and os.path.exists(args.config_path):
        with open(args.config_path, "r") as f:
            config_dict = json.load(f)
            if "model" in config_dict:
                if args.model_type == "multitask":
                    config.model = MultitaskConfig(**config_dict["model"])
                else:
                    config.model = ModelConfig(**config_dict["model"])

    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    
    if args.epochs is not None:
        config.training.num_epochs = args.epochs
    
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate

    if args.train_data is not None:
        config.data.train_data_path = args.train_data
    
    if args.val_data is not None:
        config.data.eval_data_path = args.val_data
    
    if args.test_data is not None:
        config.data.test_data_path = args.test_data

    if args.output_dir is not None:
        config.training.checkpoint_path = os.path.join(args.output_dir, "checkpoints")

    if args.use_contrastive and args.model_type == "multitask":
        config.model.use_contrastive_learning = True
    
    return config


def create_model(args, config):
    """Create model based on type and configuration."""
    if args.model_type == "multitask":
        model = create_multitask_model(config.model)
    else:
        model = create_sentence_encoder(config.model)

    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        model.load_weights(args.checkpoint_path).expect_partial()
        print(f"Loaded model weights from {args.checkpoint_path}")
    
    return model

def create_trainer(args, model, config):
    """Create trainer based on model type."""
    if args.model_type == "multitask":
        return MultitaskTrainer(model, config)
    else:
        return SentenceEncoderTrainer(model, config)


def create_dataset(config, data_path, batch_size=None, is_multitask=False):
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
        samples = dataset.load_samples(data_path, include_contrastive=config.model.use_contrastive_learning)

        use_contrastive_pairs = (
            config.model.use_contrastive_learning and 
            config.data.use_contrastive_pairs and 
            "contrastive_labels" not in samples
        )
        
        tf_dataset = dataset.create_tf_dataset(
            samples, 
            batch_size=batch_size,
            use_contrastive_pairs=use_contrastive_pairs
        )
    else:
        dataset = TripletSentenceDataset(config.data)
        samples = dataset.load_samples(data_path)
        tf_dataset = dataset.create_tf_dataset(samples, batch_size=batch_size)
    
    return dataset, tf_dataset


def train(args, config):
    """Train the model."""
    print("Creating model...")
    model = create_model(args, config)
    
    print("Creating datasets...")
    is_multitask = args.model_type == "multitask"

    _, train_dataset = create_dataset(
        config, 
        config.data.train_data_path, 
        is_multitask=is_multitask
    )

    val_dataset = None
    if config.data.eval_data_path and os.path.exists(config.data.eval_data_path):
        _, val_dataset = create_dataset(
            config, 
            config.data.eval_data_path, 
            is_multitask=is_multitask
        )
    
    print("Creating trainer...")
    trainer = create_trainer(args, model, config)

    callbacks = []
    if config.training.use_gradual_unfreezing:
        # Check if we have a multitask model or sentence encoder model
        if is_multitask:
            # Multitask model has encoder.bert_model structure
            bert_model = model.encoder.bert_model
        else:
            # SentenceEncoder has bert_model directly
            bert_model = model.bert_model
            
        callbacks.append(
            GradualLayerUnfreezing(
                bert_model,
                config.training.layer_unfreeze_epochs,
                config.training.layer_learning_rates,
                trainer.optimizer
            )
        )
    
    # Enable mixed precision if configured
    if config.training.mixed_precision:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
    
    print("Starting training...")
    history = trainer.train(train_dataset, val_dataset, callbacks=callbacks)

    export_path = args.export_path or os.path.join(args.output_dir, args.model_type)
    os.makedirs(export_path, exist_ok=True)
    
    print(f"Saving model to {export_path}...")
    trainer.save_model(export_path)
    
    return model, trainer, history


def evaluate(args, config):
    """Evaluate the model."""
    print("Creating model...")
    model = create_model(args, config)
    
    print("Creating dataset...")
    is_multitask = args.model_type == "multitask"

    _, test_dataset = create_dataset(
        config, 
        config.data.test_data_path, 
        is_multitask=is_multitask
    )
    
    print("Creating trainer...")
    trainer = create_trainer(args, model, config)
    
    print("Evaluating...")
    if is_multitask:
        total_loss = 0
        classification_accuracy = 0
        ner_accuracy = 0
        
        batches = 0
        for batch in test_dataset:
            loss = trainer.evaluate_step(batch)
            total_loss += loss
            batches += 1
        
        if batches > 0:
            total_loss /= batches
            classification_accuracy = trainer.val_classification_accuracy.result().numpy()
            ner_accuracy = trainer.val_ner_accuracy.result().numpy()
        
        print(f"Test loss: {total_loss:.4f}")
        print(f"Classification accuracy: {classification_accuracy:.4f}")
        print(f"NER accuracy: {ner_accuracy:.4f}")
    else:
        total_loss = 0
        batches = 0
        for batch in test_dataset:
            loss = trainer.evaluate_step(batch)
            total_loss += loss
            batches += 1
        
        if batches > 0:
            total_loss /= batches
        
        print(f"Test loss: {total_loss:.4f}")


def export(args, config):
    """Export the model for serving."""
    print("Creating model...")
    model = create_model(args, config)

    export_path = args.export_path or os.path.join(args.output_dir, args.model_type)
    os.makedirs(export_path, exist_ok=True)
    
    print(f"Exporting model to {export_path}...")
    model.save_model(export_path)


def main():
    """Main function."""
    np.random.seed(42)
    tf.random.set_seed(42)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    args = parse_args()

    config = load_config(args)

    if args.mode == "train":
        train(args, config)
    elif args.mode == "evaluate":
        evaluate(args, config)
    elif args.mode == "export":
        export(args, config)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()