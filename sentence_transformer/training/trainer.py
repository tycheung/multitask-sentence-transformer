import os
import time
from typing import Dict, List, Optional, Tuple, Union

import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tqdm.auto import tqdm

from sentence_transformer.config import Config
from sentence_transformer.models.multitask import MultitaskModel, TaskLoss
from sentence_transformer.models.sentence_encoder import SentenceEncoder
from sentence_transformer.training.callbacks import (
    GradualLayerUnfreezing
)


class Trainer:
    """Base trainer class for sentence transformer models."""
    
    def __init__(
        self, 
        model: Union[SentenceEncoder, MultitaskModel],
        config: Config,
        optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
    ):
        """Initialize the trainer.
        
        Args:
            model: The model to train
            config: Configuration for training
            optimizer: Optional optimizer (will create one if not provided)
        """
        self.model = model
        self.config = config

        if optimizer is None:
            self.optimizer = tf.keras.optimizers.legacy.Adam(
                learning_rate=config.training.learning_rate
            )
        else:
            self.optimizer = optimizer

        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.val_loss = tf.keras.metrics.Mean(name="val_loss")
        
        self.log_dir = os.path.join(
            config.training.checkpoint_path, "logs", time.strftime("%Y%m%d-%H%M%S")
        )
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)

        self.best_val_loss = float('inf')

        self.checkpoint_dir = os.path.join(
            config.training.checkpoint_path, "checkpoints"
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if config.training.mixed_precision:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    def train_step(self, batch, labels=None):
        """Perform a single training step."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def evaluate_step(self, batch, labels=None):
        """Perform a single evaluation step."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def train(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: Optional[tf.data.Dataset] = None,
        callbacks: List[Callback] = None
    ):
        """Train the model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            callbacks: List of Keras callbacks
        
        Returns:
            Training history
        """
        if callbacks is None:
            callbacks = []
            
        # Add gradual unfreezing callback if requested
        if self.config.training.use_gradual_unfreezing:
            # Check which model structure we're using
            if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'bert_model'):
                # For MultitaskModel
                bert_model = self.model.encoder.bert_model
            elif hasattr(self.model, 'bert_model'):
                # For SentenceEncoder
                bert_model = self.model.bert_model
            else:
                print("Warning: Could not find BERT model for gradual unfreezing")
                bert_model = None
                
            if bert_model is not None:
                callbacks.append(
                    GradualLayerUnfreezing(
                        bert_model,
                        self.config.training.layer_unfreeze_epochs,
                        self.config.training.layer_learning_rates,
                        self.optimizer
                    )
                )

        history = {
            "train_loss": [],
            "val_loss": []
        }
        
        # Begin training
        for epoch in range(self.config.training.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.config.training.num_epochs}")

            self.train_loss.reset_states()
            self.val_loss.reset_states()

            for callback in callbacks:
                callback.on_epoch_begin(epoch)
            
            # Training loop
            train_iter = tqdm(train_dataset, desc="Training")
            for batch_idx, batch in enumerate(train_iter):
                for callback in callbacks:
                    callback.on_batch_begin(batch_idx)

                loss = self.train_step(batch)
                self.train_loss.update_state(loss)

                train_iter.set_postfix({"loss": self.train_loss.result().numpy()})

                for callback in callbacks:
                    callback.on_batch_end(batch_idx)

            if val_dataset is not None:
                val_iter = tqdm(val_dataset, desc="Validation")
                for batch_idx, batch in enumerate(val_iter):
                    loss = self.evaluate_step(batch)
                    self.val_loss.update_state(loss)
                    
                    val_iter.set_postfix({"loss": self.val_loss.result().numpy()})

            train_loss_value = self.train_loss.result().numpy()
            val_loss_value = self.val_loss.result().numpy() if val_dataset else None
            
            with self.summary_writer.as_default():
                tf.summary.scalar("train_loss", train_loss_value, step=epoch)
                if val_loss_value is not None:
                    tf.summary.scalar("val_loss", val_loss_value, step=epoch)

            history["train_loss"].append(train_loss_value)
            if val_loss_value is not None:
                history["val_loss"].append(val_loss_value)

            summary = f"Epoch {epoch+1}: train_loss={train_loss_value:.4f}"
            if val_loss_value is not None:
                summary += f", val_loss={val_loss_value:.4f}"
            print(summary)

            if val_loss_value is not None and val_loss_value < self.best_val_loss:
                self.best_val_loss = val_loss_value
                checkpoint_path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch+1}.h5")
                self.model.save_weights(checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")

            for callback in callbacks:
                callback.on_epoch_end(epoch)
        
        return history
    
    def save_model(self, export_path):
        """Save the model for inference."""
        self.model.save_model(export_path)
        print(f"Model saved to {export_path}")


class SentenceEncoderTrainer(Trainer):
    """Trainer for the basic sentence encoder model."""
    
    def train_step(self, batch):
        """Perform a single training step for the sentence encoder.
        
        For the basic encoder, we use a triplet loss to train the model to produce
        similar embeddings for similar sentences and dissimilar for different ones.
        
        Args:
            batch: Tuple of (anchor, positive, negative) sentences
            
        Returns:
            Loss value
        """
        anchor, positive, negative = batch
        
        with tf.GradientTape() as tape:
            anchor_embedding = self.model(anchor)["sentence_embedding"]
            positive_embedding = self.model(positive)["sentence_embedding"]
            negative_embedding = self.model(negative)["sentence_embedding"]

            positive_distance = tf.reduce_sum(
                tf.square(anchor_embedding - positive_embedding), axis=1
            )
            negative_distance = tf.reduce_sum(
                tf.square(anchor_embedding - negative_embedding), axis=1
            )

            margin = 1.0

            loss = tf.maximum(0.0, positive_distance - negative_distance + margin)
            loss = tf.reduce_mean(loss)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        # Gradient clipping
        gradients, _ = tf.clip_by_global_norm(
            gradients, self.config.training.max_grad_norm
        )

        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss
    
    def evaluate_step(self, batch):
        """Perform a single evaluation step for the sentence encoder.
        
        Args:
            batch: Tuple of (anchor, positive, negative) sentences
            
        Returns:
            Loss value
        """
        anchor, positive, negative = batch

        anchor_embedding = self.model(anchor)["sentence_embedding"]
        positive_embedding = self.model(positive)["sentence_embedding"]
        negative_embedding = self.model(negative)["sentence_embedding"]

        positive_distance = tf.reduce_sum(
            tf.square(anchor_embedding - positive_embedding), axis=1
        )
        negative_distance = tf.reduce_sum(
            tf.square(anchor_embedding - negative_embedding), axis=1
        )
        
        margin = 1.0

        loss = tf.maximum(0.0, positive_distance - negative_distance + margin)
        loss = tf.reduce_mean(loss)
        
        return loss


class MultitaskTrainer(Trainer):
    """Trainer for the multi-task learning model."""
    
    def __init__(
        self, 
        model: MultitaskModel,
        config: Config,
        optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
    ):
        """Initialize the multi-task trainer.
        
        Args:
            model: The multi-task model to train
            config: Configuration for training
            optimizer: Optional optimizer (will create one if not provided)
        """
        super().__init__(model, config, optimizer)

        self.task_loss = TaskLoss(config.model)
        
        # Initialize task-specific metrics
        self.train_classification_loss = tf.keras.metrics.Mean(name="train_cls_loss")
        self.train_ner_loss = tf.keras.metrics.Mean(name="train_ner_loss")
        self.val_classification_loss = tf.keras.metrics.Mean(name="val_cls_loss")
        self.val_ner_loss = tf.keras.metrics.Mean(name="val_ner_loss")

        self.train_classification_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="train_cls_acc"
        )
        self.val_classification_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="val_cls_acc"
        )
        
        self.train_ner_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="train_ner_acc"
        )
        self.val_ner_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="val_ner_acc"
        )
    
    def train_step(self, batch):
        """Perform a single training step for the multi-task model."""
        inputs, targets = batch
        
        with tf.GradientTape() as tape:
            outputs = self.model(inputs, training=True)

            losses = self.task_loss.compute_total_loss(outputs, targets)
            total_loss = losses["total_loss"]

        gradients = tape.gradient(total_loss, self.model.trainable_variables)

        gradients, _ = tf.clip_by_global_norm(
            gradients, self.config.training.max_grad_norm
        )
        
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_classification_loss.update_state(losses["classification_loss"])
        self.train_ner_loss.update_state(losses["ner_loss"])

        self.train_classification_accuracy.update_state(
            targets["classification_labels"], 
            outputs["classification_logits"]
        )

        # Get the mask but truncate it to match the NER labels length
        mask = tf.cast(outputs["input_mask"], tf.bool)
        ner_labels_length = tf.shape(targets["ner_labels"])[1]
        truncated_mask = mask[:, :ner_labels_length]
        
        self.train_ner_accuracy.update_state(
            tf.boolean_mask(targets["ner_labels"], truncated_mask),
            tf.boolean_mask(outputs["ner_logits"][:, :ner_labels_length, :], truncated_mask)
        )
        
        return total_loss
    
    def evaluate_step(self, batch):
        """Perform a single evaluation step for the multi-task model."""
        inputs, targets = batch

        outputs = self.model(inputs, training=False)

        losses = self.task_loss.compute_total_loss(outputs, targets)
        total_loss = losses["total_loss"]

        self.val_classification_loss.update_state(losses["classification_loss"])
        self.val_ner_loss.update_state(losses["ner_loss"])

        self.val_classification_accuracy.update_state(
            targets["classification_labels"], 
            outputs["classification_logits"]
        )

        # Get the mask but truncate it to match the NER labels length
        mask = tf.cast(outputs["input_mask"], tf.bool)
        ner_labels_length = tf.shape(targets["ner_labels"])[1]
        truncated_mask = mask[:, :ner_labels_length]
        
        self.val_ner_accuracy.update_state(
            tf.boolean_mask(targets["ner_labels"], truncated_mask),
            tf.boolean_mask(outputs["ner_logits"][:, :ner_labels_length, :], truncated_mask)
        )
        
        return total_loss