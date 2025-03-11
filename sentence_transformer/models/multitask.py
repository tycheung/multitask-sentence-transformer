import tensorflow as tf
from tensorflow.keras import layers, Model

from sentence_transformer.config import MultitaskConfig
from sentence_transformer.models.sentence_encoder import SentenceEncoder, create_sentence_encoder


class MultitaskModel(Model):
    """Multi-task learning model for sentence classification and NER.
    
    Based on a sentence transformer backbone with task-specific heads.
    """
    
    def __init__(self, config: MultitaskConfig):
        """Initialize the multi-task model.
        
        Args:
            config: Configuration for the multi-task model
        """
        super(MultitaskModel, self).__init__()
        
        self.config = config
        self.encoder = SentenceEncoder(config, trainable=config.trainable_bert)
        
        # Task A: Sentence Classification head
        self.classification_layers = []
        
        prev_dim = config.embedding_dim
        for dim in config.classification_hidden_dims:
            self.classification_layers.extend([
                layers.Dense(dim, activation="relu"),
                layers.Dropout(config.dropout_rate)
            ])
            prev_dim = dim
        
        self.classification_output = layers.Dense(
            config.classification_classes, 
            activation="softmax",
            name="classification_output"
        )
        
        # Task B: Named Entity Recognition head
        self.ner_layers = []
        
        prev_dim = config.embedding_dim
        for dim in config.ner_hidden_dims:
            self.ner_layers.extend([
                layers.Dense(dim, activation="relu"),
                layers.Dropout(config.dropout_rate)
            ])
            prev_dim = dim

        self.ner_output = layers.Dense(
            config.ner_classes,
            activation="softmax",
            name="ner_output"
        )
    
    def call(self, inputs, training=False):
        """Forward pass for the multi-task model.
        
        Args:
            inputs: Input text 
            training: Whether in training mode
            
        Returns:
            Dictionary of outputs for each task
        """
        encoder_outputs = self.encoder(inputs, training=training)
        sentence_embedding = encoder_outputs["sentence_embedding"]
        sequence_output = encoder_outputs["sequence_output"]
        input_mask = encoder_outputs["input_mask"]
        
        # Task A: Classification
        classification_features = sentence_embedding
        for layer in self.classification_layers:
            classification_features = layer(classification_features, training=training)
        classification_logits = self.classification_output(classification_features)
        
        # Task B: NER
        ner_features = sequence_output
        for layer in self.ner_layers:
            ner_features = layer(ner_features, training=training)
        ner_logits = self.ner_output(ner_features)
        
        return {
            "sentence_embedding": sentence_embedding,
            "classification_logits": classification_logits,
            "ner_logits": ner_logits,
            "input_mask": input_mask
        }
    
    def get_signature(self):
        """Get signature for TF Serving."""
        @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
        def serving_fn(sentences):
            outputs = self(sentences)
            return {
                "embeddings": outputs["sentence_embedding"],
                "classification_logits": outputs["classification_logits"],
                "ner_logits": outputs["ner_logits"]
            }
        
        return serving_fn
    
    def save_model(self, export_path):
        """Save the model for TensorFlow Serving."""
        # Save the encoder's BERT model and tokenizer
        self.encoder.bert_model.save_pretrained(f"{export_path}/bert_model")
        self.encoder.tokenizer.save_pretrained(f"{export_path}/tokenizer")
        
        # Save the full model as a TensorFlow SavedModel
        tf.saved_model.save(
            self,
            export_path,
            signatures={
                "serving_default": self.get_signature()
            }
        )


def create_multitask_model(config: MultitaskConfig = None) -> MultitaskModel:
    """Factory function to create a multi-task model.
    
    Args:
        config: Model configuration
        
    Returns:
        Initialized multi-task model
    """
    if config is None:
        config = MultitaskConfig()
    
    model = MultitaskModel(config)
    
    sample_text = tf.constant(["Hello world"])
    _ = model(sample_text)
    
    return model


class TaskLoss:
    """Loss functions for multi-task learning."""
    
    def __init__(self, config: MultitaskConfig):
        self.config = config
        self.task_weights = config.task_weights
    
    def classification_loss(self, y_true, y_pred, sample_weight=None):
        """Sparse categorical cross-entropy loss for classification."""
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        if sample_weight is not None:
            loss = loss * sample_weight
        return tf.reduce_mean(loss)
    
    def ner_loss(self, y_true, y_pred, input_mask=None):
        """Compute NER loss with masked values.
        
        Args:
            y_true: Ground truth NER labels with shape [batch_size, seq_len]
            y_pred: Predicted logits with shape [batch_size, padded_seq_len, num_classes]
            input_mask: Optional input mask with 1s for valid positions and 0s for padding
            
        Returns:
            Loss value
        """
        seq_len = tf.shape(y_true)[1]

        y_pred_truncated = y_pred[:, :seq_len, :]

        mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)

        per_example_loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred_truncated, from_logits=True
        )

        per_example_loss = tf.reshape(per_example_loss, tf.shape(mask))

        masked_loss = per_example_loss * mask

        sum_mask = tf.reduce_sum(mask)
        if tf.greater(sum_mask, 0):
            loss = tf.reduce_sum(masked_loss) / sum_mask
        else:
            loss = tf.constant(0.0, dtype=tf.float32)
        
        return loss
    
    def contrastive_loss(self, sentence_embeddings, labels=None, margin=0.5):
        """Contrastive loss for sentence embeddings.
        
        If labels are provided, they represent which sentences should be considered similar.
        Otherwise, we assume a batch contains pairs: [anchor1, pos1, anchor2, pos2, ...]
        
        Args:
            sentence_embeddings: Tensor of sentence embeddings
            labels: Optional tensor of labels for computing similarity
            margin: Margin to separate negative examples
            
        Returns:
            Contrastive loss value
        """
        embeddings_norm = tf.math.l2_normalize(sentence_embeddings, axis=1)
        
        similarity_matrix = tf.matmul(embeddings_norm, embeddings_norm, transpose_b=True)
        
        batch_size = tf.shape(sentence_embeddings)[0]
        
        if labels is not None:
            labels_expanded = tf.expand_dims(labels, 1)
            
            positive_mask = tf.equal(labels_expanded, tf.transpose(labels_expanded))
            
            positive_mask = tf.cast(positive_mask, tf.float32)
            identity_mask = tf.eye(batch_size, dtype=tf.float32)
            positive_mask = positive_mask * (1.0 - identity_mask)
            
            negative_mask = 1.0 - positive_mask - identity_mask
            
            positive_loss = tf.reduce_sum(1.0 - similarity_matrix * positive_mask) / (tf.reduce_sum(positive_mask) + 1e-10)
            
            negative_similarities = similarity_matrix * negative_mask
            negative_loss = tf.reduce_sum(tf.maximum(negative_similarities - margin, 0.0)) / (tf.reduce_sum(negative_mask) + 1e-10)
            
            return positive_loss + negative_loss
        else:
            even_indices = tf.range(0, batch_size, 2)
            odd_indices = tf.range(1, batch_size, 2)
            
            anchors = tf.gather(embeddings_norm, even_indices)
            positives = tf.gather(embeddings_norm, odd_indices)
            
            positive_similarities = tf.reduce_sum(anchors * positives, axis=1)
            positive_loss = tf.reduce_mean(1.0 - positive_similarities)
            
            anchor_idx_expanded = tf.expand_dims(even_indices, 1)
            all_idx = tf.range(batch_size)
            
            pair_mask = tf.ones((tf.shape(even_indices)[0], batch_size), dtype=tf.float32)
            
            for i in range(tf.shape(even_indices)[0]):
                pair_mask = tf.tensor_scatter_nd_update(
                    pair_mask,
                    [[i, odd_indices[i]]],
                    [0.0]
                )
                pair_mask = tf.tensor_scatter_nd_update(
                    pair_mask,
                    [[i, even_indices[i]]],
                    [0.0]
                )
            
            full_similarities = tf.matmul(anchors, embeddings_norm, transpose_b=True)
            masked_similarities = full_similarities * pair_mask
            
            negative_loss = tf.reduce_sum(
                tf.maximum(masked_similarities - margin, 0.0)
            ) / (tf.reduce_sum(pair_mask) + 1e-10)
            
            return positive_loss + negative_loss

    def compute_total_loss(self, model_outputs, targets):
        """Compute the total loss for all tasks.
        
        Args:
            model_outputs: Dictionary of model outputs
            targets: Dictionary of targets
            
        Returns:
            Dictionary of losses
        """
        input_mask = model_outputs.get("input_mask")
        
        # Calculate classification loss
        classification_loss = self.classification_loss(
            targets["classification_labels"],
            model_outputs["classification_logits"]
        )
        
        # Calculate NER loss with input mask
        ner_loss = self.ner_loss(
            targets["ner_labels"],
            model_outputs["ner_logits"],
            input_mask
        )
        
        # Calculate sentence embedding loss (contrastive loss) if contrastive labels are provided
        sentence_embedding_loss = tf.constant(0.0, dtype=tf.float32)
        if "contrastive_labels" in targets and self.config.use_contrastive_learning:
            sentence_embedding_loss = self.contrastive_loss(
                model_outputs["sentence_embedding"],
                targets.get("contrastive_labels"),
                margin=self.config.contrastive_margin
            )
        
        # Calculate total loss with task weights
        total_loss = (
            self.config.task_weights["classification"] * classification_loss +
            self.config.task_weights["ner"] * ner_loss +
            self.config.task_weights.get("sentence_embedding", 0.0) * sentence_embedding_loss
        )
        
        return {
            "total_loss": total_loss,
            "classification_loss": classification_loss,
            "ner_loss": ner_loss,
            "sentence_embedding_loss": sentence_embedding_loss
        }