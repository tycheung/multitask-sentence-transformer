import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from typing import List
import re


class GradualLayerUnfreezing(Callback):
    """Callback for gradual unfreezing of BERT layers during training.
    
    Gradually unfreeze layers of a BERT model starting from the top layer.
    Works with Hugging Face BERT models which have a different layer naming convention.
    """
    
    def __init__(
        self, 
        bert_model,
        unfreeze_epochs: List[int],
        layer_learning_rates: List[float],
        optimizer
    ):
        """Initialize the gradual unfreezing callback.
        
        Args:
            bert_model: The BERT model to unfreeze (Hugging Face TFBertModel)
            unfreeze_epochs: List of epochs at which to unfreeze layers
            layer_learning_rates: List of learning rates for each layer
            optimizer: The optimizer used for training
        """
        super().__init__()
        self.bert_model = bert_model
        self.unfreeze_epochs = unfreeze_epochs
        self.layer_learning_rates = layer_learning_rates
        self.optimizer = optimizer
        
        # Initialize layer groups
        self.bert_layers = []
        
        # Debug info
        print(f"Found {len(self.bert_model.trainable_variables)} trainable variables in BERT model")
        
        # Group variables by layer
        layer_pattern = re.compile(r'layer\.(\d+)')
        
        for var in self.bert_model.trainable_variables:
            # Check for embeddings
            if "embeddings" in var.name and not any(layer[0] == "embeddings" for layer in self.bert_layers):
                self.bert_layers.append(("embeddings", []))
            
            # Check for encoder layers (using regex to extract layer number)
            layer_match = layer_pattern.search(var.name)
            if layer_match:
                layer_num = int(layer_match.group(1))
                if layer_num not in [layer[0] for layer in self.bert_layers if isinstance(layer[0], int)]:
                    self.bert_layers.append((layer_num, []))
        
        # Assign variables to their respective layers
        for var in self.bert_model.trainable_variables:
            # Assign embeddings
            if "embeddings" in var.name:
                for i, (layer_name, layer_vars) in enumerate(self.bert_layers):
                    if layer_name == "embeddings":
                        self.bert_layers[i][1].append(var)
                        break
            
            # Assign encoder layers
            layer_match = layer_pattern.search(var.name)
            if layer_match:
                layer_num = int(layer_match.group(1))
                for i, (layer_name, layer_vars) in enumerate(self.bert_layers):
                    if layer_name == layer_num:
                        self.bert_layers[i][1].append(var)
                        break
        
        # Sort layers from top to bottom (higher layer number = closer to output)
        self.bert_layers.sort(key=lambda x: 999 if x[0] == "embeddings" else x[0], reverse=True)
        
        # Print layer info for debugging
        print(f"Organized BERT model into {len(self.bert_layers)} layer groups:")
        for layer_name, layer_vars in self.bert_layers:
            print(f"  Layer {layer_name}: {len(layer_vars)} variables")
        
        # Freeze all layers initially
        self._freeze_all_layers()
    
    def _freeze_all_layers(self):
        """Freeze all layers in the BERT model."""
        for layer_name, layer_vars in self.bert_layers:
            for var in layer_vars:
                var._trainable = False
    
    def _unfreeze_layer(self, layer_idx):
        """Unfreeze a specific layer in the BERT model.
        
        Args:
            layer_idx: Index of the layer to unfreeze
        """
        if 0 <= layer_idx < len(self.bert_layers):
            layer_name, layer_vars = self.bert_layers[layer_idx]
            for var in layer_vars:
                var._trainable = True

            layer_desc = f"Embedding" if layer_name == "embeddings" else f"Layer {layer_name}"
            print(f"Unfreezing {layer_desc} with {len(layer_vars)} variables")
    
    def _set_layer_learning_rate(self, layer_idx):
        """Set different learning rates for different layers.
        
        Args:
            layer_idx: Index of the newly unfrozen layer
        """
        if 0 <= layer_idx < len(self.layer_learning_rates):
            new_lr = self.layer_learning_rates[layer_idx]
            print(f"Setting learning rate to {new_lr}")

            self.optimizer.learning_rate.assign(new_lr)
    
    def on_epoch_begin(self, epoch, logs=None):
        """Called at the beginning of each epoch.
        
        Args:
            epoch: Current epoch index
            logs: Training logs
        """
        if epoch in self.unfreeze_epochs:
            idx = self.unfreeze_epochs.index(epoch)
            self._unfreeze_layer(idx)
            self._set_layer_learning_rate(idx)


class TensorBoardEmbeddingVisualizer(Callback):
    """Callback for visualizing embeddings in TensorBoard."""
    
    def __init__(
        self,
        model,
        dataset,
        log_dir: str,
        metadata=None,
        embedding_freq: int = 1
    ):
        """Initialize the TensorBoard embedding visualizer.
        
        Args:
            model: The model to visualize embeddings for
            dataset: Dataset containing text samples
            log_dir: Directory to write logs to
            metadata: Optional metadata for embeddings
            embedding_freq: Frequency (in epochs) to visualize embeddings
        """
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.log_dir = log_dir
        self.metadata = metadata
        self.embedding_freq = embedding_freq

        self.writer = tf.summary.create_file_writer(
            self.log_dir + "/embeddings"
        )
    
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch.
        
        Args:
            epoch: Current epoch index
            logs: Training logs
        """
        if (epoch + 1) % self.embedding_freq != 0:
            return

        all_embeddings = []
        all_texts = []
        
        for batch in self.dataset:
            if isinstance(batch, tuple):
                texts = batch[0]
            else:
                texts = batch

            embeddings = self.model(texts)["sentence_embedding"]
            
            all_embeddings.append(embeddings.numpy())
            all_texts.extend([text.numpy().decode('utf-8') for text in texts])

        all_embeddings = tf.concat(all_embeddings, axis=0)

        with self.writer.as_default():
            tf.summary.text("metadata", tf.constant(all_texts), step=epoch)

            tensor_embeddings = tf.Variable(all_embeddings)

            config = tf.compat.v1.summary.ProjectorConfig()
            embedding = config.embeddings.add()
            embedding.tensor_name = tensor_embeddings.name

            if self.metadata is not None:
                embedding.metadata_path = self.metadata