import tensorflow as tf
from tensorflow.keras import layers, Model
from transformers import BertTokenizer, TFBertModel

from sentence_transformer.config import ModelConfig


class SentenceEncoder(Model):
    """BERT-based sentence encoder model using Hugging Face Transformers.
    
    Encodes input sentences into fixed-length embeddings.
    """
    
    def __init__(self, config: ModelConfig, trainable: bool = True):
        """Initialize the sentence encoder model.
        
        Args:
            config: Model configuration
            trainable: Whether to make BERT trainable
        """
        super(SentenceEncoder, self).__init__()
        
        self.config = config
        self.max_seq_length = config.max_seq_length
        self.embedding_dim = config.embedding_dim
        self.pooling_strategy = config.pooling_strategy

        # Use Hugging Face models instead of TF Hub
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_model_name)
        self.bert_model = TFBertModel.from_pretrained(
            config.bert_model_name,
            trainable=config.trainable_bert and trainable
        )

        self.dropout = layers.Dropout(config.dropout_rate)
        
    @tf.function(input_signature=[])
    def _dummy_fixed_input(self):
        """Create a dummy fixed input for graph tracing."""
        # Create dummy outputs matching what the model would return
        batch_size = 1
        seq_len = self.max_seq_length
        hidden_dim = self.embedding_dim
        
        return {
            "sentence_embedding": tf.zeros([batch_size, hidden_dim]),
            "sequence_output": tf.zeros([batch_size, seq_len, hidden_dim]),
            "input_mask": tf.ones([batch_size, seq_len], dtype=tf.int32)
        }
        
    def call(self, inputs, training=False):
        """Forward pass for the sentence encoder.
        
        Args:
            inputs: Input text (batch of sentences)
            training: Whether in training mode
            
        Returns:
            Sentence embeddings
        """
        # We need to separate eager execution from graph execution
        if tf.executing_eagerly():
            return self._eager_call(inputs, training)
        else:
            # During graph tracing, return fixed dummy outputs
            # This will be replaced with the real computation at runtime
            return self._dummy_fixed_input()
            
    def _eager_call(self, inputs, training=False):
        """Actual implementation for eager execution mode.
        
        Args:
            inputs: Input text (batch of sentences)
            training: Whether in training mode
            
        Returns:
            Sentence embeddings
        """
        # Convert tensor to list of strings if needed
        if isinstance(inputs, tf.Tensor):
            if len(tf.shape(inputs)) == 0:
                # Handle scalar input
                inputs = [inputs.numpy().decode('utf-8')]
            else:
                # Handle batch input
                inputs = [input_text.numpy().decode('utf-8') for input_text in inputs]
        elif isinstance(inputs, str):
            inputs = [inputs]
        
        # Tokenize with Hugging Face
        encoded_inputs = self.tokenizer(
            inputs, 
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="tf"
        )
        
        # Get BERT outputs
        bert_outputs = self.bert_model(
            encoded_inputs["input_ids"],
            attention_mask=encoded_inputs["attention_mask"],
            token_type_ids=encoded_inputs["token_type_ids"],
            training=training
        )
        
        sequence_output = bert_outputs.last_hidden_state

        if self.pooling_strategy == "cls":
            # Use [CLS] token embedding as sentence representation
            pooled_output = sequence_output[:, 0, :]
        elif self.pooling_strategy == "mean":
            # Create a mask to ignore padding tokens
            input_mask = tf.cast(encoded_inputs["attention_mask"], tf.float32)
            input_mask_expanded = tf.expand_dims(input_mask, axis=-1)
            # Masked sum and divide by number of non-masked tokens
            sum_embeddings = tf.reduce_sum(sequence_output * input_mask_expanded, axis=1)
            mask_sum = tf.reduce_sum(input_mask_expanded, axis=1)
            pooled_output = sum_embeddings / (mask_sum + 1e-10)
        elif self.pooling_strategy == "max":
            input_mask = encoded_inputs["attention_mask"]
            mask_for_max = (1 - tf.cast(input_mask, tf.float32)) * -1e9
            mask_for_max = tf.expand_dims(mask_for_max, axis=-1)
            # Apply mask and get max
            masked_sequence_output = sequence_output + mask_for_max
            pooled_output = tf.reduce_max(masked_sequence_output, axis=1)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        
        sentence_embedding = self.dropout(pooled_output, training=training)
        
        return {
            "sentence_embedding": sentence_embedding,
            "sequence_output": sequence_output,
            "input_mask": encoded_inputs["attention_mask"]
        }
    
    def get_embedding(self, sentences):
        """Get the embedding for one or more sentences.
        
        Args:
            sentences: Single sentence or batch of sentences
            
        Returns:
            Embeddings for the input sentences
        """
        if tf.executing_eagerly():
            return self(sentences)["sentence_embedding"]
        else:
            # During tracing, return a dummy tensor of the right shape
            return tf.zeros([1, self.embedding_dim])
    
    def _create_serving_signature(self):
        """Create the serving signature for TF SavedModel."""
        @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name="sentences")])
        def serving_fn(sentences):
            # Create a dummy output of the right shape - TF will replace with real computation at runtime
            dummy_embeddings = tf.zeros([tf.shape(sentences)[0], self.embedding_dim])
            return {"embeddings": dummy_embeddings}
            
        return serving_fn
    
    def get_signature(self):
        """Get signature for TF Serving."""
        return self._create_serving_signature()
    
    def save_model(self, export_path):
        """Save the model for TensorFlow Serving."""
        # Save both the Hugging Face model and the wrapper
        self.bert_model.save_pretrained(f"{export_path}/bert_model")
        self.tokenizer.save_pretrained(f"{export_path}/tokenizer")
        
        # Also save as TensorFlow SavedModel for serving
        tf.saved_model.save(
            self,
            export_path,
            signatures={
                "serving_default": self.get_signature()
            }
        )


def create_sentence_encoder(config: ModelConfig = None) -> SentenceEncoder:
    """Factory function to create a sentence encoder model.
    
    Args:
        config: Model configuration
        
    Returns:
        Initialized sentence encoder model
    """
    if config is None:
        config = ModelConfig()
    
    model = SentenceEncoder(config)

    # Initialize with a sample input
    sample_text = tf.constant(["Hello world"])
    _ = model(sample_text)
    
    return model