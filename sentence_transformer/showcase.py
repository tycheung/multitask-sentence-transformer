import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformer.config import Config, ModelConfig, MultitaskConfig
from sentence_transformer.models.sentence_encoder import create_sentence_encoder
from sentence_transformer.models.multitask import create_multitask_model, TaskLoss


def showcase_sentence_transformer(sample_sentences=None):
    """Showcase the basic sentence transformer functionality.
    
    Args:
        sample_sentences: List of sample sentences to encode
        
    Returns:
        Dictionary with model, embeddings, and visualization data
    """
    if sample_sentences is None:
        sample_sentences = [
            "I'll make him an offer he can't refuse.",
            "Keep your friends close, but your enemies closer.",
            "Luca Brasi sleeps with the fishes.",
            "A man who doesn't spend time with his family can never be a real man.",
            "Just when I thought I was out, they pull me back in.",
            "You broke my heart, Fredo. You broke my heart.",
            "In Sicily, women are more dangerous than shotguns."
        ]
    
    # Create configuration with explicit Hugging Face model name
    config = Config()
    config.model = ModelConfig(
        bert_model_name="bert-base-uncased",
        embedding_dim=768,
        pooling_strategy="cls",
        dropout_rate=0.1
    )

    # Create sentence encoder model
    model = create_sentence_encoder(config.model)

    # Convert sentences to tensor for processing
    sentences_tensor = tf.constant(sample_sentences)

    # Get model outputs
    try:
        outputs = model(sentences_tensor)
        embeddings = outputs["sentence_embedding"].numpy()
    except Exception as e:
        print(f"Error during model inference: {str(e)}")
        # Fallback to processing one sentence at a time if batch processing fails
        embeddings = []
        for sentence in sample_sentences:
            sentence_tensor = tf.constant([sentence])
            output = model(sentence_tensor)
            embeddings.append(output["sentence_embedding"].numpy()[0])
        embeddings = np.array(embeddings)

    # Compute similarity matrix
    sim_matrix = cosine_similarity(embeddings)

    tsne_data = None
    if len(sample_sentences) >= 4:
        # Use a lower perplexity for small datasets
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(len(sample_sentences)-1, 5))
        embeddings_2d = tsne.fit_transform(embeddings)
        tsne_data = {
            "embeddings_2d": embeddings_2d,
            "sentences": sample_sentences
        }
    
    return {
        "model": model,
        "model_config": config.model,
        "sentences": sample_sentences,
        "embeddings": embeddings,
        "similarity_matrix": sim_matrix,
        "tsne_data": tsne_data
    }


def showcase_multitask_model(sample_sentences=None):
    """Showcase the multi-task model functionality.
    
    Args:
        sample_sentences: List of sample sentences to process
        
    Returns:
        Dictionary with model outputs and visualization data
    """
    if sample_sentences is None:
        sample_sentences = [
            "I'll make him an offer he can't refuse.",
            "Keep your friends close, but your enemies closer.",
            "Luca Brasi sleeps with the fishes.",
            "A man who doesn't spend time with his family can never be a real man.",
            "Just when I thought I was out, they pull me back in.",
            "You broke my heart, Fredo. You broke my heart.",
            "In Sicily, women are more dangerous than shotguns."
        ]
    
    # Create configuration with Hugging Face parameters
    config = Config()
    config.model = MultitaskConfig(
        bert_model_name="bert-base-uncased",
        embedding_dim=768,
        pooling_strategy="cls",
        dropout_rate=0.1,
        classification_classes=3,
        ner_classes=9,
        use_contrastive_learning=True
    )

    # Create multitask model
    model = create_multitask_model(config.model)
    
    # Convert sentences to tensor
    sentences_tensor = tf.constant(sample_sentences)

    # Get model outputs with error handling
    try:
        outputs = model(sentences_tensor)
        embeddings = outputs["sentence_embedding"].numpy()
        classification_logits = outputs["classification_logits"].numpy()
        ner_logits = outputs["ner_logits"].numpy()
    except Exception as e:
        print(f"Error during model inference: {str(e)}")
        # Fallback to processing one sentence at a time
        embeddings = []
        classification_logits = []
        ner_logits_list = []
        
        for sentence in sample_sentences:
            sentence_tensor = tf.constant([sentence])
            output = model(sentence_tensor)
            embeddings.append(output["sentence_embedding"].numpy()[0])
            classification_logits.append(output["classification_logits"].numpy()[0])
            ner_logits_list.append(output["ner_logits"].numpy()[0])
        
        embeddings = np.array(embeddings)
        classification_logits = np.array(classification_logits)
        ner_logits = np.array(ner_logits_list)

    # Get predicted classes
    predicted_classes = np.argmax(classification_logits, axis=-1)
    
    # Define expected classes for our sample data
    expected_classes = [2, 1, 2, 1, 2, 2, 1]

    class_names = ["Positive", "Neutral", "Negative"]
    predicted_class_names = [class_names[i] for i in predicted_classes]
    expected_class_names = [class_names[i] for i in expected_classes]
    
    # Prepare classification results
    classification_results = []
    for i, sentence in enumerate(sample_sentences):
        class_probs = classification_logits[i]
        top_class_idx = predicted_classes[i]
        confidence = float(class_probs[top_class_idx]) * 100  # Convert to Python float for JSON serialization
        
        classification_results.append({
            "sentence": sentence,
            "predicted_class": predicted_class_names[i],
            "expected_class": expected_class_names[i],
            "confidence": confidence
        })

    # Define NER tags and expected entities
    ner_tag_names = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
    
    expected_entities = [
        [],  # No named entities
        [],  # No named entities
        [("Luca Brasi", "PER")],  # Person
        [],  # No named entities
        [],  # No named entities
        [("Fredo", "PER")],  # Person
        [("Sicily", "LOC")]  # Location
    ]

    # Process NER results
    ner_results = []
    for i, sentence in enumerate(sample_sentences):
        # Split into tokens
        tokens = sentence.split()

        # Get predicted tags
        tag_indices = np.argmax(ner_logits[i], axis=-1)
        
        ner_tags = []
        for j in range(min(len(tokens), len(tag_indices))):
            tag_idx = tag_indices[j]
            if tag_idx < len(ner_tag_names):  # Ensure index is in range
                tag_name = ner_tag_names[tag_idx]
                ner_tags.append(tag_name)
            else:
                ner_tags.append("O")  # Default to "O" tag if out of range

        expected_ents = expected_entities[i]
        
        ner_results.append({
            "sentence": sentence,
            "tokens": tokens[:len(ner_tags)],
            "ner_tags": ner_tags,
            "expected_entities": expected_ents
        })

    # Compute similarity matrix
    sim_matrix = cosine_similarity(embeddings)

    # t-SNE visualization
    try:
        perplexity = min(len(sample_sentences)-1, 3)
        if perplexity < 2:
            perplexity = 2  # Minimum perplexity for t-SNE
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings_2d = tsne.fit_transform(embeddings)
    except Exception as e:
        print(f"Error during t-SNE: {str(e)}")
        # Fallback to random embedding if t-SNE fails
        embeddings_2d = np.random.randn(len(sample_sentences), 2)
    
    return {
        "model": model,
        "model_config": config.model,
        "sentences": sample_sentences,
        "embeddings": embeddings,
        "classification_results": classification_results,
        "ner_results": ner_results,
        "similarity_matrix": sim_matrix,
        "tsne_data": {
            "embeddings_2d": embeddings_2d,
            "sentences": sample_sentences
        }
    }


def showcase_training_approaches(freeze_options=None):
    """Showcase different training approaches with freezing options.
    
    Args:
        freeze_options: Dict with different freezing configurations to test
        
    Returns:
        Dictionary with models and configurations
    """
    if freeze_options is None:
        freeze_options = {
            "freeze_entire_network": True,
            "freeze_transformer_only": True,
            "freeze_task_a_head_only": True
        }

    # Create config with Hugging Face model name
    config = Config()
    config.model = MultitaskConfig(
        bert_model_name="bert-base-uncased",
        trainable_bert=not freeze_options.get("freeze_transformer_only", False),
        embedding_dim=768
    )

    # Create model
    model = create_multitask_model(config.model)
    
    # Implement freezing options
    if freeze_options.get("freeze_entire_network", False):
        model.trainable = False
    
    if freeze_options.get("freeze_task_a_head_only", False) and not freeze_options.get("freeze_entire_network", False):
        model.trainable = True

        for layer in model.classification_layers:
            layer.trainable = False
        model.classification_output.trainable = False

    # Count parameters
    trainable_params = np.sum([np.prod(v.get_shape()) for v in model.trainable_variables])
    non_trainable_params = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_variables])
    
    return {
        "model": model,
        "config": config,
        "freeze_options": freeze_options,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "total_params": trainable_params + non_trainable_params,
        "trainable_percentage": trainable_params / (trainable_params + non_trainable_params) * 100
    }


def showcase_transfer_learning():
    """Showcase transfer learning approaches.
    
    Returns:
        Dictionary with model configurations and training strategies
    """
    configs = []

    # Configuration 1: Freeze all BERT layers
    config1 = Config()
    config1.model = MultitaskConfig(
        bert_model_name="bert-base-uncased",
        trainable_bert=False,
        embedding_dim=768
    )
    configs.append(("Freeze all BERT layers", config1))

    # Configuration 2: Gradual unfreezing
    config2 = Config()
    config2.model = MultitaskConfig(
        bert_model_name="bert-base-uncased",
        trainable_bert=True,
        embedding_dim=768
    )
    config2.training.use_gradual_unfreezing = True
    configs.append(("Gradual unfreezing of BERT layers", config2))

    # Configuration 3: Fine-tune everything
    config3 = Config()
    config3.model = MultitaskConfig(
        bert_model_name="bert-base-uncased",
        trainable_bert=True,
        embedding_dim=768
    )
    configs.append(("Fine-tune the entire model", config3))
    
    # Define training strategies
    strategies = [
        {
            "name": "Freeze all BERT layers",
            "pre_trained_model": "BERT base uncased (12-layer, 768-hidden, 12-heads)",
            "frozen_layers": "All BERT layers",
            "trainable_layers": "Only task-specific heads for classification and NER",
            "rationale": "This approach is useful when you have limited training data or want to avoid catastrophic forgetting of pre-trained knowledge. It's computationally efficient but less adaptable to the target domain.",
            "config": config1
        },
        {
            "name": "Gradual unfreezing of BERT layers",
            "pre_trained_model": "BERT base uncased (12-layer, 768-hidden, 12-heads)",
            "frozen_layers": "Initially all BERT layers, gradually unfreezing from top to bottom",
            "trainable_layers": "Task-specific heads + progressively more BERT layers",
            "rationale": "This approach balances adaptation with preservation of pre-trained knowledge. By unfreezing layers gradually (starting from the top), the model can adapt to the target domain while maintaining lower-level features.",
            "config": config2
        },
        {
            "name": "Fine-tune the entire model",
            "pre_trained_model": "BERT base uncased (12-layer, 768-hidden, 12-heads)", 
            "frozen_layers": "None",
            "trainable_layers": "All layers (BERT + task-specific heads)",
            "rationale": "This approach allows maximum adaptation to the target domain but requires more training data and computational resources. It's best when the target domain differs significantly from BERT's pre-training data.",
            "config": config3
        }
    ]
    
    return {
        "strategies": strategies,
        "configs": configs
    }


def showcase_training_loop(create_synthetic_data=True):
    """Showcase the training loop implementation for multi-task learning.
    
    Args:
        create_synthetic_data: Whether to create synthetic data for demonstration
        
    Returns:
        Dictionary with model, synthetic data, and training specifications
    """
    if create_synthetic_data:
        # Sample text data
        texts = [
            "I'll make him an offer he can't refuse.",
            "Keep your friends close, but your enemies closer.",
            "Luca Brasi sleeps with the fishes.",
            "A man who doesn't spend time with his family can never be a real man.",
            "Just when I thought I was out, they pull me back in.",
            "You broke my heart, Fredo. You broke my heart.",
            "In Sicily, women are more dangerous than shotguns."
        ]
        
        # Classification labels: 0=positive, 1=neutral, 2=negative
        classification_labels = [2, 1, 2, 1, 2, 2, 1]

        # NER labels
        ner_labels = []
        for i, text in enumerate(texts):
            tokens = text.split()
            tags = ["O"] * len(tokens)
            
            if i == 2:  # "Luca Brasi sleeps with the fishes."
                tags[0] = "B-PER"  # Luca
                tags[1] = "I-PER"  # Brasi
            elif i == 5:  # "You broke my heart, Fredo. You broke my heart."
                tags[4] = "B-PER"  # Fredo
            elif i == 6:  # "In Sicily, women are more dangerous than shotguns."
                tags[1] = "B-LOC"  # Sicily
            
            ner_labels.append(tags)
        
        # Convert to tensors
        texts_tensor = tf.constant(texts)
        classification_labels_tensor = tf.constant(classification_labels)
        
        # Pad NER labels to the same length
        max_length = max(len(tags) for tags in ner_labels)
        ner_tag_ids = {"O": 0, "B-PER": 1, "I-PER": 2, "B-ORG": 3, "I-ORG": 4, 
                       "B-LOC": 5, "I-LOC": 6, "B-MISC": 7, "I-MISC": 8}
        
        padded_ner_labels = []
        for tags in ner_labels:
            tag_ids = [ner_tag_ids[tag] for tag in tags]
            padded = tag_ids + [0] * (max_length - len(tag_ids))
            padded_ner_labels.append(padded)
        
        ner_labels_tensor = tf.constant(padded_ner_labels)
        
        # Prepare inputs and targets
        inputs = texts_tensor
        targets = {
            "classification_labels": classification_labels_tensor,
            "ner_labels": ner_labels_tensor
        }
    else:
        inputs = None
        targets = None
    
    # Create configuration with Hugging Face model
    config = Config()
    config.model = MultitaskConfig(
        bert_model_name="bert-base-uncased",
        embedding_dim=768,
        classification_classes=3,
        ner_classes=9,
        task_weights={
            "classification": 1.0,
            "ner": 1.0,
            "sentence_embedding": 0.5
        }
    )
    
    # Create model
    model = create_multitask_model(config.model)
    
    # Create task loss calculator
    task_loss = TaskLoss(config.model)
    
    # Create optimizer, using legacy.Adam for TF 2.10 compatibility
    learning_rate = 2e-5
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    
    # Training pseudocode
    training_pseudocode = """
    # Training loop pseudocode for multi-task learning
    for epoch in range(num_epochs):
        # Reset metrics
        train_loss.reset_states()
        train_classification_loss.reset_states()
        train_ner_loss.reset_states()
        train_classification_accuracy.reset_states()
        train_ner_accuracy.reset_states()
        
        for batch in train_dataset:
            inputs, targets = batch
            
            with tf.GradientTape() as tape:
                outputs = model(inputs, training=True)
                
                losses = task_loss.compute_total_loss(outputs, targets)
                total_loss = losses["total_loss"]
            
            gradients = tape.gradient(total_loss, model.trainable_variables)
            
            gradients, _ = tf.clip_by_global_norm(gradients, max_grad_norm)
            
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            train_loss.update_state(total_loss)
            train_classification_loss.update_state(losses["classification_loss"])
            train_ner_loss.update_state(losses["ner_loss"])
            
            train_classification_accuracy.update_state(
                targets["classification_labels"], 
                outputs["classification_logits"]
            )
            
            mask = tf.cast(outputs["input_mask"], tf.bool)
            train_ner_accuracy.update_state(
                tf.boolean_mask(targets["ner_labels"], mask),
                tf.boolean_mask(outputs["ner_logits"], mask)
            )
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Loss: {train_loss.result():.4f}")
        print(f"  Classification Loss: {train_classification_loss.result():.4f}")
        print(f"  NER Loss: {train_ner_loss.result():.4f}")
        print(f"  Classification Accuracy: {train_classification_accuracy.result():.4f}")
        print(f"  NER Accuracy: {train_ner_accuracy.result():.4f}")
    """
    
    # Prepare training information
    training_info = {
        "model": model,
        "optimizer": "Adam (legacy version for TF 2.10)",
        "learning_rate": learning_rate,
        "loss_function": "Task-specific losses with weighted combination",
        "metrics": ["Classification Accuracy", "NER Accuracy"],
        "training_pseudocode": training_pseudocode
    }
    
    # Compute sample outputs with error handling
    if create_synthetic_data:
        try:
            with tf.GradientTape() as tape:
                outputs = model(inputs, training=True)
                
                losses = task_loss.compute_total_loss(outputs, targets)
                total_loss = losses["total_loss"]
            
            gradients = tape.gradient(total_loss, model.trainable_variables)
            
            training_info["sample_batch"] = {
                "inputs": texts,
                "classification_labels": classification_labels,
                "ner_labels": ner_labels
            }
            
            training_info["sample_outputs"] = {
                "total_loss": float(total_loss.numpy()),  # Convert to Python scalar for serialization
                "classification_loss": float(losses["classification_loss"].numpy()),
                "ner_loss": float(losses["ner_loss"].numpy()),
                "sentence_embedding_loss": float(losses["sentence_embedding_loss"].numpy())
            }
        except Exception as e:
            print(f"Error computing sample outputs: {str(e)}")
            training_info["sample_outputs"] = {
                "error": f"Could not compute sample outputs: {str(e)}"
            }
    
    return training_info


def plot_similarity_matrix(similarity_matrix, labels=None, title="Sentence Similarity Matrix"):
    """Plot a similarity matrix.
    
    Args:
        similarity_matrix: 2D array of similarity scores
        labels: Optional labels for the axes
        title: Title for the plot
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap="YlGnBu")
    plt.colorbar(label="Cosine Similarity")
    plt.title(title)
    
    if labels:
        short_labels = [f"S{i+1}: {l[:20]}..." if len(l) > 20 else f"S{i+1}: {l}" for i, l in enumerate(labels)]
        plt.xticks(range(len(labels)), short_labels, rotation=45, ha="right")
        plt.yticks(range(len(labels)), short_labels)
    
    for i in range(similarity_matrix.shape[0]):
        for j in range(similarity_matrix.shape[1]):
            plt.text(j, i, f"{similarity_matrix[i, j]:.2f}", 
                     ha="center", va="center", color="black")
    
    plt.tight_layout()
    return plt.gcf()


def plot_embeddings_tsne(tsne_data, title="t-SNE Visualization of Sentence Embeddings"):
    """Plot t-SNE visualization of embeddings.
    
    Args:
        tsne_data: Dictionary with embeddings_2d and sentences
        title: Title for the plot
    """
    embeddings_2d = tsne_data["embeddings_2d"]
    sentences = tsne_data["sentences"]
    
    plt.figure(figsize=(12, 10))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], marker="o", s=100)
    
    for i, sentence in enumerate(sentences):
        if len(sentence) > 30:
            short_sentence = sentence[:27] + "..."
        else:
            short_sentence = sentence
        
        plt.annotate(
            f"S{i+1}: {short_sentence}",
            (embeddings_2d[i, 0], embeddings_2d[i, 1]),
            fontsize=9, 
            ha="center",
            va="bottom",
            xytext=(0, 5),
            textcoords="offset points"
        )
    
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    return plt.gcf()


def highlight_entities(text, entities):
    """Create HTML with highlighted entities for display.
    
    Args:
        text: The input text
        entities: List of (start, end, label) tuples
        
    Returns:
        HTML string with highlighted entities
    """
    import html

    colors = {
        "PER": "#ff9999",  # Light red for Person
        "ORG": "#99ccff",  # Light blue for Organization
        "LOC": "#99ff99",  # Light green for Location
        "MISC": "#ffcc99"  # Light orange for Miscellaneous
    }

    escaped_text = html.escape(text)

    fragments = []
    last_end = 0

    for start, end, entity_tag in sorted(entities, key=lambda x: x[0]):
        if start < 0 or end <= start or start >= len(text) or end > len(text):
            continue

        if "-" in entity_tag:
            _, entity_type = entity_tag.split("-", 1)
        else:
            entity_type = entity_tag
            
        color = colors.get(entity_type, "#cccccc")

        if start > last_end:
            fragments.append(escaped_text[last_end:start])

        fragments.append(
            f'<span style="background-color: {color}; padding: 2px; border-radius: 3px; font-weight: bold;">'
            f'{escaped_text[start:end]}'
            f'<sup style="font-size: 0.7em; margin-left: 2px;">{entity_type}</sup></span>'
        )
        
        last_end = end

    if last_end < len(escaped_text):
        fragments.append(escaped_text[last_end:])

    return "".join(fragments)


def find_token_positions(sentence, tokens):
    """Find the positions of tokens in the original sentence.
    
    Args:
        sentence: The original sentence
        tokens: List of tokens to find
        
    Returns:
        List of (start_idx, end_idx) tuples for each token
    """
    positions = []
    pos = 0
    
    for token in tokens:
        idx = sentence.find(token, pos)
        if idx >= 0:
            positions.append((idx, idx + len(token)))
            pos = idx + len(token)
        else:
            lower_sentence = sentence.lower()
            lower_token = token.lower()
            idx = lower_sentence.find(lower_token, pos)
            if idx >= 0:
                positions.append((idx, idx + len(token)))
                pos = idx + len(token)
            else:
                positions.append((-1, -1))
    
    return positions


def display_ner_results(ner_results):
    """Display named entity recognition results with proper highlighting.
    
    Args:
        ner_results: List of dictionaries containing NER results
    """
    from IPython.display import HTML, display
    
    for result in ner_results:
        sentence = result["sentence"]
        tokens = result["tokens"]
        ner_tags = result["ner_tags"]
        expected_entities = result["expected_entities"]
        
        print(f"Sentence: {sentence}")
        print("Named Entities:")

        pos = 0
        token_positions = []
        
        for token in tokens:
            token_pos = sentence.find(token, pos)
            if token_pos >= 0:
                token_positions.append((token_pos, token_pos + len(token)))
                pos = token_pos + len(token)
            else:
                token_positions.append((-1, -1))
        
        entities = []
        i = 0
        while i < len(tokens):
            if i < len(ner_tags) and ner_tags[i].startswith("B-"):
                start_pos, _ = token_positions[i]
                
                if start_pos >= 0:
                    entity_end = i
                    for j in range(i + 1, min(len(tokens), len(ner_tags))):
                        if ner_tags[j].startswith("I-") and ner_tags[j][2:] == ner_tags[i][2:]:
                            entity_end = j
                        else:
                            break
                    
                    _, end_pos = token_positions[entity_end]
                    
                    if end_pos >= 0:
                        entities.append((start_pos, end_pos, ner_tags[i]))
                    
                    i = entity_end + 1
                else:
                    i += 1
            else:
                i += 1
        
        if entities:
            display(HTML(highlight_entities(sentence, entities)))
        else:
            print("No entities detected.")
        
        if expected_entities:
            print("Expected entities:")
            for entity, entity_type in expected_entities:
                print(f"  - {entity} ({entity_type})")
        else:
            print("No expected entities.")
        
        print("\n" + "-"*60)