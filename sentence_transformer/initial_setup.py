import os
import sys
import time
import shutil
import tensorflow as tf
import logging
from transformers import BertTokenizer, TFBertModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('setup.log')
    ]
)
logger = logging.getLogger(__name__)

def download_model(model_name):
    """Download and cache a Hugging Face Transformers model.
    
    Args:
        model_name: Name of the model on Hugging Face Hub
    """
    logger.info(f"Downloading {model_name} from Hugging Face")
    try:
        start_time = time.time()

        tokenizer = BertTokenizer.from_pretrained(model_name)
        logger.info(f"Successfully downloaded {model_name} tokenizer")

        model = TFBertModel.from_pretrained(model_name)
        elapsed_time = time.time() - start_time
        logger.info(f"Successfully downloaded {model_name} model in {elapsed_time:.2f} seconds")

        save_dir = f"model_weights/{model_name}"
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        logger.info(f"Saved model and tokenizer to {save_dir}")
        
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error downloading {model_name}: {str(e)}")
        raise


def test_models(model, tokenizer):
    """Test the downloaded models with a sample input.
    
    Args:
        model: Downloaded TFBertModel
        tokenizer: Downloaded BertTokenizer
    """
    logger.info("Testing models with sample input")

    sample_text = ["Hello, this is a test."]
    
    try:
        logger.info("Processing sample input")

        encoded_input = tokenizer(
            sample_text, 
            padding='max_length', 
            truncation=True, 
            max_length=128, 
            return_tensors="tf"
        )

        outputs = model(
            encoded_input["input_ids"],
            attention_mask=encoded_input["attention_mask"],
            token_type_ids=encoded_input["token_type_ids"]
        )
        
        last_hidden_state_shape = outputs.last_hidden_state.shape
        pooler_output_shape = outputs.pooler_output.shape
        
        logger.info(f"Model test successful!")
        logger.info(f"Last hidden state shape: {last_hidden_state_shape}")
        logger.info(f"Pooler output shape: {pooler_output_shape}")
        
        return True
    except Exception as e:
        logger.error(f"Model test failed: {str(e)}")
        return False


def create_sample_data():
    """Create sample data for demonstration."""
    logger.info("Creating sample data")
    
    # Create the required directory structure based on DataConfig defaults
    os.makedirs("training_data/train", exist_ok=True)
    os.makedirs("training_data/eval", exist_ok=True)
    os.makedirs("training_data/test", exist_ok=True)
    
    sentences = [
        "I'll make him an offer he can't refuse.",
        "Keep your friends close, but your enemies closer.",
        "Luca Brasi sleeps with the fishes.",
        "A man who doesn't spend time with his family can never be a real man.",
        "Just when I thought I was out, they pull me back in.",
        "You broke my heart, Fredo. You broke my heart.",
        "In Sicily, women are more dangerous than shotguns." 
    ]

    classification_labels = [2, 1, 2, 1, 2, 2, 1]
    
    ner_tags = [
        ["O", "O", "O", "O", "O", "O", "O", "O", "O"],
        ["O", "O", "O", "O", "O", "O", "O", "O", "O"],
        ["B-PER", "I-PER", "O", "O", "O", "O", "O"],
        ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"],
        ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O"],
        ["O", "O", "O", "O", "B-PER", "O", "O", "O", "O", "O"],
        ["O", "B-LOC", "O", "O", "O", "O", "O", "O", "O"]
    ]
    
    # Create data for multitask learning in each directory
    for directory in ["train", "eval", "test"]:
        multitask_data_path = f"training_data/{directory}/multitask_data.csv"
        with open(multitask_data_path, "w") as f:
            f.write("text,class_label,ner_labels\n")
            
            for i, sentence in enumerate(sentences):
                text = sentence.replace(",", " ")
                class_label = ["positive", "neutral", "negative"][classification_labels[i]]
                ner = str(ner_tags[i]).replace(",", ";")
                f.write(f'"{text}","{class_label}","{ner}"\n')
        logger.info(f"Created multitask data at {multitask_data_path}")
        
        # Create triplet data for sentence embedding
        triplet_data_path = f"training_data/{directory}/triplets.csv"
        with open(triplet_data_path, "w") as f:
            f.write("anchor,positive,negative\n")
            
            # Create some triplets from the sentences
            f.write(f'"{sentences[0]}","{sentences[4]}","{sentences[1]}"\n')
            f.write(f'"{sentences[2]}","{sentences[5]}","{sentences[3]}"\n')
            f.write(f'"{sentences[1]}","{sentences[6]}","{sentences[0]}"\n')
        logger.info(f"Created triplet data at {triplet_data_path}")
        
        # Create a general data.csv file for compatibility with previous code
        data_path = f"training_data/{directory}/data.csv"
        with open(data_path, "w") as f:
            f.write("text,class_label,ner_labels\n")
            
            for i, sentence in enumerate(sentences):
                text = sentence.replace(",", " ")
                class_label = ["positive", "neutral", "negative"][classification_labels[i]]
                ner = str(ner_tags[i]).replace(",", ";")
                f.write(f'"{text}","{class_label}","{ner}"\n')
        logger.info(f"Created general data file at {data_path}")
    
    # Create compatibility symlinks if needed (but not on Windows where symlinks are problematic)
    try:
        os.makedirs("data", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        
        # Create symlinks only if they don't exist
        for directory in ["train", "eval", "test"]:
            src_dir = f"training_data/{directory}"
            dst_dir = f"data/{directory}"
            
            if not os.path.exists(dst_dir) and not os.path.islink(dst_dir):
                try:
                    # Try creating a symlink first
                    os.symlink(f"../training_data/{directory}", dst_dir)
                    logger.info(f"Created symlink from {dst_dir} to {src_dir}")
                except (OSError, AttributeError):
                    # Fallback to copying files on Windows
                    os.makedirs(dst_dir, exist_ok=True)
                    for filename in os.listdir(src_dir):
                        src_file = os.path.join(src_dir, filename)
                        dst_file = os.path.join(dst_dir, filename)
                        if os.path.isfile(src_file):
                            shutil.copy2(src_file, dst_file)
                    logger.info(f"Copied files from {src_dir} to {dst_dir}")
        
        # Create model symlinks
        if os.path.exists("model_weights") and not os.path.exists("models/bert-base-uncased") and not os.path.islink("models/bert-base-uncased"):
            try:
                os.symlink("../model_weights/bert-base-uncased", "models/bert-base-uncased")
                logger.info("Created symlink from models/bert-base-uncased to model_weights/bert-base-uncased")
            except (OSError, AttributeError):
                if os.path.exists("model_weights/bert-base-uncased"):
                    os.makedirs("models/bert-base-uncased", exist_ok=True)
                    for item in os.listdir("model_weights/bert-base-uncased"):
                        src = os.path.join("model_weights/bert-base-uncased", item)
                        dst = os.path.join("models/bert-base-uncased", item)
                        if os.path.isfile(src):
                            shutil.copy2(src, dst)
                    logger.info("Copied files from model_weights/bert-base-uncased to models/bert-base-uncased")
    except Exception as e:
        logger.warning(f"Error creating compatibility links/copies: {str(e)}")


def check_tf_version():
    """Check if TensorFlow version is compatible."""
    version = tf.__version__
    parts = version.split('.')
    if len(parts) >= 2:
        major, minor = parts[0], parts[1]
        if major == '2' and int(minor) == 10:
            logger.info(f"Using TensorFlow version {version} - Compatible ✓")
            return True
        else:
            logger.warning(f"Using TensorFlow version {version} - Expected 2.10.x")
            logger.warning("This script is optimized for TensorFlow 2.10")
            return False
    else:
        logger.warning(f"Unable to parse TensorFlow version: {version}")
        return False


def main():
    """Main setup function."""
    logger.info("Starting initial setup")

    # First check TensorFlow version
    check_tf_version()
    
    try:
        # Create directories
        os.makedirs("model_weights", exist_ok=True)
        os.makedirs("training_data", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)
        
        # Download base model
        model_name = "bert-base-uncased"
        model, tokenizer = download_model(model_name)
        
        # Test the model
        test_models(model, tokenizer)
        
        # Create sample data
        create_sample_data()
        
        logger.info("Initial setup completed successfully")
        logger.info(f"Models saved in './model_weights/{model_name}'")
        logger.info("Sample data created in './training_data/' with train, eval, and test subfolders")
        logger.info("Created checkpoint directory at './checkpoints/'")
        logger.info("Created compatibility links in './data/' and './models/'")
        logger.info("")
        logger.info("You can now run the example notebook or scripts")
        
    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()