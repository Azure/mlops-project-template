import os
import mlflow
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init():
    """
    Initialize the model for batch scoring.
    This function is called once when the batch deployment starts.
    """
    global model
    
    # The model path is provided by Azure ML
    # For batch deployments, the model is in AZUREML_MODEL_DIR
    model_dir = os.environ.get("AZUREML_MODEL_DIR")
    logger.info(f"AZUREML_MODEL_DIR: {model_dir}")
    
    if model_dir:
        # List contents to debug
        logger.info(f"Contents of model directory: {os.listdir(model_dir)}")
        
        # The MLflow model should be directly in the model directory
        model_path = model_dir
    else:
        model_path = "./model"
    
    # Load the MLflow model
    try:
        model = mlflow.pyfunc.load_model(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {str(e)}")
        raise


def run(mini_batch):
    """
    Process a mini-batch of data.
    
    Args:
        mini_batch: List of file paths to process
        
    Returns:
        DataFrame with predictions (for append_row output action)
    """
    logger.info(f"Processing mini-batch with {len(mini_batch)} files")
    
    results = []
    
    for file_path in mini_batch:
        try:
            logger.info(f"Processing file: {file_path}")
            
            # Read the CSV file
            data = pd.read_csv(file_path)
            logger.info(f"Read {len(data)} rows from {file_path}")
            
            # Make predictions
            predictions = model.predict(data)
            logger.info(f"Generated {len(predictions)} predictions")
            
            # Append predictions as rows (for append_row output action)
            for pred in predictions:
                results.append([pred])
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            raise
    
    # Return as DataFrame for append_row output action
    return pd.DataFrame(results)
