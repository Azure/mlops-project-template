import os
import mlflow
import pandas as pd


def init():
    """
    Initialize the model for batch scoring.
    This function is called once when the batch deployment starts.
    """
    global model
    
    # The model path is provided by Azure ML
    model_path = os.environ.get("AZUREML_MODEL_DIR", "./model")
    model_path = os.path.join(model_path, "taxi-model")
    
    # Load the MLflow model
    model = mlflow.pyfunc.load_model(model_path)
    print(f"Model loaded successfully from {model_path}")


def run(mini_batch):
    """
    Process a mini-batch of data.
    
    Args:
        mini_batch: List of file paths to process
        
    Returns:
        List of predictions (one per input row)
    """
    results = []
    
    for file_path in mini_batch:
        # Read the CSV file
        data = pd.read_csv(file_path)
        
        # Make predictions
        predictions = model.predict(data)
        
        # Format results
        for pred in predictions:
            results.append(f"{pred:.2f}")
    
    return results
