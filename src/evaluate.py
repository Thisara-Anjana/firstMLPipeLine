import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import yaml
import os
import mlflow
from urllib.parse import urlparse

os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/thisaraanjana11/firstMLPipeLine.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "thisaraanjana11"  
os.environ["MLFLOW_TRACKING_PASSWORD"] = "014d43d3ca2087aab546a616df6603886395eb0c"

#load parameters from params.yaml
params = yaml.safe_load(open("params.yaml"))["train"]

def evaluate(model_path, data_path):
    data = pd.read_csv(data_path)
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]

    mlflow.set_tracking_uri("https://dagshub.com/thisaraanjana11/firstMLPipeLine.mlflow")

    ##loading the model from the specified path
    model = pickle.load(open(model_path, 'rb'))

    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    

    mlflow.log_metric("accuracy", accuracy)
    print(f"Model evaluation completed with accuracy: {accuracy}")

if __name__ == "__main__":
    evaluate(params["data_path"], params["model_path"])
