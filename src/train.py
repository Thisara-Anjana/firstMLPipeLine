import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from mlflow.models import infer_signature
import os
from sklearn.model_selection import train_test_split,GridSearchCV
import yaml
from urllib.parse import urlparse
import mlflow


os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/thisaraanjana11/firstMLPipeLine.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "thisaraanjana11"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "014d43d3ca2087aab546a616df6603886395eb0c"

def hyperparameter_tuning(X_train,y_train,param_grid):
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search

##load parameters from params.yaml

params= yaml.safe_load(open("params.yaml"))["train"]

def train(data_path,model_path,random_state,n_estimators,max_depth):
    data = pd.read_csv(data_path)
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]

    mlflow.set_tracking_uri("https://dagshub.com/thisaraanjana11/firstMLPipeLine.mlflow")

    ## start mlflow run
    with mlflow.start_run():
        ## split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
        signature = infer_signature(X_train, y_train)

        ##define the hyperparameter grid for tuning

        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        }

        #Perform hyperparameter tuning

        grid_search = hyperparameter_tuning(X_train, y_train, param_grid)
        
        best_model = grid_search.best_estimator_

        ## Predict and evaluate the model
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Accuracy: {accuracy}")

        ## log additional metrics 
        mlflow.log_metric("accuracy", accuracy)

        mlflow.log_param("n_estimators", grid_search.best_params_["n_estimators"])
        mlflow.log_param("max_depth", grid_search.best_params_["max_depth"])
        mlflow.log_param("min_samples_split", grid_search.best_params_["min_samples_split"])
        mlflow.log_param("min_samples_leaf", grid_search.best_params_["min_samples_leaf"])

        ##log the confusion matrix and classification report as artifacts
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)

        mlflow.log_text(str(conf_matrix), "confusion_matrix.txt")
        mlflow.log_text(class_report, "classification_report.txt")

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(sk_model=best_model,name="model",registered_model_name="BestModel", signature=signature)
        else:
            mlflow.sklearn.save_model(best_model,"model", signature=signature)

        
        ##create the directory to save the model if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        filename = model_path
        pickle.dump(best_model, open(filename, "wb"))

        print(f"Model trained and saved successfully to {model_path}")

if __name__ == "__main__":
    train(params["data_path"], params["model_path"], params["random_state"], params["n_estimators"], params["max_depth"])
