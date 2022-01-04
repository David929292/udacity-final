import argparse
import joblib
import numpy as np
import os
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from azureml.core import Dataset #, Datastore
# from azureml.core import Workspace, Experiment
from azureml.core.run import Run
# from azureml.data.datapath import DataPath
# from azureml.data.dataset_factory import TabularDatasetFactory


def main():
    # Get run context; AZ specific
    run = Run.get_context()
    
    # Parse data
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_url', type=str, default='http://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv', help='URL of dataset to be used')
    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength ie. stronger regularization the smaller the value")
    parser.add_argument('--max_iter', type=int, default=100, help="Max number of allowed iterations")

    args = parser.parse_args()

    # logging parameters
    run.log("Regularization Strength", np.float(args.C))
    run.log("Max iterations", np.int(args.max_iter))

    # import the data
    dataset = Dataset.Tabular.from_delimited_files(args.data_url)

    # turn into pd DF
    df = dataset.to_pandas_dataframe()

    # get regressand & regressors
    y = df.pop('DEATH_EVENT')
    x = df

    # Split in train/test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    
    # run a Logit for classification purposes using the args given by args-parse
    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    
    y_pred = model.predict(x_test)
    
    fbeta = fbeta_score(y_test, y_pred, average='macro', beta=0.5)
    
    # source: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
    auc = roc_auc_score(y_test, model.predict_proba(x_test)[:,1], average='weighted')
    
    run.log("Accuracy", np.float(accuracy))
    
    run.log("F(beta)", np.float(fbeta))

    run.log("AUC_weighted", np.float(auc))

    # Make a dir to save the model, files in "outputs" are auto-uploaded to the run's history
    os.makedirs('outputs', exist_ok=True)
    
    joblib.dump(model, 'outputs/model.joblib')

if __name__ == '__main__':
    main()
