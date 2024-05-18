import argparse
import warnings
from typing import Union
from logging import INFO
from datasets import Dataset, DatasetDict
import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import flwr as fl
from flwr_datasets import FederatedDataset
from flwr.common.logger import log
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Parameters,
    Status,
)
from flwr_datasets.partitioner import IidPartitioner


warnings.filterwarnings("ignore", category=UserWarning)

# Define arguments parser for the client/partition ID.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--partition-id",
    default=0,
    type=int,
    help="Partition ID used for the current client.",
)
parser.add_argument(
    "--dataset-path",
    type=str,
    required=True,
    help="Path to the dataset file.",
)
args = parser.parse_args()




# Define data partitioning related functions
def train_test_split(partition: Dataset, test_fraction: float, seed: int):
    """Split the data into train and validation set given split rate."""
    train_test = partition.train_test_split(test_size=test_fraction, seed=seed)
    partition_train = train_test["train"]
    partition_test = train_test["test"]

    num_train = len(partition_train)
    num_test = len(partition_test)

    return partition_train, partition_test, num_train, num_test


def transform_dataset_to_dmatrix(data: Union[Dataset, DatasetDict]) -> xgb.core.DMatrix:
    """Transform dataset to DMatrix format for xgboost."""
    x = data["inputs"]
    y = data["label"]
    new_data = xgb.DMatrix(x, label=y)
    return new_data


def load_and_preprocess_dataset(dataset_path):
    # Load the dataset
    data = pd.read_csv(dataset_path, sep=";|,", engine='python')
    
    # Preprocess the dataset
    data = data.drop(['id'], axis=1)
    
    # Define input features (X) and target variable (y)
    input_columns = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'active']
    X = data[input_columns].values
    y = data['cardio'].values

    # Create DMatrix for XGBoost
    train_dmatrix = xgb.DMatrix(X, label=y, enable_categorical=True)
    valid_dmatrix = xgb.DMatrix(X, label=y, enable_categorical=True)

    num_train = len(X)
    num_val = len(X)

    return train_dmatrix, valid_dmatrix, num_train, num_val

dataset_path = args.dataset_path
train_dmatrix, valid_dmatrix, num_train, num_val = load_and_preprocess_dataset(dataset_path)

# Hyper-parameters for xgboost training
num_local_round = 5
params = {
    "objective": "binary:logistic",
    "eta": 0.1,  # Learning rate
    "max_depth": 8,
    "eval_metric": "auc",
    "nthread": 2,
    "num_parallel_tree": 1,
    "subsample": 1,
    "tree_method": "hist",
    # "gpu_id":0,
    # "predictor": "gpu_predictor",
}


# Define Flower client
class XgbClient(fl.client.Client):
    def __init__(self):
        self.bst = None
        self.config = None

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        _ = (self, ins)
        return GetParametersRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[]),
        )

    def _local_boost(self):
        # Update trees based on local training data.
        for i in range(num_local_round):
            self.bst.update(train_dmatrix, self.bst.num_boosted_rounds())

        # Extract the last N=num_local_round trees for sever aggregation
        bst = self.bst[
            self.bst.num_boosted_rounds()
            - num_local_round : self.bst.num_boosted_rounds()
        ]

        return bst

    def fit(self, ins: FitIns) -> FitRes:
        if not self.bst:
            # First round local training
            log(INFO, "Start training at round 1")
            bst = xgb.train(
                params,
                train_dmatrix,
                num_boost_round=num_local_round,
                evals=[(valid_dmatrix, "validate"), (train_dmatrix, "train")],
            )
            self.config = bst.save_config()
            self.bst = bst
        else:
            for item in ins.parameters.tensors:
                global_model = bytearray(item)

            # Load global model into booster
            self.bst.load_model(global_model)
            self.bst.load_config(self.config)

            bst = self._local_boost()

        local_model = bst.save_raw("json")
        local_model_bytes = bytes(local_model)

        return FitRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            num_examples=num_train,
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        eval_results = self.bst.eval_set(
            evals=[(valid_dmatrix, "valid")],
            iteration=self.bst.num_boosted_rounds() - 1,
        )
        auc = round(float(eval_results.split("\t")[1].split(":")[1]), 4)

        return EvaluateRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            loss=0.0,
            num_examples=num_val,
            metrics={"AUC": auc},
        )


# Start Flower client
fl.client.start_client(server_address="127.0.0.1:8085", client=XgbClient().to_client())