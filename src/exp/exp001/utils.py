import random
import os
import numpy as np
import torch
import mlflow
import yaml


def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results

    Arguments:
        seed {int} -- Number of the seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# MlFlowClient のRun ID を引き回すためのラッパー
class MlflowWriter():
    def __init__(self, experiment_name, **kwargs):
        self.client = mlflow.tracking.MlflowClient(**kwargs)
        try:
            self.experiment_id = self.client.create_experiment(experiment_name)
        except:
            self.experiment_id = self.client.get_experiment_by_name(experiment_name).experiment_id
        self.run_id = None

    def set_artifact_location_to_gs(self, bucket, mlruns_filepath):
        meta_filepath = f'{mlruns_filepath}/mlruns/{self.experiment_id}/meta.yaml'
        with open(meta_filepath) as file:
            meta = yaml.safe_load(file)
        meta['artifact_location'] = 'gs://{}/artifacts'.format(bucket)
        with open(meta_filepath, 'w') as file:
            yaml.dump(meta, file, default_flow_style=False)

    def create_run_id(self, tags=None):
        self.run_id = self.client.create_run(self.experiment_id, tags=tags).info.run_id

    def log_params_from_config(self, config, target=None):
        for key, value in config.items():
            if target is not None:
                key = '{}.{}'.format(target, key)
            self._explore_recursive(key, value)

    def _explore_recursive(self, parent, element):
        if isinstance(element, dict):
            for key, value in element.items():
                self._explore_recursive(f'{parent}.{key}', value)
        elif isinstance(element, list):
            for idx, value in enumerate(element):
                self._explore_recursive(f'{parent}.{idx}', value)
        else:
            self.client.log_param(self.run_id, parent, element)

    def log_param(self, key, value):
        self.client.log_param(self.run_id, key, value)

    def log_metric(self, key, value, step=None):
        if step:
            self.client.log_metric(self.run_id, key, value, step=step)
        else:
            self.client.log_metric(self.run_id, key, value)

    def log_artifact(self, local_path):
        self.client.log_artifact(self.run_id, local_path=local_path)

    def set_terminated(self):
        self.client.set_terminated(self.run_id)

if __name__ == "__main__":
    FILE_DIR = os.path.dirname(os.path.abspath(__file__))
    SRC_DIR = os.path.dirname(os.path.dirname(FILE_DIR))
    writer = MlflowWriter("test_experiment")
    writer.set_artifact_location_to_gs('bucket', SRC_DIR)
    writer.create_run_id()
    config = {'A': {'a': 1}, 'B': '2'}
    writer.log_params_from_config(config)
    writer.log_metric('A', 3, step=0)
    writer.log_metric('A', 4, step=0)
    writer.set_terminated()