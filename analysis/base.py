from pathlib import Path


def get_experiment_path(experiment):
    data_path = Path.home() / 'data/delib-results/results'
    return data_path / experiment


def get_results_path(experiment):
    return get_experiment_path(experiment) / 'results'
