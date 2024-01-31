import os
from characteristics import PROJECT_PATH

DATA_DIR = os.path.join(PROJECT_PATH, 'data')
RESULT_DIR = os.path.join(PROJECT_PATH, 'results')
CONFIG_DIR = os.path.join(PROJECT_PATH, 'config_files')
RAW_DATA_FOLDER = 'raw'
DATASET_NAME = 'dataset.tsv'
AMAZON = 'amazon-book'
GOWALLA = 'gowalla'
YELP = 'yelp2018'


def dataset_directory(dataset_name: str) -> str:
    """
    Given the dataset name returns the dataset directory
    @param dataset_name: name of the dataset
    @return: the path of the directory containing the dataset data
    """
    dataset_dir = os.path.join(DATA_DIR, dataset_name)
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f'Directory at {dataset_dir} not found. Please, check that dataset directory exists')
    return os.path.abspath(dataset_dir)


def dataset_raw_folder(dataset_name: str) -> str:
    """
    Given the dataset name returns the directory containing the raw data of the dataset
    @param dataset_name: name of the dataset
    @return: the path of the directory containing the dataset data
    """
    return os.path.join(dataset_directory(dataset_name), RAW_DATA_FOLDER)


def dataset_filepath(dataset_name: str) -> str:
    """
    Given the dataset name returns the path of the dataset data
    @param dataset_name: name of the dataset
    @return: the path of the dataset data
    """
    return os.path.join(dataset_directory(dataset_name), DATASET_NAME)