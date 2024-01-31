import pandas as pd
from characteristics.io.paths import *


def inline_items_to_inrow_items(data_path):
    rows, cols = [], []

    with open(data_path, 'r') as file:
        print(f'reading file at \'{data_path}\' ')
        for line in file:
            els = line.split(' ')
            if not els[1] == '\n':
                user = int(els[0])
                items = [int(el) for el in els[1:]]
                rows.extend([user] * len(items))
                cols.extend(items)
    return pd.concat([pd.Series(rows), pd.Series(cols)], axis=1)


def load_csv(data_path):
    print(f'reading file at \'{data_path}\' ')
    return pd.read_csv(data_path)


def amazon_book_preprocessing():
    """
    This function loads the raw data related to the Amazon Book dataset and transform them to be processed by Elliot
    Returns: None
    """
    train_name = 'train.txt'
    test_name = 'test.txt'

    train_path = os.path.join(dataset_raw_folder(AMAZON), train_name)
    test_path = os.path.join(dataset_raw_folder(AMAZON), test_name)

    train = inline_items_to_inrow_items(train_path)
    test = inline_items_to_inrow_items(test_path)

    dataset_path = dataset_filepath(AMAZON)

    df = pd.concat([train, test], axis=0).sort_values([0, 1]).reset_index(drop=True)
    df.to_csv(dataset_path, sep='\t', header=False, index=False)
    print(f'Amazon Book dataset stored at \'{dataset_path}\'')


def gowalla_preprocessing():
    """
    This function loads the raw data related to the Gowalla dataset and transform them to be processed by Elliot
    Returns: None
    """
    train_name = 'train.txt'
    test_name = 'test.txt'

    train_path = os.path.join(dataset_raw_folder(GOWALLA), train_name)
    test_path = os.path.join(dataset_raw_folder(GOWALLA), test_name)

    train = inline_items_to_inrow_items(train_path)
    test = inline_items_to_inrow_items(test_path)

    dataset_path = dataset_filepath(GOWALLA)

    df = pd.concat([train, test], axis=0).sort_values([0, 1]).reset_index(drop=True)
    df.to_csv(dataset_path, sep='\t', header=False, index=False)
    print(f'Gowalla dataset stored at \'{dataset_path}\'')


def yelp_preprocessing():
    """
    This function loads the raw data related to the Yelp dataset and transform them to be processed by Elliot
    Returns: None
    """
    train_name = 'train.csv'
    test_name = 'test.csv'

    train_path = os.path.join(dataset_raw_folder(YELP), train_name)
    test_path = os.path.join(dataset_raw_folder(YELP), test_name)

    train = load_csv(train_path)
    test = load_csv(test_path)

    dataset_path = dataset_filepath(YELP)

    df = pd.concat([train, test], axis=0).sort_values(['user', 'item']).reset_index(drop=True)
    df.to_csv(dataset_path, sep='\t', header=False, index=False)
    print(f'Yelp2018 dataset stored at \'{dataset_path}\'')

