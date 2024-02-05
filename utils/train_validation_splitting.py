import numpy as np

from sklearn.model_selection import train_test_split

def split_data(raw_df, class_label, ratio, random_seed, verbose = False):
    np.random.seed(random_seed)

    if verbose:
        print("Total number of samples: ", len(raw_df))
        print("Total number of samples from the negative class: {} ({:.2f}%)".format(len(raw_df[raw_df[class_label] == 0]), 100 * len(raw_df[raw_df[class_label] == 0]) / len(raw_df)))
        print("Total number of samples form the positive class: {} ({:.2f}%)".format(len(raw_df[raw_df[class_label] == 1]), 100 * len(raw_df[raw_df[class_label] == 1]) / len(raw_df)))

    # Split into train and test (stratified)
    train_df, test_df = train_test_split(raw_df, test_size=ratio, random_state=random_seed, stratify=raw_df[class_label])

    if verbose:
        print("Train:")
        print("Total number of samples: ", len(train_df))
        print("Total number of samples from negative class: {} ({:.2f}%)".format(len(train_df[train_df[class_label] == 0]), 100 * len(train_df[train_df[class_label] == 0]) / len(train_df)))
        print("Total number of samples from positive class: {} ({:.2f}%)".format(len(train_df[train_df[class_label] == 1]), 100 * len(train_df[train_df[class_label] == 1]) / len(train_df)))

        print("Test:")
        print("Total number of samples: ", len(test_df))
        print("Total number of samples from negative class: {} ({:.2f}%)".format(len(test_df[test_df[class_label] == 0]), 100 * len(test_df[test_df[class_label] == 0]) / len(test_df)))
        print("Total number of samples from positive class: {} ({:.2f}%)".format(len(test_df[test_df[class_label] == 1]), 100 * len(test_df[test_df[class_label] == 1]) / len(test_df)))

    return train_df, test_df