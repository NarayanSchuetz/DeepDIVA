"""
This script allows for creation of a validation set from the training set.
"""

# Utils
import argparse
import os
import shutil
import sys
import numpy as np

# Torch related stuff
import torchvision.datasets as datasets
from sklearn.model_selection import train_test_split


def split_dataset_train_test_val(dataset_folder, split, symbolic, seed=42, val_split=0.1, shuffle=True,
                                 target_folder=None):
    """
    Splits a folder with structure dataset_folder/labels/files into train test and validation set, leaving the original
    folder untouched.

    :param dataset_folder: the full path to the dataset containing directory
    :type dataset_folder: str
    :param split: percentage of the test set (based on the full dataset), 0 < split < 1
    :type split: float
    :param symbolic: whether to use symbolic links for the new datasets, makes copies if false
    :type symbolic: bool
    :param seed: controls the random state of dataset creation.
    :type seed: int
    :param val_split: percentage of the train set to be used as validation set, 0 < val_split < 1 - default=0.1
    :type val_split: float
    :param shuffle: whether to shuffle the data before splitting - default=True
    :type shuffle: bool
    :return: None
    :rtype: None
    """

    target_folder = dataset_folder if target_folder is None else target_folder

    class_dirs = _list_subdirs(dataset_folder)
    images = []
    labels = []

    for cls in class_dirs:
        full_class_path = os.path.join(dataset_folder, cls)
        class_image_files = _list_files(full_class_path)
        class_labels = [cls] * len(class_image_files)

        images.extend(class_image_files)
        labels.extend(class_labels)

    X_train, X_test, y_train, y_test = train_test_split(images, labels,
                                                        test_size=split,
                                                        random_state=seed,
                                                        stratify=labels,
                                                        shuffle=shuffle)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=val_split,
                                                      random_state=seed,
                                                      stratify=y_train,
                                                      shuffle=shuffle)

    _create_dataset("train", dataset_folder, target_folder, X_train, y_train, symbolic)
    _create_dataset("test", dataset_folder, target_folder, X_test, y_test, symbolic)
    _create_dataset("val", dataset_folder, target_folder, X_val, y_val, symbolic)


def _move_dir(full_path_dir, full_path_new_dir):
    shutil.move(full_path_dir, full_path_new_dir)
    return full_path_new_dir


def _check_dir_exists(full_path_dir):
    if not os.path.isdir(full_path_dir):
        print("Folder specified in args.dataset_folder={} does not exist!".format(full_path_dir))
        sys.exit(-1)


def _list_subdirs(path):
    return [e for e in os.listdir(path) if os.path.isdir(os.path.join(path, e))]


def _list_files(path):
    return [e for e in os.listdir(path) if not os.path.isdir(os.path.join(path, e))]


def _make_folder_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def _copy_to_class_folder(new_dataset_dir, original_dataset_dir, file_names, labels, symbolic):
    for l in set(labels):
        new_label_dir = os.path.join(new_dataset_dir, l)
        _make_folder_if_not_exists(new_label_dir)

    for f_name, label in zip(file_names, labels):
        file_path_original = os.path.join(original_dataset_dir, label, f_name)
        file_path_new = os.path.join(new_dataset_dir, label, f_name)

        if symbolic:
            os.symlink(file_path_original, file_path_new)
        else:
            shutil.copy(file_path_original, file_path_new)


def _create_dataset(name, dataset_folder, target_folder, file_names, labels, symbolic):
    _check_dir_exists(dataset_folder)

    new_dataset_dir = os.path.join(target_folder, name)
    _make_folder_if_not_exists(new_dataset_dir)
    _check_dir_exists(new_dataset_dir)

    _copy_to_class_folder(new_dataset_dir, dataset_folder, file_names, labels, symbolic)


# FIXME: misleading name, it should rather be called 'split_dataset_train_val' or something of the likes
def split_dataset(dataset_folder, split, symbolic):
    """
    Partition a dataset into train/val splits on the filesystem.

    Parameters
    ----------
    dataset_folder : str
        Path to the dataset folder (see datasets.image_folder_dataset.load_dataset for details).
    split : float
        Specifies how much of the training set should be converted into the validation set.
    symbolic : bool
        Does not make a copy of the data, but only symbolic links to the original data

    Returns
    -------
        None
    """

    # Getting the train dir
    traindir = os.path.join(dataset_folder, 'train')

    # Rename the original train dir
    shutil.move(traindir, os.path.join(dataset_folder, 'original_train'))
    traindir = os.path.join(dataset_folder, 'original_train')

    # Sanity check on the training folder
    if not os.path.isdir(traindir):
        print("Train folder not found in the args.dataset_folder={}".format(dataset_folder))
        sys.exit(-1)

    # Load the dataset file names

    train_ds = datasets.ImageFolder(traindir)

    # Extract the actual file names and labels as entries
    fileNames = np.asarray([item[0] for item in train_ds.imgs])
    labels = np.asarray([item[1] for item in train_ds.imgs])

    # Split the data into two sets
    X_train, X_val, y_train, y_val = train_test_split(fileNames, labels,
                                                      test_size=split,
                                                      random_state=42,
                                                      stratify=labels)

    # Print number of elements for each class
    for c in train_ds.classes:
        print("labels ({}) {}".format(c, np.size(np.where(y_train == train_ds.class_to_idx[c]))))
    for c in train_ds.classes:
        print("split_train ({}) {}".format(c, np.size(np.where(y_train == train_ds.class_to_idx[c]))))
    for c in train_ds.classes:
        print("split_val ({}) {}".format(c, np.size(np.where(y_val == train_ds.class_to_idx[c]))))

    # Create the folder structure to accommodate the two new splits
    split_train_dir = os.path.join(dataset_folder, "train")
    if os.path.exists(split_train_dir):
        shutil.rmtree(split_train_dir)
    os.makedirs(split_train_dir)

    for class_label in train_ds.classes:
        path = os.path.join(split_train_dir, class_label)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    split_val_dir = os.path.join(dataset_folder, "val")
    if os.path.exists(split_val_dir):
        shutil.rmtree(split_val_dir)
    os.makedirs(split_val_dir)

    for class_label in train_ds.classes:
        path = os.path.join(split_val_dir, class_label)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    # Copying the splits into their folders
    for X, y in zip(X_train, y_train):
        src = X
        file_name = os.path.basename(src)
        dest = os.path.join(split_train_dir, train_ds.classes[y], file_name)
        if symbolic:
            os.symlink(src, dest)
        else:
            shutil.copy(X, dest)

    for X, y in zip(X_val, y_val):
        src = X
        file_name = os.path.basename(src)
        dest = os.path.join(split_val_dir, train_ds.classes[y], file_name)
        if symbolic:
            os.symlink(src, dest)
        else:
            shutil.copy(X, dest)

    return


if __name__ == "__main__":

    # TODO: add argumetns for shuffle, val_split and seed.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='This script creates train/val splits '
                                                 'from a specified dataset folder.')

    parser.add_argument('--dataset-folder',
                        help='path to root of the dataset.',
                        required=True,
                        type=str,
                        default=None)

    parser.add_argument('--split',
                        help='Ratio of the split for validation set.'
                             'Example: if 0.2 the training set will be 80% and val 20%.',
                        type=float,
                        default=0.2)

    parser.add_argument('--symbolic',
                        help='Make symbolic links instead of copies.',
                        action='store_true',
                        default=False)

    parser.add_argument('--type',
                        help='What kind of split should be created, currently train-val and train-test-val are '
                             'supported.',
                        type=str,
                        default="train-val")

    args = parser.parse_args()

    if args.type == "train-val":
        split_dataset(dataset_folder=args.dataset_folder, split=args.split, symbolic=args.symbolic)
    elif args.type == "train-test-val":
        split_dataset_train_test_val(dataset_folder=args.dataset_folder, split=args.split, symbolic=args.symbolic)
    else:
        raise ValueError("%s is not a currently supported split type, use 'train-val' or 'train-test-val'", args.type)
