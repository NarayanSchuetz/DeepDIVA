# Utils
import argparse
import inspect
import os
import shutil
import sys

import numpy as np
import scipy
# Torch
import torch
import torchvision
from PIL import Image

# DeepDIVA
from util.data.dataset_splitter import split_dataset


# Utils
import argparse
import inspect
import os
import shutil
import sys
import requests
import zipfile

import numpy as np
import scipy

# Torch
import torch
import torchvision
from PIL import Image

# DeepDIVA
from util.data.dataset_splitter import split_dataset, split_dataset_train_test_val


def hist_colorectal(args):
    """
    Fetches and prepares teh hist dataset.
    :param args:
    :type args:
    :return:
    :rtype:
    """
    DIR_NAME = "ColorectalHist"
    URL = "https://zenodo.org/record/53169/files/Kather_texture_2016_image_tiles_5000.zip?download=1"
    UNZIPPED_DIR_NAME = "Kather_texture_2016_image_tiles_5000"

    output_dir = os.path.join(args.output_folder, DIR_NAME)
    full_path_zipfile = os.path.join(args.output_folder, "hist_tmp.zip")

    _download_file(URL, full_path_zipfile)
    _unzip_file(full_path_zipfile, output_dir=output_dir)
    _remove_file(full_path_zipfile)

    full_dir_unzipped = os.path.join(output_dir, UNZIPPED_DIR_NAME)

    labels = _list_subdirs(full_dir_unzipped)
    for label in labels:
        label_dir = os.path.join(full_dir_unzipped, label)
        _convert_images(label_dir)

    split_dataset_train_test_val(
        dataset_folder=full_dir_unzipped,
        split=0.2,
        symbolic=False,
        target_folder=output_dir)


def _convert_images(image_folder, to_format="png", delete=True):
    """Convert image types"""

    images = _list_files(image_folder)

    for img in images:
        img_path_src = os.path.join(image_folder, img)
        img_path_wo_extension, _ = os.path.splitext(img_path_src)
        img_path_target = os.path.join(img_path_wo_extension + "." + to_format)

        im = Image.open(img_path_src)
        im.save(img_path_target)

        if delete:
            os.remove(img_path_src)


def irmas(args, parts="both"):
    """
    Fetches and prepares the IRMAS instrument dataset (https://zenodo.org/record/1290750#.W-2s4S2ZPa5)

    :param args: List of arguments necessary to run this routine. In particular its necessary to provide an
                 output_folder as String containing the path where the dataset will be downloaded.
    :type args: dict
    :param parts: which part to load (train | test | both) default=train
    :type parts: str
    :return: None
    :rtype: None
    """
    DIR_NAME = "IRMAS"
    URL_TRAIN = "https://zenodo.org/record/1290750/files/IRMAS-TrainingData.zip?download=1"
    URL_TEST_TEMPLATE = "https://zenodo.org/record/1290750/files/IRMAS-TestingData-Part{}.zip?download=1"
    TEST_PARTS = (1, 2, 3)

    urls = []

    if parts == "train":
        urls.append(URL_TRAIN)
    elif parts == "test":
        urls.extend([URL_TEST_TEMPLATE.format(i) for i in TEST_PARTS])
    elif parts == "both":
        urls.append(URL_TRAIN)
        urls.extend([URL_TEST_TEMPLATE.format(i) for i in TEST_PARTS])
    else:
        raise ValueError("argument %s for parameter 'parts' is invalid, use (train | test | both)" % parts)

    for url in urls:
        full_path_zipfile = os.path.join(args.output_folder, "irmas_tmp.zip")
        _download_file(url, full_path_zipfile)
        _unzip_file(full_path_zipfile, os.path.join(args.output_folder, DIR_NAME))
        _remove_file(full_path_zipfile)


def _download_file(url, full_path_out, chunk_size=1024):

    r = requests.get(url, allow_redirects=True, stream=True)

    with open(full_path_out, "wb") as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)


def _unzip_file(full_filepath, output_dir):
    with zipfile.ZipFile(full_filepath, "r") as f:
        f.extractall(output_dir)


def _remove_file(full_filepath):
    os.remove(full_filepath)


def _list_subdirs(path):
    return [e for e in os.listdir(path) if os.path.isdir(os.path.join(path, e))]


def _list_files(path):
    return [e for e in os.listdir(path) if not os.path.isdir(os.path.join(path, e))]


def mnist(args):
    """
    Fetches and prepares (in a DeepDIVA friendly format) the MNIST dataset to the location specified
    on the file system

    Parameters
    ----------
    args : dict
        List of arguments necessary to run this routine. In particular its necessary to provide
        output_folder as String containing the path where the dataset will be downloaded

    Returns
    -------
        None
    """
    # Use torchvision to download the dataset
    torchvision.datasets.MNIST(root=args.output_folder, download=True)

    # Load the data into memory
    train_data, train_labels = torch.load(os.path.join(args.output_folder,
                                                       'processed',
                                                       'training.pt'))
    test_data, test_labels = torch.load(os.path.join(args.output_folder,
                                                     'processed',
                                                     'test.pt'))

    # Make output folders
    dataset_root = os.path.join(args.output_folder, 'MNIST')
    train_folder = os.path.join(dataset_root, 'train')
    test_folder = os.path.join(dataset_root, 'test')

    _make_folder_if_not_exists(dataset_root)
    _make_folder_if_not_exists(train_folder)
    _make_folder_if_not_exists(test_folder)

    def _write_data_to_folder(arr, labels, folder):
        for i, (img, label) in enumerate(zip(arr, labels)):
            dest = os.path.join(folder, str(label))
            _make_folder_if_not_exists(dest)
            Image.fromarray(img.numpy(), mode='L').save(os.path.join(dest, str(i) + '.png'))

    # Write the images to the folders
    _write_data_to_folder(train_data, train_labels, train_folder)
    _write_data_to_folder(test_data, test_labels, test_folder)

    shutil.rmtree(os.path.join(args.output_folder, 'raw'))
    shutil.rmtree(os.path.join(args.output_folder, 'processed'))

    split_dataset(dataset_folder=dataset_root, split=0.2, symbolic=False)


def fashion_mnist(args):
    """
    Fetches and prepares (in a DeepDIVA friendly format) the MNIST dataset to the location specified
    on the file system

    Parameters
    ----------
    args : dict
        List of arguments necessary to run this routine. In particular its necessary to provide
        output_folder as String containing the path where the dataset will be downloaded

    Returns
    -------
        None
    """
    # Use torchvision to download the dataset
    torchvision.datasets.FashionMNIST(root=args.output_folder, download=True)

    # Load the data into memory
    train_data, train_labels = torch.load(os.path.join(args.output_folder,
                                                       'processed',
                                                       'training.pt'))
    test_data, test_labels = torch.load(os.path.join(args.output_folder,
                                                     'processed',
                                                     'test.pt'))

    # Make output folders
    dataset_root = os.path.join(args.output_folder, 'FashionMNIST')
    train_folder = os.path.join(dataset_root, 'train')
    test_folder = os.path.join(dataset_root, 'test')

    _make_folder_if_not_exists(dataset_root)
    _make_folder_if_not_exists(train_folder)
    _make_folder_if_not_exists(test_folder)

    def _write_data_to_folder(arr, labels, folder):
        for i, (img, label) in enumerate(zip(arr, labels)):
            dest = os.path.join(folder, str(label))
            _make_folder_if_not_exists(dest)
            Image.fromarray(img.numpy(), mode='L').save(os.path.join(dest, str(i) + '.png'))

    # Write the images to the folders
    _write_data_to_folder(train_data, train_labels, train_folder)
    _write_data_to_folder(test_data, test_labels, test_folder)

    shutil.rmtree(os.path.join(args.output_folder, 'raw'))
    shutil.rmtree(os.path.join(args.output_folder, 'processed'))

    split_dataset(dataset_folder=dataset_root, split=0.2, symbolic=False)


def svhn(args):
    """
    Fetches and prepares (in a DeepDIVA friendly format) the SVHN dataset to the location specified
    on the file system

    Parameters
    ----------
    args : dict
        List of arguments necessary to run this routine. In particular its necessary to provide
        output_folder as String containing the path where the dataset will be downloaded

    Returns
    -------
        None
    """
    # Use torchvision to download the dataset
    torchvision.datasets.SVHN(root=args.output_folder, split='train', download=True)
    torchvision.datasets.SVHN(root=args.output_folder, split='test', download=True)

    # Load the data into memory
    train = scipy.io.loadmat(os.path.join(args.output_folder,
                                          'train_32x32.mat'))
    train_data, train_labels = train['X'], train['y'].astype(np.int64).squeeze()
    np.place(train_labels, train_labels == 10, 0)
    train_data = np.transpose(train_data, (3, 0, 1, 2))

    test = scipy.io.loadmat(os.path.join(args.output_folder,
                                         'test_32x32.mat'))
    test_data, test_labels = test['X'], test['y'].astype(np.int64).squeeze()
    np.place(test_labels, test_labels == 10, 0)
    test_data = np.transpose(test_data, (3, 0, 1, 2))

    # Make output folders
    dataset_root = os.path.join(args.output_folder, 'SVHN')
    train_folder = os.path.join(dataset_root, 'train')
    test_folder = os.path.join(dataset_root, 'test')

    _make_folder_if_not_exists(dataset_root)
    _make_folder_if_not_exists(train_folder)
    _make_folder_if_not_exists(test_folder)

    def _write_data_to_folder(arr, labels, folder):
        for i, (img, label) in enumerate(zip(arr, labels)):
            dest = os.path.join(folder, str(label))
            _make_folder_if_not_exists(dest)
            Image.fromarray(img).save(os.path.join(dest, str(i) + '.png'))

    # Write the images to the folders
    _write_data_to_folder(train_data, train_labels, train_folder)
    _write_data_to_folder(test_data, test_labels, test_folder)

    os.remove(os.path.join(args.output_folder, 'train_32x32.mat'))
    os.remove(os.path.join(args.output_folder, 'test_32x32.mat'))

    split_dataset(dataset_folder=dataset_root, split=0.2, symbolic=False)


def cifar10(args):
    """
    Fetches and prepares (in a DeepDIVA friendly format) the CIFAR dataset to the location specified
    on the file system

    Parameters
    ----------
    args : dict
        List of arguments necessary to run this routine. In particular its necessary to provide
        output_folder as String containing the path where the dataset will be downloaded

    Returns
    -------
        None
    """
    # Use torchvision to download the dataset
    cifar_train = torchvision.datasets.CIFAR10(root=args.output_folder, train=True, download=True)
    cifar_test = torchvision.datasets.CIFAR10(root=args.output_folder, train=False, download=True)

    # Load the data into memory
    train_data, train_labels = cifar_train.train_data, cifar_train.train_labels

    test_data, test_labels = cifar_test.test_data, cifar_test.test_labels

    # Make output folders
    dataset_root = os.path.join(args.output_folder, 'CIFAR10')
    train_folder = os.path.join(dataset_root, 'train')
    test_folder = os.path.join(dataset_root, 'test')

    _make_folder_if_not_exists(dataset_root)
    _make_folder_if_not_exists(train_folder)
    _make_folder_if_not_exists(test_folder)

    def _write_data_to_folder(arr, labels, folder):
        for i, (img, label) in enumerate(zip(arr, labels)):
            dest = os.path.join(folder, str(label))
            _make_folder_if_not_exists(dest)
            Image.fromarray(img).save(os.path.join(dest, str(i) + '.png'))

    # Write the images to the folders
    _write_data_to_folder(train_data, train_labels, train_folder)
    _write_data_to_folder(test_data, test_labels, test_folder)

    os.remove(os.path.join(args.output_folder, 'cifar-10-python.tar.gz'))
    shutil.rmtree(os.path.join(args.output_folder, 'cifar-10-batches-py'))

    split_dataset(dataset_folder=dataset_root, split=0.2, symbolic=False)


def _make_folder_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    downloadable_datasets = [name[0] for name in inspect.getmembers(sys.modules[__name__],
                                                                    inspect.isfunction) if not name[0].startswith('_')]
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='This script can be used to download some '
                                                 'datasets and prepare them in a standard format')

    parser.add_argument('--dataset',
                        help='name of the dataset',
                        type=str,
                        choices=downloadable_datasets)
    parser.add_argument('--output-folder',
                        help='path to where the dataset should be generated.',
                        required=False,
                        type=str,
                        default='./data/')
    args = parser.parse_args()

    getattr(sys.modules[__name__], args.dataset)(args)
