# Utils
import argparse
import inspect
import numpy as np
import os
import pandas as pd
import rarfile
import re
import requests
import shutil
import sys
import csv

# Torch
import torch
import torchvision
import urllib
import wget
import zipfile
from PIL import Image
from scipy.io import loadmat as _loadmat
from sklearn.model_selection import train_test_split as _train_test_split

# DeepDIVA
from util.data.dataset_splitter import split_dataset
from util.misc import get_all_files_in_folders_and_subfolders \
    as _get_all_files_in_folders_and_subfolders


# Torch


def _make_folder_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)




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


def _download_file(url, full_path_out, chunk_size=1024):

    r = requests.get(url, allow_redirects=True, stream=True)

    with open(full_path_out, "wb") as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)


def _unzip_file(full_filepath, output_dir):
    with open(full_filepath, 'rb') as MyZip:
        if str(MyZip.read(4)) != "\x50\x4B\x03\x04":
            print("This is not a zipfile: it might be a rar or something else.")
            sys.exit(-1)
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
                                                       'MNIST',
                                                       'processed',
                                                       'training.pt'))
    test_data, test_labels = torch.load(os.path.join(args.output_folder,
                                                     'MNIST',
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
        for i, (img, label) in enumerate(zip(arr, labels.detach().numpy())):
            dest = os.path.join(folder, str(label))
            _make_folder_if_not_exists(dest)
            Image.fromarray(img.numpy(), mode='L').save(os.path.join(dest, str(i) + '.png'))

    # Write the images to the folders
    _write_data_to_folder(train_data, train_labels, train_folder)
    _write_data_to_folder(test_data, test_labels, test_folder)

    shutil.rmtree(os.path.join(args.output_folder, 'MNIST', 'raw'))
    shutil.rmtree(os.path.join(args.output_folder, 'MNIST', 'processed'))

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
    train = _loadmat(os.path.join(args.output_folder,
                                  'train_32x32.mat'))
    train_data, train_labels = train['X'], train['y'].astype(np.int64).squeeze()
    np.place(train_labels, train_labels == 10, 0)
    train_data = np.transpose(train_data, (3, 0, 1, 2))

    test = _loadmat(os.path.join(args.output_folder,
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
    train_data, train_labels = cifar_train.data, cifar_train.targets

    test_data, test_labels = cifar_test.data, cifar_test.targets

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


def miml(args):
    """
    Fetches and prepares (in a DeepDIVA friendly format) the Multi-Instance Multi-Label Image Dataset
    on the file system. Dataset available at: http://lamda.nju.edu.cn/data_MIMLimage.ashx

    Parameters
    ----------
    args : dict
        List of arguments necessary to run this routine. In particular its necessary to provide
        output_folder as String containing the path where the dataset will be downloaded

    Returns
    -------
        None
    """
    # Download the files
    url = 'http://lamda.nju.edu.cn/files/miml-image-data.rar'
    if not os.path.exists(os.path.join(args.output_folder, 'miml-image-data.rar')):
        print('Downloading file!')
        filename = wget.download(url, out=args.output_folder)
    else:
        print('File already downloaded!')
        filename = os.path.join(args.output_folder, 'miml-image-data.rar')

    # Extract the files
    path_to_rar = filename
    path_to_output = os.path.join(args.output_folder, 'tmp_miml')
    rarfile.RarFile(path_to_rar).extractall(path_to_output)
    path_to_rar = os.path.join(path_to_output, 'original.rar')
    rarfile.RarFile(path_to_rar).extractall(path_to_output)
    path_to_rar = os.path.join(path_to_output, 'processed.rar')
    rarfile.RarFile(path_to_rar).extractall(path_to_output)
    print('Extracted files...')

    # Load the mat file
    mat = _loadmat(os.path.join(path_to_output, 'miml data.mat'))
    targets = mat['targets'].T
    classes = [item[0][0] for item in mat['class_name']]
    # Add filename at 0-index to correctly format the CSV headers
    classes.insert(0, 'filename')

    # Get list of all image files in the folder
    images = [item for item in _get_all_files_in_folders_and_subfolders(path_to_output)
              if item.endswith('jpg')]
    images = sorted(images, key=lambda e: int(os.path.basename(e).split('.')[0]))

    # Make splits
    train_data, test_data, train_labels, test_labels = _train_test_split(images, targets, test_size=0.2,
                                                                         random_state=42)
    train_data, val_data, train_labels, val_labels = _train_test_split(train_data, train_labels, test_size=0.2,
                                                                       random_state=42)

    # print('Size of splits\ntrain:{}\nval:{}\ntest:{}'.format(len(train_data),
    #                                                     len(val_data),
    #                                                     len(test_data)))

    # Make output folders
    dataset_root = os.path.join(args.output_folder, 'MIML')
    train_folder = os.path.join(dataset_root, 'train')
    val_folder = os.path.join(dataset_root, 'val')
    test_folder = os.path.join(dataset_root, 'test')

    _make_folder_if_not_exists(dataset_root)
    _make_folder_if_not_exists(train_folder)
    _make_folder_if_not_exists(val_folder)
    _make_folder_if_not_exists(test_folder)

    def _write_data_to_folder(data, labels, folder, classes):
        dest = os.path.join(folder, 'images')
        _make_folder_if_not_exists(dest)
        for image, label in zip(data, labels):
            shutil.copy(image, dest)

        rows = np.column_stack(([os.path.join('images', os.path.basename(item)) for item in data], labels))
        rows = sorted(rows, key=lambda e: int(e[0].split('/')[1].split('.')[0]))
        output_csv = pd.DataFrame(rows)
        output_csv.to_csv(os.path.join(folder, 'labels.csv'), header=classes, index=False)
        return

    # Write the images to the correct folders
    print('Writing the data to the filesystem')
    _write_data_to_folder(train_data, train_labels, train_folder, classes)
    _write_data_to_folder(val_data, val_labels, val_folder, classes)
    _write_data_to_folder(test_data, test_labels, test_folder, classes)

    os.remove(filename)
    shutil.rmtree(path_to_output)
    print('All done!')
    return



def diva_hisdb(args):
    """
    Fetches and prepares (in a DeepDIVA friendly format) the DIVA HisDB-all dataset for semantic segmentation to the location specified
    on the file system
    See also: https://diuf.unifr.ch/main/hisdoc/diva-hisdb
    Output folder structure: ../HisDB/CB55/train
                             ../HisDB/CB55/val
                             ../HisDB/CB55/test
                             ../HisDB/CB55/test/data -> images
                             ../HisDB/CB55/test/gt   -> pixel-wise annotated ground truth
    Parameters
    ----------
    args : dict
        List of arguments necessary to run this routine. In particular its necessary to provide
        output_folder as String containing the path where the dataset will be downloaded
    Returns
    -------
        None
    """
    # make the root folder
    dataset_root = os.path.join(args.output_folder, 'HisDB')
    _make_folder_if_not_exists(dataset_root)

    # links to HisDB data sets
    link_public = urllib.parse.urlparse(
        'https://diuf.unifr.ch/main/hisdoc/sites/diuf.unifr.ch.main.hisdoc/files/uploads/diva-hisdb/hisdoc/all.zip')
    link_test_private = urllib.parse.urlparse(
        'https://diuf.unifr.ch/main/hisdoc/sites/diuf.unifr.ch.main.hisdoc/files/uploads/diva-hisdb/hisdoc/private-test/all-privateTest.zip')
    download_path_public = os.path.join(dataset_root, link_public.geturl().rsplit('/', 1)[-1])
    download_path_private = os.path.join(dataset_root, link_test_private.geturl().rsplit('/', 1)[-1])

    # download files
    print('Downloading {}...'.format(link_public.geturl()))
    urllib.request.urlretrieve(link_public.geturl(), download_path_public)

    print('Downloading {}...'.format(link_test_private.geturl()))
    urllib.request.urlretrieve(link_test_private.geturl(), download_path_private)
    print('Download complete. Unpacking files...')

    # unpack relevant folders
    zip_file = zipfile.ZipFile(download_path_public)

    # unpack imgs and gt
    data_gt_zip = {f: re.sub(r'img', 'pixel-level-gt', f) for f in zip_file.namelist() if 'img' in f}
    dataset_folders = [data_file.split('-')[-1][:-4] for data_file in data_gt_zip.keys()]
    for data_file, gt_file in data_gt_zip.items():
        dataset_name = data_file.split('-')[-1][:-4]
        dataset_folder = os.path.join(dataset_root, dataset_name)
        _make_folder_if_not_exists(dataset_folder)

        for file in [data_file, gt_file]:
            zip_file.extract(file, dataset_folder)
            with zipfile.ZipFile(os.path.join(dataset_folder, file), "r") as zip_ref:
                zip_ref.extractall(dataset_folder)
                # delete zips
                os.remove(os.path.join(dataset_folder, file))

        # create folder structure
        for partition in ['train', 'val', 'test', 'test-public']:
            for folder in ['data', 'gt']:
                _make_folder_if_not_exists(os.path.join(dataset_folder, partition, folder))

    # move the files to the correct place
    for folder in dataset_folders:
        for k1, v1 in {'pixel-level-gt': 'gt', 'img': 'data'}.items():
            for k2, v2 in {'public-test': 'test-public', 'training': 'train', 'validation': 'val'}.items():
                current_path = os.path.join(dataset_root, folder, k1, k2)
                new_path = os.path.join(dataset_root, folder, v2, v1)
                for f in [f for f in os.listdir(current_path) if os.path.isfile(os.path.join(current_path, f))]:
                    shutil.move(os.path.join(current_path, f), os.path.join(new_path, f))
            # remove old folders
            shutil.rmtree(os.path.join(dataset_root, folder, k1))

    # fix naming issue
    for old, new in {'CS18': 'CSG18', 'CS863': 'CSG863'}.items():
        os.rename(os.path.join(dataset_root, old), os.path.join(dataset_root, new))

    # unpack private test folders
    zip_file_private = zipfile.ZipFile(download_path_private)

    data_gt_zip_private = {f: re.sub(r'img', 'pixel-level-gt', f) for f in zip_file_private.namelist() if 'img' in f}

    for data_file, gt_file in data_gt_zip_private.items():
        dataset_name = re.search('-(.*)-', data_file).group(1)
        dataset_folder = os.path.join(dataset_root, dataset_name)

        for file in [data_file, gt_file]:
            zip_file_private.extract(file, dataset_folder)
            with zipfile.ZipFile(os.path.join(dataset_folder, file), "r") as zip_ref:
                zip_ref.extractall(os.path.join(dataset_folder, file[:-4]))
            # delete zip
            os.remove(os.path.join(dataset_folder, file))

        # create folder structure
        for folder in ['data', 'gt']:
            _make_folder_if_not_exists(os.path.join(dataset_folder, 'test', folder))

        for old, new in {'pixel-level-gt': 'gt', 'img': 'data'}.items():
            current_path = os.path.join(dataset_folder, "{}-{}-privateTest".format(old, dataset_name), dataset_name)
            new_path = os.path.join(dataset_folder, "test", new)
            for f in [f for f in os.listdir(current_path) if os.path.isfile(os.path.join(current_path, f))]:
                shutil.move(os.path.join(current_path, f), os.path.join(new_path, f))

            # remove old folders
            shutil.rmtree(os.path.dirname(current_path))

    print('Finished. Data set up at {}.'.format(dataset_root))


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

    split_dataset(
        dataset_folder=full_dir_unzipped,
        split=0.2,
        symbolic=False,
        target_folder=output_dir)


def icdar2017_clamm(args):

    url = "http://clamm.irht.cnrs.fr/wp-content/uploads/ICDAR2017_CLaMM_Training.zip"
    print("Downloading " + url)
    zip_name = "ICDAR2017_CLaMM_Training.zip"
    local_filename, headers = urllib.request.urlretrieve(url, zip_name)
    zfile = zipfile.ZipFile(local_filename)

    # Make output folders
    dataset_root = os.path.join(args.output_folder, 'ICDAR2017-CLAMM')
    dataset_manuscriptDating = os.path.join(dataset_root , 'ManuscriptDating')
    dataset_md_train = os.path.join(dataset_manuscriptDating , 'train')
    dataset_styleClassification = os.path.join(dataset_root , 'StyleClassification')
    dataset_sc_train = os.path.join(dataset_styleClassification, 'train')
    test_sc_folder = os.path.join(dataset_styleClassification, 'test')
    test_md_folder = os.path.join(dataset_manuscriptDating, 'test')

    _make_folder_if_not_exists(dataset_root)
    _make_folder_if_not_exists(dataset_manuscriptDating)
    _make_folder_if_not_exists(dataset_styleClassification)
    _make_folder_if_not_exists(test_sc_folder)

    def _write_data_to_folder(zipfile, filenames, labels, folder, start_index,  isTest):
        print("Writing data\n")
        sorted_labels = [None]*len(labels)
        if isTest == 1:
            for i in range(len(zipfile.infolist())):
                entry = zipfile.infolist()[i]
                if "IRHT_P_009793.tif" in entry.filename:
                    zipfile.infolist().remove(entry)
                    break

        zip_infolist = zipfile.infolist()[1:]

        for i in range(len(zip_infolist)):
            entry = zip_infolist[i]
            entry_index_infilenames = filenames.index(entry.filename[start_index:])
            sorted_labels[i] = labels[entry_index_infilenames]

        for i, (enrty, label) in enumerate(zip(zipfile.infolist()[1:], sorted_labels)):
            with zipfile.open(enrty) as file:
                img = Image.open(file)
                dest = os.path.join(folder, str(label))
                _make_folder_if_not_exists(dest)
                img.save(os.path.join(dest, str(i) + '.png'), "PNG", quality=100)

    def getLabels(zfile):
        print("Extracting labels\n")
        filenames, md_labels, sc_labels = [], [], []
        zip_infolist = zfile.infolist()[1:]
        for entry in zip_infolist:
            if '.csv' in entry.filename:
                with zfile.open(entry) as file:
                    cf = file.read()
                    c = csv.StringIO(cf.decode())
                    next(c) # Skip the first line which is the header of csv file
                    for row in c:

                        md_label_strt_ind = row.rfind(';')
                        md_label_end_ind = row.rfind("\r")
                        md_labels.append(row[md_label_strt_ind+1:md_label_end_ind])
                        sc_labels_strt_ind = row[:md_label_strt_ind].rfind(';')
                        sc_labels.append(row[sc_labels_strt_ind+1:md_label_strt_ind])
                        filename_ind = row[:sc_labels_strt_ind].rfind(';')

                        if filename_ind > -1:
                            f_name = row[filename_ind+1:sc_labels_strt_ind]
                        else:
                            f_name = row[:sc_labels_strt_ind]
                        if isTest == 1 and f_name == 'IRHT_P_009783.tif':
                            print('No file named ' + f_name + ". This filename will not be added!")
                        else:
                            filenames.append(f_name)

                zfile.infolist().remove(entry) # remove the csv file from infolist
            if '.db' in entry.filename: # remove the db file from infolist
                zfile.infolist().remove(entry)
        return filenames, sc_labels, md_labels

    isTest = 0
    filenames, sc_labels, md_labels = getLabels(zfile)
    start_index_training = len("ICDAR2017_CLaMM_Training/")
    print("Training data is being prepared for style classification!\n")
    _write_data_to_folder(zfile, filenames, sc_labels, dataset_sc_train, start_index_training, isTest)
    print("Training data is being prepared for manuscript dating!\n")
    _write_data_to_folder(zfile, filenames, md_labels, dataset_md_train, start_index_training, isTest)

    os.remove(os.path.join(zfile.filename))

    url = "http://clamm.irht.cnrs.fr/wp-content/uploads/ICDAR2017_CLaMM_task1_task3.zip"
    print("Downloading " + url)
    zip_name_test = "ICDAR2017_CLaMM_task1_task3.zip"
    local_filename_test, headers_test = urllib.request.urlretrieve(url, zip_name_test)
    zfile_test = zipfile.ZipFile(local_filename_test)

    isTest = 1
    filenames_test, sc_test_labels, md_test_labels = getLabels(zfile_test)
    start_index_test = len("ICDAR2017_CLaMM_task1_task3/")
    print("Test data is being prepared for style classification!\n")
    _write_data_to_folder(zfile_test, filenames_test, sc_test_labels, test_sc_folder, start_index_test, 1)
    print("Test data is being prepared for manuscript dating!\n")
    _write_data_to_folder(zfile_test, filenames_test, md_test_labels, test_md_folder, start_index_test, 1)

    os.remove(os.path.join(zfile_test.filename))
    print("Training-Validation splitting\n")
    split_dataset(dataset_folder=dataset_manuscriptDating, split=0.2, symbolic=False)
    split_dataset(dataset_folder=dataset_styleClassification, split=0.2, symbolic=False)
    print("ICDAR2017 CLaMM data is ready!")


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


def HAM10000(output_folder, **kwargs):
    """
    Fetches and prepares (in a DeepDIVA friendly format) the HAM10000 dataset to the location specified
    on the file system
    Dataset presented at:
        https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
    However, you have to manually download it from here:
        https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000/downloads/skin-cancer-mnist-ham10000.zip/2

    IMPORTANT:
    It is expected that the main folder is there after you manually download it (its behind terms&condition acceptance)
    Moreover, extract the zip and put every image in a folder called images as follows:

    /output_folder/HAM10000
        HAM10000_metadata.csv
        images
            file0.jpg
            ...
            file10014.pjg

    This script will then read the metadata and format the dataset as expected from DeepDIVA.

    Parameters
    ----------
    output_folder : str
        String containing the path where the dataset will be downloaded

    Returns
    -------
        None
    """

    DIR_NAME = "HAM10000"
    output_dir = os.path.join(output_folder, DIR_NAME)
    IMAGES_DIR = os.path.join(output_dir, "images")

    # Convert images to PNG
    #_convert_images(IMAGES_DIR)

    # Load the CSV
    f = open(os.path.join(output_dir,'HAM10000_metadata.csv'))
    csv_f = csv.reader(f)
    next(csv_f) # Skip header line

    # Process the images and put them in their proper folder
    for row in csv_f:
        # Create if necessary the class folder
        class_folder_path = os.path.join(IMAGES_DIR, "train", row[2])  # row[2] is the class (dx)
        _make_folder_if_not_exists(class_folder_path)
        #Move the image to proper location
        os.rename(os.path.join(IMAGES_DIR, row[1] + ".png"), os.path.join(class_folder_path, row[1] + ".png"))  # row[1] is the filename (image_id)

    # Split train/test
    split_dataset(dataset_folder=IMAGES_DIR, split=0.2, symbolic=False)
    # Mirror the val/test
    print("WARNING: THERE IS NO TEST SET FOR THIS DATASET")
    os.system("cp -rf {src:s} {dst:s}".format(src=os.path.join(IMAGES_DIR, "val"), dst=os.path.join(IMAGES_DIR, "test")))
    os.system("mv images / *./")
    os.system("rm HAM10000_metadata.csv")
    os.system("rm -rf images")

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

    getattr(sys.modules[__name__], args.dataset)(**args.__dict__)
