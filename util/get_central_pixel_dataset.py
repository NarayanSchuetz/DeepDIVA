import os
import argparse
import random
import numpy as np


def get_all_files_in_folders_and_subfolders(root_dir=None):
    """Get all the files in a folder and sub-folders.

    Parameters
    ----------
    root_dir : str
        All files in this directory and it's sub-folders will be returned by this method.

    Returns
    -------
    paths : list of str
        List of paths to all files in this folder and it's subfolders.
    """
    paths = []
    for path, subdirs, files in os.walk(root_dir):
        for name in files:
            paths.append(os.path.join(path, name))
    return paths


def convert_to_single_label(gt):
    # Override text area with background
    locs = np.where(gt[:, :, 2] == 128)
    gt[locs[0], locs[1], 0] = 1

    new_img = np.zeros((gt.shape[0], gt.shape[1]), dtype=np.uint8)

    # set deco
    any_deco_locs = np.where(gt[:, :, 0] == 6)
    gt[any_deco_locs[0], any_deco_locs[1], 0] = 4

    any_deco_locs = np.where(gt[:, :, 0] == 12)
    gt[any_deco_locs[0], any_deco_locs[1], 0] = 4

    any_deco_locs = np.where(gt[:, :, 0] == 14)
    gt[any_deco_locs[0], any_deco_locs[1], 0] = 4

    # set comment
    any_comment_locs = np.where(gt[:, :, 0] == 10)
    gt[any_comment_locs[0], any_comment_locs[1], 0] = 2

    txt_locs = np.where(gt[:, :, 0] == 8)
    # new_img[txt_locs[0], txt_locs[1]] = np.array([0,0,255])
    new_img[txt_locs[0], txt_locs[1]] = 1

    deco_locs = np.where(gt[:, :, 0] == 4)
    # new_img[deco_locs[0], deco_locs[1]] = np.array([0,255,0])
    new_img[deco_locs[0], deco_locs[1]] = 2

    comment_locs = np.where(gt[:, :, 0] == 2)
    # new_img[comment_locs[0], comment_locs[1]] = np.array([255,0,0])
    new_img[comment_locs[0], comment_locs[1]] = 3

    return new_img


def get_gt_path(path):
    return os.path.join('/', *path.split('/')[:-2], 'gt', path.split('/')[-1][:-4] + '.png')


def get_crop_around_point(images, loc, half_crop):
    x = loc[1]
    y = loc[2]
    return images[loc[0], x:x + 2 * half_crop + 1, y:y + 2 * half_crop + 1, :]


def _make_folder_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main(args):
    root = args.root
    output = args.output
    crop_size = args.crop_size
    half_crop = int(crop_size / 2)

    splits = zip(['train', 'val', 'test'], [0.7, 0.15, 0.15])

    for folder, size in splits:
        print("Current split is {}".format(folder))
        current = os.path.join(root, folder, 'data')

        files = get_all_files_in_folders_and_subfolders(current)

        # Load all data images
        images = np.array([Image.open(item) for item in files])

        # Load all gt images1
        gts = np.array([convert_to_single_label(Image.open(get_gt_path(item))) for item in files])

        num_classes = len(np.unique(gts))

        split_size = int(args.total_size * size)

        for cls in range(num_classes):
            print("Current class is {}".format(cls))

            class_size = int(split_size / num_classes)
            locs = np.where(gts[:, half_crop:-half_crop, half_crop:-half_crop] == cls)
            locs = list(zip(locs[0], locs[1], locs[2]))
            random.shuffle(locs)
            locs = locs[:class_size]

            target_folder = os.path.join(output, folder, str(cls))
            _make_folder_if_not_exists(target_folder)

            for i, item in enumerate(locs):
                img = get_crop_around_point(images, item, half_crop)
                Image.save(os.path.join(target_folder, str(i) + '.png'), img)

    print("ALL DONE!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='This script can be used to download some '
                                                 'datasets and prepare them in a standard format')

    parser.add_argument('--root',
                        help='Path to a root manuscript folder')
    parser.add_argument('--output',
                        help='Path to an output folder')
    parser.add_argument('--crop-size',
                        help='Size to crop around each image',
                        default=32,
                        type=int)
    parser.add_argument('--total-size',
                        help='Total number of images',
                        default=100000)

    args = parser.parse_args()
    main(args)
