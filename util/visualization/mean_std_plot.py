import argparse
import os

import matplotlib as mpl
# To facilitate plotting on a headless server
from util.misc import tensor_to_image, save_numpy_image

mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_mean_std(x=None, arr=None, suptitle='', title='', xlabel='X', ylabel='Y', xlim=None, ylim=None):
    """
    Plots the accuracy/loss curve over several runs with standard deviation and mean.
    Parameters
    ----------
    x: numpy.ndarray
        contains the ticks on the x-axis
    arr: numpy.ndarray
        contains the accuracy values for each epoch per run
    suptitle: str
        title for the plot
    title: str
        sub-title for the plot
    xlabel: str
        label for the x-axis
    ylabel: str
        label for the y-axis
    xlim: float or None
        optionally specify a upper limit on the x-axis
    ylim: float or None
        optionally specify a upper limit on the y-axis

    Returns
    -------
    data: numpy.ndarray
        Contains an RGB image of the plotted accuracy curves
    """
    fig = plt.figure(1)
    arr_mean = np.mean(arr, 0)
    arr_std = np.std(arr, 0)
    arr_min = np.min(arr, 0)
    arr_max = np.max(arr, 0)
    with sns.axes_style('darkgrid'):
        fig.suptitle(suptitle)
        plt.title(title)
        axes = plt.gca()
        if ylim is not None:
            axes.set_ylim(ylim)
        if xlim is not None:
            axes.set_xlim(xlim)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if x is None:
            plt.plot(arr_mean, '-', color='#0000b3', label='Score')
            plt.plot(arr_min, color='#4d4dff', linestyle='dashed', label='Min')
            plt.plot(arr_max, color='#4d4dff', linestyle='dashed', label='Max')
            axes.fill_between(np.arange(len(arr_mean)), arr_mean - arr_std, arr_mean + arr_std, color='#9999ff',
                              alpha=0.2)
        else:
            plt.plot(x, arr_mean, '-', color='#0000b3', label='Score')
            plt.plot(x, arr_min, color='#4d4dff', linestyle='dashed', label='Min')
            plt.plot(x, arr_max, color='#4d4dff', linestyle='dashed', label='Max')
            axes.fill_between(np.arange(len(arr_mean)) - 1, arr_mean - arr_std, arr_mean + arr_std, color='#9999ff',
                              alpha=0.2)
        plt.legend(loc='best')

        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        fig.clf()
        plt.close()
    return data


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#                                          description='This script can be used to create nice plots with '
#                                                      'mean and std represented as shaded area as in multi-run')
#
#     parser.add_argument('--input-folder',
#                         help='path to the *.npy files.',
#                         required=True,
#                         type=str)
#     args = parser.parse_args()
#
#     print("starting...")
#
#     MODELS = ["RNDFirst", "FFTFirst", "DCTFirst"]
#     DATASETS = ["CB55", "ColorectalHist", "HAM10000", "Flowers"]
#
#     for model in MODELS:
#         for dataset in DATASETS:
#             print("cd /local/scratch/albertim/output \n"
#                   "cd multi_multi_{}_".format(model), "\n",
#                   "cd `ls`\n",
#                   "cd `ls`\n",
#                   "cd {}".format(dataset), "\n",
#                   "cd `ls`\n",
#                   "cd `ls`\n",
#                   "cd `ls`\n",
#                   "cd `ls`\n",
#                   "cd `ls`\n",
#                   "cd `ls`\n",
#                   "cd `ls`\n",
#                   "cd `ls`\n",
#                   "cd `ls`\n",
#                   "cd `ls`\n",
#                   "cd `ls`\n",
#                   "cp val_values.npy /local/scratch/albertim/output/npy/{}_{}.npy".format(model, dataset))
#     print("cd /local/scratch/albertim/output \n")

    # # Select all the numpy matrices saved there
    # file_names = [f for f in os.listdir(args.input_folder) if os.path.isfile(os.path.join(args.input_folder, f)) and ".npy" in f]

    # image = plot_mean_std(x = (np.arange(args.epochs + 1) - 1),
    # arr = np.roll(val_scores[:i + 1], axis=1, shift=1),
    # suptitle = 'Multi-Run: Val',
    # title = 'Runs: {}'.format(i + 1),
    # xlabel = 'Epoch', ylabel = 'Score',
    # ylim = [0, 100.0])
    #
    #
    # # Ensuring the data passed as parameter is healthy
    # image = tensor_to_image(image)
    #
    # # Write image to output folder
    # dest_filename = os.path.join(output_folder, 'images', tag + '.png')
    # save_numpy_image(dest_filename, image)