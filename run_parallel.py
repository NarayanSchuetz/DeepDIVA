"""
 Created by Narayan Schuetz at 09/01/2019
 University of Bern
 
 This file is subject to the terms and conditions defined in
 file 'LICENSE.txt', which is part of this source code package.
"""

import os
from multiprocessing import Process, Queue
import torch

EXPERIMENT_NAME = "ICML"
LOG_FOLDER = "log"
NUMBER_EPOCHS = 600
PROCESSES_PER_GPU = 2

"""
MODELS_150 = ["CosineBidirectional_150x150_Unfixed",
              "Cosine_150x150_Unfixed",
              "HybridCosineBidirectional_150x150_Fixed",
              "HybridCosine_150x150_Unfixed",
              "Cosine_150x150_Fixed",
              "HybridFirstCosine_150x150_Fixed",
              "HybridFourier_150x150_Unfixed",
              "FirstCosine_150x150_Fixed",
              "DctIIRandomWeights_150x150",
              "HybridFirstCosine_150x150_Unfixed",
              "FirstCosine_150x150_Unfixed",
              "Fourier_150x150_Fixed",
              "HybridFirstFourier_150x150_Fixed",
              "Fourier_150x150_Unfixed",
              "CosineBidirectional_150x150_Fixed",
              "HybridCosine_150x150_Fixed",
              "HybridCosineBidirectional_150x150_Unfixed",
              "HybridFirstFourier_150x150_Unfixed",
              "FourierBidirectional_150x150_Fixed",
              "FourierBidirectional_150x150_Unfixed",
              "PureConv_150x150",
              "FirstFourier_150x150_Fixed",
              "HybridFourier_150x150_Fixed",
              "FirstFourier_150x150_Unfixed"]
"""

MODELS_32 = ["HybridCosineBidirectional_32x32_Fixed",
             "HybridCosineBidirectional_32x32_Unfixed",
             "HybridFourierBidirectional_32x32_Fixed",
             "HybridFourierBidirectional_32x32_Unfixed",
             "HybridFirstCosine_32x32_Fixed",
             "HybridFirstCosine_32x32_Unfixed",
             "HybridFirstFourier_32x32_Fixed",
             "HybridFirstFourier_32x32_Unfixed",
             "CosineBidirectional_32x32_Fixed",
             "CosineBidirectional_32x32_Unfixed",
             "FourierBidirectional_32x32_Fixed",
             "FourierBidirectional_32x32_Unfixed",
             "FirstCosine_32x32_Fixed",
             "FirstFourier_32x32_Unfixed",
             "PureConv_32x32"]

DATASETS_32 = ["toy_dataset/CIFAR10",
               "toy_dataset/FashionMNIST"]

DATASETS_150 = ["toy_dataset/ColorectalHist"]


class Experiment(object):

    def __init__(self,
                 experiment_name,
                 model_name,
                 output_folder,
                 dataset_folder,
                 number_epochs,
                 gpu_index=None,
                 number_repetitions=1):
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.output_folder = output_folder
        self.dataset_folder = dataset_folder
        self.number_epochs = number_epochs
        self.gpu_index = gpu_index
        self.number_repetitions = number_repetitions

    def get_cmd(self):

        cmd = "python template/RunMe.py " \
                   "--experiment-name {EXPERIMENT_NAME:s} " \
                   "--model {MODEL:s} " \
                   "--output-folder {OUTPUT_FOLDER:s} " \
                   "--dataset-folder {DATASET_FOLDER:s} " \
                   "--ignoregit " \
                   "--epoch {NUMBER_EPOCHS:d} " \
                   "--sig-opt-token ZSPFRNSZRKKOREEETGGDQXEAEQLBZJKEZOCGDAFHZPQEVNHT " \
                   "--sig-opt-runs 50 " \
                   "--sig-opt spectralSigOpt.txt".format(
                EXPERIMENT_NAME=self.experiment_name+'_'+self.model_name+'_'+self.dataset_folder,
                MODEL=self.model_name,
                OUTPUT_FOLDER=self.output_folder,
                DATASET_FOLDER=self.dataset_folder,
                NUMBER_EPOCHS=self.number_epochs)

        if self.gpu_index is not None:
            cmd = cmd + " --gpu-id {GPU_ID:d} ".format(GPU_ID=self.gpu_index)

        return cmd

    def __repr__(self):
        return self.get_cmd()

class ExperimentsBuilder(object):

    @staticmethod
    def build_model_dataset_combinations(model_list, dataset_folders_list, experiment_name, output_folder,
                                         learning_rate, epochs, number_repetitions=1):
        experiments = []
        for model in model_list:
            for dataset in dataset_folders_list:
                experiment = Experiment(experiment_name, model, output_folder, dataset, learning_rate, epochs,
                                        number_repetitions=number_repetitions)
                experiments.append(experiment)
        return experiments


class ExperimentProcess(Process):
    def __init__(self, queue, gpu_idx):
        super().__init__()
        self.gpu_index = gpu_idx
        self.queue = queue

    def run(self):
        while not self.queue.empty():
            experiment = self.queue.get()
            experiment.gpu_index = self.gpu_index
            os.system(experiment.get_cmd())

if __name__ == '__main__':

    print("started...")

    # add experiments with 32x32 input
    experiments = ExperimentsBuilder.build_model_dataset_combinations(
        MODELS_32, DATASETS_32, EXPERIMENT_NAME, LOG_FOLDER, NUMBER_EPOCHS,
    )

    # add experiments with 150x150 inputs
    # experiments.extend(ExperimentsBuilder.build_model_dataset_combinations(
    #     MODELS_150, DATASETS_150, EXPERIMENT_NAME, LOG_FOLDER, LEARNING_RATE, OPTIMIZER, NUMBER_EPOCHS, BATCH_SIZE,
    #     NUMBER_REPETITIONS
    # ))
    #

    number_gpus = torch.cuda.device_count()

    queue = Queue()
    [queue.put(e) for e in experiments]

    processes = []
    for i in range(number_gpus):
        for j in range(PROCESSES_PER_GPU):
            process = ExperimentProcess(queue=queue, gpu_idx=i)
            process.start()
            processes.append(process)

    [process.join() for process in processes]

    print("...finished!")
