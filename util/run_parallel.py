import os
import sys
from multiprocessing import Process, Queue
import torch
from sigopt import Connection

#SIGOPT_TOKEN = "YEQGRJZHNJMNHHZTDJIQKOXILQCSHZVFWWJIIWYNSWKQPGOA"  # production
SIGOPT_TOKEN = "UQOOVYGGZNNDDFUAQQCCGMVNLVATTXDFKTXFXWIYUGRMJQHW"  # dev

EXPERIMENT_NAME_PREFIX = "dev"
LOG_FOLDER = "output"
# LOG_FOLDER_LONG = "log"
NUMBER_EPOCHS = 30
RUNS_PER_MODEL = 1
PROCESSES_PER_GPU = 4

MODELS = [
    "BaselineConv",
    "BaselineRND",
    "DCTFirst",
    "DCTFirst_Fixed",
    "FFTFirst",
    "FFTFirst_Fixed",
    "DCTBidir",
    "DCTBidir_Fixed",
    "FFTBidir",
    "FFTBidir_Fixed",
]

DATASETS = [
    "/net/dataset/albertim/ColorectalHist",
]

##########################################################################
# Creating Experiments
##########################################################################
class Experiment(object):
    def __init__(self,
                 experiment_name_prefix,
                 model_name,
                 output_folder,
                 dataset_folder,
                 number_epochs,
                 additional,
                 gpu_index=None,):
        self.experiment_name_prefix = experiment_name_prefix
        self.model_name = model_name
        self.output_folder = output_folder
        self.dataset_folder = dataset_folder
        self.number_epochs = number_epochs
        self.additional = additional
        self.gpu_index = gpu_index

    def get_cmd(self):

        cmd = "python template/RunMe.py --ignoregit --disable-dataset-integrity " \
              "--experiment-name {EXPERIMENT_NAME:s} " \
              "--model {MODEL:s} " \
              "--output-folder {OUTPUT_FOLDER:s} " \
              "--dataset-folder {DATASET_FOLDER:s} " \
              "--epoch {NUMBER_EPOCHS:d} ".format(
            EXPERIMENT_NAME=self.experiment_name_prefix + '_' + self.model_name + '_' + self.dataset_folder,
            MODEL=self.model_name,
            OUTPUT_FOLDER=self.output_folder,
            DATASET_FOLDER=self.dataset_folder,
            NUMBER_EPOCHS=self.number_epochs)

        if self.gpu_index is not None:
            cmd = cmd + " --gpu-id {GPU_ID:d} ".format(GPU_ID=self.gpu_index)

        if self.additional:
            cmd = cmd + self.additional

        return cmd

    def __repr__(self):
        return self.get_cmd()


class ExperimentsBuilder(object):

    @staticmethod
    def build_sigopt_combinations(model_list, dataset_folders_list, experiment_name_prefix, output_folder, epochs):
        experiments = []
        for model in model_list:
            for dataset in dataset_folders_list:
                experiment = Experiment(experiment_name_prefix, model, output_folder, dataset, epochs,
                                        "--momentum 0.9 "
                                        "--sig-opt-token {SIGOPT_TOKEN:s} "
                                        "--sig-opt-runs {RUNS_PER_MODEL:s} "
                                        "--sig-opt spectralSigOpt.txt ".format(
                                            SIGOPT_TOKEN=SIGOPT_TOKEN,
                                            RUNS_PER_MODEL=str(RUNS_PER_MODEL)))
                experiments.append(experiment)
        return experiments


    # @staticmethod
    # def build_longruns_combinations(model_list, dataset_folders_list, experiment_name_prefix, output_folder, epochs):
    #     experiments = []
    #     for model in model_list:
    #         for dataset in dataset_folders_list:
    #             best_parameters = ExperimentsBuilder._get_best_parameters(experiment_name_prefix + '_' + model + '_' + dataset)
    #
    #             experiment = Experiment("long_" + experiment_name_prefix, model, output_folder, dataset, epochs,
    #                                     "--momentum 0.9 " \
    #                                     "--lr {LR:f} " \
    #                                     "--weight-decay {WD:f}".format(
    #                                         LR=best_parameters["lr"],
    #                                         WD=best_parameters["weight_decay"])
    #                                     )
    #             experiments.append(experiment)
    #     return experiments

    @staticmethod
    def _retrieve_id_by_name(conn, name):
        experiment_list = conn.experiments().fetch()
        retrieved = []
        for n in experiment_list.data:
            if name in n.name:
                retrieved.append(n.id)
        return retrieved

    @staticmethod
    def _get_best_parameters(experiment_name):

        conn = Connection(client_token=SIGOPT_TOKEN)
        conn.set_api_url("https://api.sigopt.com")

        EXPERIMENT_ID = ExperimentsBuilder._retrieve_id_by_name(conn, experiment_name)

        if len(EXPERIMENT_ID) > 1:
            print("Experiments have duplicate names! Archive older ones before proceeding.")
            sys.exit(-1)
        if not EXPERIMENT_ID:
            print("Experiments not found")
            sys.exit(-1)

        EXPERIMENT_ID = EXPERIMENT_ID[0]
        return conn.experiments(EXPERIMENT_ID).best_assignments().fetch().data[0].assignments

##########################################################################
# Running Experiments
##########################################################################
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


def run_experiments(number_gpus, processes_per_gpu, queue):
    processes = []
    for i in range(number_gpus):
        for j in range(processes_per_gpu):
            process = ExperimentProcess(queue=queue, gpu_idx=i)
            process.start()
            processes.append(process)
    [process.join() for process in processes]


if __name__ == '__main__':
    # Init parameter
    NUM_GPUs = torch.cuda.device_count()

    # Init queue item
    queue = Queue()

    print("started...")

    experiments = ExperimentsBuilder.build_sigopt_combinations(
        MODELS, DATASETS, EXPERIMENT_NAME_PREFIX, LOG_FOLDER, NUMBER_EPOCHS,
    )
    [queue.put(e) for e in experiments]
    run_experiments(NUM_GPUs, PROCESSES_PER_GPU, queue)

    # print("...begin phase 2...")
    #
    # experiments = ExperimentsBuilder.build_longruns_combinations(
    #     MODELS, DATASETS, EXPERIMENT_NAME_PREFIX, LOG_FOLDER_LONG, NUMBER_EPOCHS_LONG,
    # )
    # [queue.put(e) for e in experiments]
    # run_experiments(NUM_GPUs, PROCESSES_PER_GPU, queue)

    print("...finished!")
