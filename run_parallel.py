import os
import sys
from multiprocessing import Process, Queue
import torch
from sigopt import Connection
import numpy as np

SIGOPT_TOKEN = "YEQGRJZHNJMNHHZTDJIQKOXILQCSHZVFWWJIIWYNSWKQPGOA"  # production
#SIGOPT_TOKEN = "UQOOVYGGZNNDDFUAQQCCGMVNLVATTXDFKTXFXWIYUGRMJQHW"  # dev

EXPERIMENT_NAME_PREFIX = "variance"
LOG_FOLDER = "/local/scratch/albertim/output/rerun"
NUMBER_EPOCHS = 60 # For CB55 is /5
RUNS_PER_MODEL = 20
PROCESSES_PER_GPU = 1

MODELS = [
    "BaselineDeep",
    #"BaselineConv",
    #"RNDFirst",
    "RNDBidir",
    #"DCTFirst",
    #"DCTFirst_Fixed",
    #"FFTFirst",
    #"FFTFirst_Fixed",
    #"DCTBidir",
    #"DCTBidir_Fixed",
    #"FFTBidir",
    #"FFTBidir_Fixed",
    #"resnet18",
    #"alexnet",
]

DATASETS = [
    "/local/scratch/ColorectalHist",
    #"/local/scratch/CB55",
    # #"/local/scratch/CSG18",
    # #"/local/scratch/CSG863",
    # #"/local/scratch/ImageNet",
    #"/local/scratch/Flowers",
    #"/local/scratch/HAM10000",
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
            NUMBER_EPOCHS=int(self.number_epochs/5) if "CB55" in self.dataset_folder else self.number_epochs)

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
        for dataset in dataset_folders_list:
            for model in model_list:
                experiment = Experiment(experiment_name_prefix, model, output_folder, dataset, epochs,
                                        "--momentum 0.9 "
                                        "--batch-size 128 "
                                        "-j {WORKERS:d} "
                                        "--sig-opt-token {SIGOPT_TOKEN:s} "
                                        "--sig-opt-runs {RUNS_PER_MODEL:s} "
                                        "--sig-opt-project {SIGOPT_PROJECT:s} "
                                        "--sig-opt spectralSigOpt.txt ".format(
                                            WORKERS=int(np.floor(64/np.min(
                                                [(torch.cuda.device_count()*PROCESSES_PER_GPU),
                                                 len(DATASETS)*len(MODELS)]))),
                                            SIGOPT_TOKEN=SIGOPT_TOKEN,
                                            RUNS_PER_MODEL=str(RUNS_PER_MODEL),
                                            SIGOPT_PROJECT="_spectral"))
                experiments.append(experiment)
        return experiments


    @staticmethod
    def build_variance_combinations(model_list, dataset_folders_list, experiment_name_prefix, output_folder, epochs):
        conn = Connection(client_token=SIGOPT_TOKEN)
        conn.set_api_url("https://api.sigopt.com")

        # Fetch all experiments
        sigopt_list = []
        for experiment in conn.experiments().fetch().iterate_pages():
            sigopt_list.append(experiment)

        experiments = []
        for dataset in dataset_folders_list:
            for model in model_list:
                best_parameters = ExperimentsBuilder._get_best_parameters(conn, sigopt_list, [model, dataset])

                experiments.append(Experiment("multi_" + experiment_name_prefix, model, output_folder, dataset, epochs,
                                              "--momentum 0.9 "
                                              "--batch-size 128 "
                                              "-j {WORKERS:d} "
                                              "--multi-run {MULTI_RUN:d} "
                                              "--lr {LR:f} "
                                              "--weight-decay {WD:f}".format(
                                                  MULTI_RUN=RUNS_PER_MODEL,
                                                  WORKERS=int(np.floor(64 / np.min(
                                                      [(torch.cuda.device_count() * PROCESSES_PER_GPU),
                                                        len(DATASETS) * len(MODELS)]))),
                                                  LR=best_parameters["lr"],
                                                  WD=best_parameters["weight_decay"])
                                              ))
        return experiments

    @staticmethod
    def _retrieve_id_by_name(sigopt_list, parts):
        retrieved = []
        for experiment in sigopt_list:
            if all(p in experiment.name for p in parts):
                retrieved.append(experiment.id)
        return retrieved

    @staticmethod
    def _get_best_parameters(conn, sigopt_list, parts):

        EXPERIMENT_ID = ExperimentsBuilder._retrieve_id_by_name(sigopt_list, parts)

        if not EXPERIMENT_ID:
            print("Experiments not found")
            sys.exit(-1)

        # Select the experiments with the highest score
        scores = [conn.experiments(ID).best_assignments().fetch().data[0].value for ID in EXPERIMENT_ID]
        EXPERIMENT_ID = EXPERIMENT_ID[scores.index(max(scores))]

        # Return the assignments
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

    print("sigopt...")
    experiments = []
    # experiments.extend(ExperimentsBuilder.build_sigopt_combinations(
    #     MODELS, DATASETS, EXPERIMENT_NAME_PREFIX, LOG_FOLDER, NUMBER_EPOCHS,
    # ))
    # experiments.extend(ExperimentsBuilder.build_sigopt_combinations(
    #      ["BaselineDeep"], ["/local/scratch/ColorectalHist"], EXPERIMENT_NAME_PREFIX, LOG_FOLDER, NUMBER_EPOCHS,
    # ))
    # experiments.extend(ExperimentsBuilder.build_sigopt_combinations(
    #     ["RNDBidir"],["/local/scratch/Flowers"], EXPERIMENT_NAME_PREFIX, LOG_FOLDER, NUMBER_EPOCHS,
    # ))
    # [queue.put(e) for e in experiments]
    # run_experiments(NUM_GPUs, PROCESSES_PER_GPU, queue)

    print("variance...")
    experiments = []

    #experiments.extend(ExperimentsBuilder.build_variance_combinations(
    #    MODELS, DATASETS, EXPERIMENT_NAME_PREFIX, LOG_FOLDER, NUMBER_EPOCHS,
    #))
    experiments.extend(ExperimentsBuilder.build_variance_combinations(
          ["BaselineDeep"], ["/local/scratch/ColorectalHist"], EXPERIMENT_NAME_PREFIX, LOG_FOLDER, NUMBER_EPOCHS,
    ))
    experiments.extend(ExperimentsBuilder.build_variance_combinations(
         ["RNDBidir"],["/local/scratch/Flowers"], EXPERIMENT_NAME_PREFIX, LOG_FOLDER, NUMBER_EPOCHS,
    ))
    # experiments.extend(ExperimentsBuilder.build_variance_combinations(
    #     ["FFTBidir"],["/local/scratch/HAM10000"], EXPERIMENT_NAME_PREFIX, LOG_FOLDER, NUMBER_EPOCHS,
    # ))
    [queue.put(e) for e in experiments]
    run_experiments(NUM_GPUs, PROCESSES_PER_GPU, queue)

    print("...finished!")
