#!/usr/bin/env bash


EXPERIMENT_NAME="SpectralRun4"
OPTIMIZER="Adam"
LR="0.001"
MULTI_RUN=10
BATCH_SIZE=64
EPOCHS=10

declare -a ALL_MODELS=("Fourier_32x32_Fixed"
                       "HybridFourier_32x32_Fixed"
                       "CosineBidirectional_150x150_Unfixed"
                       "Cosine_150x150_Unfixed"
                       "HybridCosineBidirectional_150x150_Fixed"
                       "FourierBidirectional_32x32_Fixed"
                       "Cosine_32x32_Unfixed"
                        "HybridCosine_150x150_Unfixed"
                        "Cosine_150x150_Fixed"
                        "DctIIRandomWeights_32x32"
                        "HybridFirstCosine_150x150_Fixed"
                        "HybridFourier_150x150_Unfixed"
                        "FirstCosine_150x150_Fixed"
                        "FirstFourier_32x32_Unfixed"
                        "DctIIRandomWeights_150x150"
                        "HybridFirstCosine_150x150_Unfixed"
                        "FirstCosine_150x150_Unfixed"
                        "HybridCosineBidirectional_32x32_Unfixed"
                        "HybridCosineBidirectional_32x32_Fixed"
                        "Fourier_150x150_Fixed"
                        "FirstFourier_32x32_Fixed"
                        "FourierBidirectional_32x32_Unfixed"
                        "HybridFirstFourier_150x150_Fixed"
                        "CosineBidirectional_32x32_Unfixed"
                        "HybridCosine_32x32_Unfixed"
                        "HybridFirstFourier_32x32_Unfixed"
                        "Fourier_150x150_Unfixed"
                        "CosineBidirectional_150x150_Fixed"
                        "Cosine_32x32_Fixed"
                        "HybridCosine_150x150_Fixed"
                        "HybridCosineBidirectional_150x150_Unfixed"
                        "FirstCosine_32x32_Unfixed"
                        "CosineBidirectional_32x32_Fixed"
                        "HybridCosine_32x32_Fixed"
                        "HybridFirstFourier_150x150_Unfixed"
                        "FourierBidirectional_150x150_Fixed"
                        "Fourier_32x32_Unfixed"
                        "HybridFirstCosine_32x32_Unfixed"
                        "PureConv_32x32"
                        "HybridFourier_32x32_Unfixed"
                        "HybridFirstCosine_32x32_Fixed"
                        "FourierBidirectional_150x150_Unfixed"
                        "FirstCosine_32x32_Fixed"
                        "PureConv_150x150"
                        "HybridFirstFourier_32x32_Fixed"
                        "FirstFourier_150x150_Fixed"
                        "HybridFourier_150x150_Fixed"
                        "FirstFourier_150x150_Unfixed")

declare -a MODELS_32=("Fourier_32x32_Fixed"
                       "HybridFourier_32x32_Fixed"
                       "FourierBidirectional_32x32_Fixed"
                       "Cosine_32x32_Unfixed"
                       "DctIIRandomWeights_32x32"
                       "FirstFourier_32x32_Unfixed"
                       "HybridCosineBidirectional_32x32_Unfixed"
                       "HybridCosineBidirectional_32x32_Fixed"
                       "FirstFourier_32x32_Fixed"
                       "FourierBidirectional_32x32_Unfixed"
                       "CosineBidirectional_32x32_Unfixed"
                        "HybridCosine_32x32_Unfixed"
                        "HybridFirstFourier_32x32_Unfixed"
                        "Cosine_32x32_Fixed"
                        "FirstCosine_32x32_Unfixed"
                        "CosineBidirectional_32x32_Fixed"
                        "HybridCosine_32x32_Fixed"
                        "Fourier_32x32_Unfixed"
                        "HybridFirstCosine_32x32_Unfixed"
                        "PureConv_32x32"
                        "HybridFourier_32x32_Unfixed"
                        "HybridFirstCosine_32x32_Fixed"
                        "FirstCosine_32x32_Fixed"
                        "HybridFirstFourier_32x32_Fixed")

declare -a DATASET_FOLDER_CLASSIC=("/media/blob/MNIST"
                                   "/media/blob/CIFAR10"
                                   "/media/blob/FashionMNIST")

# Run all models on the hist dataset
for i in "${ALL_MODELS[@]}"
do
    python template/RunMe.py --model "$i" --output-folder log --dataset-folder /media/blob/ColorectalHist/ --lr "$LR" --ignoregit --output-channels 8 --optimizer-name "$OPTIMIZER" --epoch "$EPOCHS" --batch "$BATCH_SIZE" --experiment-name "$EXPERIMENT_NAME" --multi-run "$MULTI_RUN"
done

# Run the small models on the classic 32x32 datasets
for i in "${MODELS_32[@]}"
do
    for j in "${DATASET_FOLDER_CLASSIC[@]}"
    do
        python template/RunMe.py --model "$i" --output-folder log --dataset-folder "$j" --lr "$LR" --ignoregit --output-channels 10 --optimizer-name "$OPTIMIZER" --epoch "$EPOCHS" --batch "$BATCH_SIZE" --experiment-name "$EXPERIMENT_NAME" --multi-run "$MULTI_RUN"
    done
done
