#!/bin/bash

for dataset in gtzan_noSeperation_LibrosaCPUSTFT gtzan_SpleeterGPUPreprocessor_2Stems_keepOriginal_LibrosaCPUSTFT gtzan_SpleeterGPUPreprocessor_2Stems_LibrosaCPUSTFT gtzan_SpleeterGPUPreprocessor_4Stems_keepOriginal_LibrosaCPUSTFT gtzan_SpleeterGPUPreprocessor_4Stems_LibrosaCPUSTFT
do  
    echo "Dataset: $dataset"
    for i in {1..5}
    do
        echo "Run: $i"
        docker run -it --rm --gpus all -v $PWD:/tmp -w /tmp -v '/home/steffen/Datasets:/data' techisland/cuda-conda-tensorflow-gpu:v1 python ./src/__main__.py $dataset
    done
done


