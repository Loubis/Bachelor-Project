#!/bin/bash

for dataset in fma/fma_medium_AudioSplit_in_3_SpleeterGPUPreprocessor_spleeter:4stems_keepOriginal_LibrosaCPUSTFT
do
    echo "Dataset: $dataset"
    for i in {1..4}
    do
        echo "Run: $i"
        python ./src/__main__.py $dataset
    done
done
