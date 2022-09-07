#!/bin/bash

DATASET=cite7
DATASET_YEAR=2019

OUTPUT_PREFIX=/media/nvme2n1/project_struct_cite/aae/prefetcher_outputs_lists_final
GPU=2

#THRES=1
mkdir -p $OUTPUT_PREFIX


for RUN in 1 #2 3
do
    for THRES in 1 2
    do
        for DROP in 0.2 0.5 0.8
        do
            echo python3 main2.py $DATASET_YEAR -d $DATASET -m $THES -dr $DROP -o $OUTPUT_PREFIX/$DATASET-$DATASET_YEAR-$THRES-$RUN-$DROP-none.txt
            CUDA_VISIBLE_DEVICES=$GPU python3 main2.py $DATASET_YEAR -d $DATASET -m $THRES -dr $DROP -o $OUTPUT_PREFIX/$DATASET-$DATASET_YEAR-$THRES-$RUN-$DROP-none.txt --autoencoder
        done
    done
done
exit 0
