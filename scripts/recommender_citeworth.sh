#!/bin/bash

DATASET=cite5
DATASET_YEAR=2019
VAL_YEAR=2018
OUTPUT_PREFIX=/media/nvme2n1/project_struct_cite/aae/final_cite_sec

#THRES=1
mkdir -p $OUTPUT_PREFIX
for RUN in 1 2 3
do
    for THRES in 1 2 3 4 #5 10 15 20 25
    do
        for DROP in 0.2 0.5 0.8
        do
            echo python3 main.py $DATASET_YEAR -d $DATASET -m $THRES -dr $DROP -o $OUTPUT_PREFIX/$DATASET-$DATASET_YEAR-$THRES-$RUN-$DROP.txt
            CUDA_VISIBLE_DEVICES=1 python3 main.py $DATASET_YEAR --val $VAL_YEAR -d $DATASET -m $THRES -dr $DROP -o $OUTPUT_PREFIX/cite4-$DATASET_YEAR-$THRES-$RUN-$DROP-cond.txt --conditioned_autoencoders
        done
    done
done

for RUN in 1 2 3
do
    for THRES in 5 10 15 20 25
    do
        for DROP in 0.2 0.5 0.8
        do
            echo python3 main.py $DATASET_YEAR -d $DATASET -m $THES -dr $DROP -o $OUTPUT_PREFIX/$DATASET-$DATASET_YEAR-$THRES-$RUN-$DROP-section.txt
            CUDA_VISIBLE_DEVICES=1 python3 main.py $DATASET_YEAR --val $VAL_YEAR -d $DATASET -m $THRES -dr $DROP -o $OUTPUT_PREFIX/cite4-$DATASET_YEAR-$THRES-$RUN-$DROP-section.txt --conditioned_autoencoder --use_section
        done
    done
done
exit 0
