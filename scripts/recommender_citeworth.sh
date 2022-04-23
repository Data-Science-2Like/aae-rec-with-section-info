#!/bin/bash

DATASET=cite2
DATASET_YEAR=2019
VAL_YEAR=2018
OUTPUT_PREFIX=/media/nvme2n1/project_struct_cite/aae

THRES=5
mkdir -p $OUTPUT_PREFIX
for RUN in 1 #2 3
do
    for DROP in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8
    do
        echo python3 main.py $DATASET_YEAR --val $VAL_YEAR  -d $DATASET -m $THRES -dr $DROP -o $OUTPUT_PREFIX/$DATASET-$DATASET_YEAR-$THRES-$RUN-$DROP.txt
        CUDA_VISIBLE_DEVICES=2 python3 main.py $DATASET_YEAR --val $VAL_YEAR -d $DATASET -m $THRES -dr $DROP -o $OUTPUT_PREFIX/$DATASET-$DATASET_YEAR-$THRES-$RUN-$DROP-cond.txt --conditioned_autoencoders
    done
done

for RUN in 1 #2 3
do
    for DROP in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8
    do
        echo python3 main.py $DATASET_YEAR --val $VAL_YEAR -d $DATASET -m $THES -dr $DROP -o $OUTPUT_PREFIX/$DATASET-$DATASET_YEAR-$THRES-$RUN-$DROP-section.txt
        CUDA_VISIBLE_DEVICES=2 python3 main.py $DATASET_YEAR --val $VAL_YEAR -d $DATASET -m $THRES -dr $DROP -o $OUTPUT_PREFIX/$DATASET-$DATASET_YEAR-$THRES-$RUN-$DROP-section.txt --conditioned_autoencoder --use_section
    done
done
exit 0
