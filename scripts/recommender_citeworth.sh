#!/bin/bash

DATASET=cite
DATASET_YEAR=2014
OUTPUT_PREFIX=/media/nvme2n1/project_struct_cite/aae

THRES=55
mkdir -p $OUTPUT_PREFIX
for RUN in 1 2 3
do
    for DROP in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8
    do
        echo python3 main.py $DATASET_YEAR -d $DATASET -m $THRES -dr $DROP -o $OUTPUT_PREFIX/$DATASET-$DATASET_YEAR-$THRESHOLD-$RUN-$DROP.txt
        CUDA_VISIBLE_DEVICES=3 python3 main.py $DATASET_YEAR -d $DATASET -m $THRES -dr $DROP -o $OUTPUT_PREFIX/$DATASET-$DATASET_YEAR-$THRESHOLD-$RUN-$DROP-cond.txt --conditioned_autoencoders
    done
done
exit 0
