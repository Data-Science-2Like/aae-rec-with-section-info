#!/bin/bash

DATASET=dblp
DATASET_YEAR=2014
OUTPUT_PREFIX=/media/nvme2n1/project_struct_cite/aae

THRES=55
mkdir -p $OUTPUT_PREFIX
for RUN in 1 #2 3
do
    for DROP in 0.5 #0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8
    do
        echo python3 main.py $DATASET_YEAR -d $DATASET -m $THRES -dr $DROP -o $OUTPUT_PREFIX/$DATASET-$DATASET_YEAR-$THRES-$RUN-$DROP-baseline.txt
        CUDA_VISIBLE_DEVICES=2 python3 main.py $DATASET_YEAR -d $DATASET -m $THRES -dr $DROP -o $OUTPUT_PREFIX/$DATASET-$DATASET_YEAR-$THRES-$RUN-$DROP-baseline.txt --baseline
    done
done

