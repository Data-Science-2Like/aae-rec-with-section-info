#!/bin/bash

DATASET=aan
DATASET_YEAR=2014
OUTPUT_PREFIX=/media/nvme2n1/project_struct_cite/aae

DROP=0.5
THRES=5
mkdir -p $OUTPUT_PREFIX
for CODE in 20 30 40 50 60 70 80
do
    for HIDDEN in 180 200 220 240 #40 60 80 100 120 140 160
    do
        echo code: $CODE hidden: $HIDDEN
        echo code: $CODE hidden: $HIDDEN >> test.txt
        CUDA_VISIBLE_DEVICES=3 python3 main.py $DATASET_YEAR -d $DATASET -m $THRES -dr $DROP --code $CODE --hidden $HIDDEN -o $OUTPUT_PREFIX/$DATASET-$DATASET_YEAR-$THRES-$RUN-$DROP-cond.txt --conditioned_autoencoders
    done
done
exit 0
