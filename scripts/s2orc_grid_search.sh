#!/bin/bash

DATASET=cite5
TEST_YEAR=2019
VAL_YEAR=2018

OUTPUT_PREFIX=/media/nvme2n1/project_struct_cite/aae/grid_search

#THRES=1
mkdir -p $OUTPUT_PREFIX
for RUN in 1 #2 3
do
    for THRES in 25 #5 #3 4 5 #10 15 20 25
    do
        for DROP in 0.2 0.5 0.8
        do
            CUDA_VISIBLE_DEVICES=2 python3 main.py $VAL_YEAR --val $VAL_YEAR --end $TEST_YEAR -d $DATASET -m $THRES -dr $DROP -o $OUTPUT_PREFIX/cite5-$DATASET_YEAR-$THRES-$RUN-$DROP-grid.txt --conditioned_autoencoders
        done
    done
done

