#!/bin/bash

DATASET=cite5
DATASET_YEAR=2019

OUTPUT_PREFIX=/media/nvme2n1/project_struct_cite/aae/test_s2orc_old

#THRES=1
mkdir -p $OUTPUT_PREFIX
for RUN in 1 #2 3
do
    for THRES in 2 5 #3 4 5 #10 15 20 25
    do
        for DROP in 0.2 0.5 0.8
        do
            echo python3 main.py $DATASET_YEAR -d $DATASET -m $THRES -dr $DROP -o $OUTPUT_PREFIX/$DATASET-$DATASET_YEAR-$THRES-$RUN-$DROP.txt
            CUDA_VISIBLE_DEVICES=2 python3 main.py $DATASET_YEAR -d $DATASET -m $THRES -dr $DROP -o $OUTPUT_PREFIX/cite5-$DATASET_YEAR-$THRES-$RUN-$DROP-cond.txt --conditioned_autoencoders
        done
    done
done

for RUN in 1 #2 3
do
    for THRES in 2 5 #3 4 5
    do
        for DROP in 0.2 0.5 0.8
        do
            echo python3 main.py $DATASET_YEAR -d $DATASET -m $THES -dr $DROP -o $OUTPUT_PREFIX/$DATASET-$DATASET_YEAR-$THRES-$RUN-$DROP-section.txt
            CUDA_VISIBLE_DEVICES=2 python3 main.py $DATASET_YEAR -d $DATASET -m $THRES -dr $DROP -o $OUTPUT_PREFIX/cite5-$DATASET_YEAR-$THRES-$RUN-$DROP-section.txt --conditioned_autoencoder --use_section --use_sdict
        done
    done
done
exit 0
