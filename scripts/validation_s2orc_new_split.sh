#!/bin/bash

DATASET=cite5
DATASET_YEAR=2019
END_YEAR=2020 # we want to ignore the last year

OUTPUT_PREFIX=/media/nvme2n1/project_struct_cite/aae/val_s2orc_new

#THRES=1
mkdir -p $OUTPUT_PREFIX
for RUN in 1 2 3
do
    for THRES in 2 3 4 5 #10 15 20 25
    do
        for DROP in 0.2 0.5 0.8
        do
            echo python3 main.py $DATASET_YEAR -d $DATASET -m $THRES -dr $DROP -o $OUTPUT_PREFIX/$DATASET-$DATASET_YEAR-$THRES-$RUN-$DROP.txt
            CUDA_VISIBLE_DEVICES=2 python3 main.py $DATASET_YEAR --end $END_YEAR -d $DATASET -m $THRES -dr $DROP -o $OUTPUT_PREFIX/cite5-$DATASET_YEAR-$THRES-$RUN-$DROP-cond.txt --conditioned_autoencoders --eval_each 
        done
    done
done

for RUN in 1 2 3
do
    for THRES in 2 3 4 5
    do
        for DROP in 0.2 0.5 0.8
        do
            echo python3 main.py $DATASET_YEAR -d $DATASET -m $THES -dr $DROP -o $OUTPUT_PREFIX/$DATASET-$DATASET_YEAR-$THRES-$RUN-$DROP-section.txt
            CUDA_VISIBLE_DEVICES=2 python3 main.py $DATASET_YEAR --end $END_YEAR -d $DATASET -m $THRES -dr $DROP -o $OUTPUT_PREFIX/cite5-$DATASET_YEAR-$THRES-$RUN-$DROP-section.txt --conditioned_autoencoder --use_section --use_sdict --eval_each
        done
    done
done
exit 0
