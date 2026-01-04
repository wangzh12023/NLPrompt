#!/bin/bash

# GCE_NLPrompt training script
# Usage: bash scripts/gcenlprompt/main.sh <DATASET> <SHOTS> <RATE> <TYPE> <CLASS>

# custom config
DATA=/public/home/SI251/zhongzhch2023-si251/NLPrompt/data/
TRAINER=GCE_NLPrompt

DATASET=$1
CFG=rn50  # config file
SHOTS=$2  # number of shots (1, 2, 4, 8, 16)
RATE=$3
TYPE=$4
CLASS=$5

for SEED in 1
do
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/noise_${TYPE}_${RATE}/seed${SEED}
    mkdir -p ${DIR}
    # Check if training is complete by looking for the best model file
    if [ -d "$DIR" ] && [ -f "$DIR/prompt_learner/model-best.pth.tar" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job and save the output to ${DIR}"
        python -u train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/GCE_NLPROMPT/${CFG}.yaml \
        --output-dir ${DIR} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.NOISE_RATE ${RATE} \
        DATASET.NOISE_TYPE ${TYPE} \
        DATASET.num_class ${CLASS} 2>&1
    fi
done

