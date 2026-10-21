#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    echo 'Usage: DATA=/path/to/data bash scripts/base_to_new.sh [dataset|all] ["seed ..."]'
    echo 'Default: dtd, seed=1, attribute-dropout training followed by New AA-DPE evaluation.'
    echo 'Set SUBSAMPLE_CLASSES="base new" to evaluate both splits.'
    exit 0
fi
(( $# <= 2 )) || { echo 'Expected [dataset|all] ["seed ..."]' >&2; exit 2; }

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "${PROJECT_ROOT}"

export TAP_ATTR_BRANCH_DROPOUT_P=${ATTR_DROPOUT_P:-${TAP_ATTR_BRANCH_DROPOUT_P:-${ADPE_ATTR_BRANCH_DROPOUT_P:-0.15}}}
export TAP_ATTR_BRANCH_DROPOUT_RESCALE=${TAP_ATTR_BRANCH_DROPOUT_RESCALE:-${ADPE_ATTR_BRANCH_DROPOUT_RESCALE:-0}}
export TAP_ATTR_DROPOUT_FUSION_CE_WEIGHT=${FUSION_CE_WEIGHT:-${TAP_ATTR_DROPOUT_FUSION_CE_WEIGHT:-${ADPE_ATTR_DROPOUT_FUSION_CE_WEIGHT:-0.5}}}

DATA=${DATA:-/mnt/a/cs_ai_wys/lijia/Data}
TRAINER=ADPE
PYTHON_BIN=${PYTHON_BIN:-python}
CONFIG_FILE=${CONFIG_FILE:-configs/trainers/ADPE/ADPE.yaml}
OUTPUT_ROOT=${OUTPUT_ROOT:-output/base2new}
SHOTS=${SHOTS:-16}
TARGET=${1:-${DATASETS:-dtd}}
if [[ ${TARGET} == all ]]; then DATASETS=${DATASETS:-all}; else DATASETS=${TARGET}; fi
SEEDS=${2:-${SEEDS:-1}}

# dataset, quantile, text_lr, vision_lr, u3
DATASET_SETTINGS=(
    "dtd,1.0,0.002,0.006,3.0"
    "eurosat,1.0,0.004,0.004,1.5"
    "oxford_pets,1.0,0.004,0.004,1.5"
    "food101,1.0,0.004,0.004,1.5"
    "oxford_flowers,1.0,0.002,0.006,3.0"
    "stanford_cars,0.95,0.002,0.006,3.0"
    "fgvc_aircraft,1.0,0.004,0.004,1.5"
    "ucf101,0.95,0.002,0.006,3.0"
    "caltech101,1.0,0.002,0.006,3.0"
    "sun397,0.95,0.004,0.004,1.5"
    "imagenet,0.95,0.004,0.004,1.5"
)

vision_num_epochs=${VISION_EPOCHS:-50}
text_num_epochs=${TEXT_EPOCHS:-10}
bs=${TRAIN_BATCH_SIZE:-32}
OPTIM_NAME=${OPTIM_NAME:-adamw}
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-100}
SUBSAMPLE_CLASSES=${SUBSAMPLE_CLASSES:-new}
RUN_TRAIN=${RUN_TRAIN:-1}
SKIP_EXISTING_TRAIN=${SKIP_EXISTING_TRAIN:-1}

DPE_ENABLED=${DPE_ENABLED:-True}
DPE_SHOTS=${DPE_SHOTS:-3}
DPE_ALPHA=${DPE_ALPHA:-6.0}
DPE_BETA=${DPE_BETA:-5.0}
DPE_TEXT_LR=${DPE_TEXT_LR:-0.0006}
DPE_IMAGE_LR=${DPE_IMAGE_LR:-0.0006}
DPE_ALIGN=${DPE_ALIGN:-0.5}
DPE_ENTROPY_THRESHOLD=${DPE_ENTROPY_THRESHOLD:-0.1}
DPE_ATTR_AFFINITY_WEIGHT=${AA_WEIGHT:-${DPE_ATTR_AFFINITY_WEIGHT:-${ADPE_DPE_ATTR_AFFINITY_WEIGHT:-${TAP_DPE_ATTR_AFFINITY_WEIGHT:-0.4}}}}
export TAP_DPE_ATTR_AFFINITY_WEIGHT="${DPE_ATTR_AFFINITY_WEIGHT}"

for name in SHOTS vision_num_epochs text_num_epochs bs TEST_BATCH_SIZE; do
    [[ ${!name} =~ ^[1-9][0-9]*$ ]] || { echo "${name} must be a positive integer" >&2; exit 2; }
done
num_epochs=$((vision_num_epochs + text_num_epochs))
LOAD_EPOCH=${LOAD_EPOCH:-${num_epochs}}
[[ ${LOAD_EPOCH} =~ ^[1-9][0-9]*$ ]] || { echo 'LOAD_EPOCH must be a positive integer' >&2; exit 2; }
if [[ ${RUN_TRAIN} == 1 ]] && (( LOAD_EPOCH > num_epochs )); then
    echo "LOAD_EPOCH cannot exceed ${num_epochs}" >&2
    exit 2
fi
for name in RUN_TRAIN SKIP_EXISTING_TRAIN; do
    [[ ${!name} == 0 || ${!name} == 1 ]] || { echo "${name} must be 0 or 1" >&2; exit 2; }
done
read -r -a SEED_LIST <<< "${SEEDS}"
read -r -a DATASET_LIST <<< "${DATASETS}"
read -r -a SPLIT_LIST <<< "${SUBSAMPLE_CLASSES}"
(( ${#SEED_LIST[@]} && ${#DATASET_LIST[@]} && ${#SPLIT_LIST[@]} )) || exit 2
for seed in "${SEED_LIST[@]}"; do
    [[ ${seed} =~ ^(0|[1-9][0-9]*)$ ]] || { echo "Invalid seed: ${seed}" >&2; exit 2; }
done
for dataset in "${DATASET_LIST[@]}"; do
    found=0
    for setting in "${DATASET_SETTINGS[@]}"; do
        if [[ ${dataset} == all || ${setting%%,*} == "${dataset}" ]]; then found=1; break; fi
    done
    [[ ${found} == 1 ]] || { echo "Unsupported dataset: ${dataset}" >&2; exit 2; }
done
for split in "${SPLIT_LIST[@]}"; do
    [[ ${split} == base || ${split} == new || ${split} == all ]] || { echo "Invalid split: ${split}" >&2; exit 2; }
done

require_checkpoint() {
    local part
    for part in vision_prompt_learner text_prompt_learner; do
        [[ -f $1/${part}/model.pth.tar-$2 ]] || { echo "Missing checkpoint: $1/${part}/model.pth.tar-$2" >&2; exit 1; }
    done
}

for seed in "${SEED_LIST[@]}"; do
    for setting in "${DATASET_SETTINGS[@]}"; do
        IFS=',' read -r dataset quantile text_lr vision_lr u3 <<< "${setting}"
        if [[ " ${DATASETS} " != *" all "* && " ${DATASETS} " != *" ${dataset} "* ]]; then continue; fi

        run_path="${dataset}/shots_${SHOTS}/${TRAINER}/seed${seed}/${TAP_ATTR_BRANCH_DROPOUT_P}_${TAP_ATTR_DROPOUT_FUSION_CE_WEIGHT}"
        train_dir="${OUTPUT_ROOT}/train_base/${run_path}"
        args=(
            train.py --root "${DATA}" --seed "${seed}" --trainer "${TRAINER}"
            --dataset-config-file "configs/datasets/${dataset}.yaml"
            --config-file "${CONFIG_FILE}"
        )
        common_options=(
            OPTIM.VISION_MAX_EPOCH "${vision_num_epochs}" OPTIM.TEXT_MAX_EPOCH "${text_num_epochs}"
            OPTIM.NAME "${OPTIM_NAME}"
            OPTIM.VISION_LR "${VISION_LR:-${vision_lr}}" OPTIM.TEXT_LR "${TEXT_LR:-${text_lr}}"
            DATASET.NUM_SHOTS "${SHOTS}"
            DATALOADER.TRAIN_X.BATCH_SIZE "${bs}" DATALOADER.TEST.BATCH_SIZE "${TEST_BATCH_SIZE}"
            MODEL.ADPE.U3 "${U3:-${u3}}" MODEL.ADPE.MAX_LEN_QUANTILE "${MAX_LEN_QUANTILE:-${quantile}}"
            MODEL.ADPE.BRANCH_ABLATION.ENABLED False
        )

        if [[ ${RUN_TRAIN} == 1 ]]; then
            if [[ ${SKIP_EXISTING_TRAIN} == 1 && -f "${train_dir}/vision_prompt_learner/model.pth.tar-${num_epochs}" && -f "${train_dir}/text_prompt_learner/model.pth.tar-${num_epochs}" ]]; then
                echo "[SKIP] ${dataset} seed${seed}"
            else
                echo "[TRAIN] ${dataset} seed${seed}, dropout=${TAP_ATTR_BRANCH_DROPOUT_P}, fusion_ce=${TAP_ATTR_DROPOUT_FUSION_CE_WEIGHT}"
                "${PYTHON_BIN}" "${args[@]}" --output-dir "${train_dir}" \
                    "${common_options[@]}" DATASET.SUBSAMPLE_CLASSES base \
                    MODEL.ADPE.ATTR_BRANCH_DROPOUT_P "${TAP_ATTR_BRANCH_DROPOUT_P}" \
                    MODEL.ADPE.ATTR_DROPOUT_FUSION_CE_WEIGHT "${TAP_ATTR_DROPOUT_FUSION_CE_WEIGHT}" \
                    MODEL.ADPE.DPE.ENABLED False TEST.NO_TEST True
            fi
            require_checkpoint "${train_dir}" "${num_epochs}"
        fi

        require_checkpoint "${train_dir}" "${LOAD_EPOCH}"
        for split in "${SPLIT_LIST[@]}"; do
            test_dir="${OUTPUT_ROOT}/test_${split}_aa_dpe/${run_path}/rho_${DPE_ATTR_AFFINITY_WEIGHT}"
            mkdir -p -- "${test_dir}"
            echo "[EVAL] ${split} ${dataset} seed${seed}, epoch=${LOAD_EPOCH}, rho=${DPE_ATTR_AFFINITY_WEIGHT}"
            "${PYTHON_BIN}" "${args[@]}" --output-dir "${test_dir}" \
                --eval-only --model-dir "${train_dir}" --load-epoch "${LOAD_EPOCH}" \
                "${common_options[@]}" DATASET.SUBSAMPLE_CLASSES "${split}" \
                MODEL.ADPE.ATTR_BRANCH_DROPOUT_P 0.0 MODEL.ADPE.DPE.ENABLED "${DPE_ENABLED}" \
                MODEL.ADPE.DPE.SHOT_CAPACITY "${DPE_SHOTS}" \
                MODEL.ADPE.DPE.ALPHA "${DPE_ALPHA}" MODEL.ADPE.DPE.BETA "${DPE_BETA}" \
                MODEL.ADPE.DPE.TEXT_LR "${DPE_TEXT_LR}" MODEL.ADPE.DPE.IMAGE_LR "${DPE_IMAGE_LR}" \
                MODEL.ADPE.DPE.ALIGN "${DPE_ALIGN}" MODEL.ADPE.DPE.ENTROPY_THRESHOLD "${DPE_ENTROPY_THRESHOLD}" \
                MODEL.ADPE.DPE.ATTR_AFFINITY_WEIGHT "${DPE_ATTR_AFFINITY_WEIGHT}" \
                TEST.NO_TEST False TEST.SPLIT test 2>&1 | tee "${test_dir}/console.log"
        done
    done
done
