#!/usr/bin/env bash
# ============================================================
# StateMoE 로버스트니스 평가
# clean -> GPU 1
# noisy -> GPU 2, missing -> GPU 3, mixed -> GPU 4
# ============================================================
set -euo pipefail

GPU_ID="${1:-1}"
DATASET_DIR="${2:-/dataset/clothing}"
COND_TYPES_RAW="${3:-mixed missing noisy}"
RATIOS="20 40 60 80"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
LOG_DIR="${SRC_DIR}/log/clothing"
mkdir -p "${LOG_DIR}"

DATASET_NAME="$(basename "${DATASET_DIR}")"
DATA_PATH="$(dirname "${DATASET_DIR}")/"

MODEL_NAME="StateMoE"
MODEL_CONFIG_NAME="${MODEL_CONFIG_NAME:-StateMoE_clothing}"
CLEAN_IMG_FILE="${CLEAN_IMG_FILE:-image_feat.npy}"
CLEAN_TXT_FILE="${CLEAN_TXT_FILE:-text_feat.npy}"
SUMMARY_FILE="${LOG_DIR}/${MODEL_NAME}-${DATASET_NAME}-summary-$(date +%Y%m%d_%H%M%S).txt"
RESULT_TMP_DIR="${LOG_DIR}/.tmp_${MODEL_NAME}_$$"
mkdir -p "${RESULT_TMP_DIR}"

NOISY_GPU="${NOISY_GPU:-2}"
MISSING_GPU="${MISSING_GPU:-3}"
MIXED_GPU="${MIXED_GPU:-4}"

cleanup() {
    rm -rf "${RESULT_TMP_DIR}"
}
trap cleanup EXIT

if [[ "${GPU_ID}" == *,* ]]; then
    GRAPH_BUILD_DEVICE="${GRAPH_BUILD_DEVICE:-cuda:1}"
else
    GRAPH_BUILD_DEVICE="${GRAPH_BUILD_DEVICE:-cpu}"
fi
GRAPH_BUILD_CHUNK_SIZE="${GRAPH_BUILD_CHUNK_SIZE:-1024}"
USE_GPU="${USE_GPU:-1}"
DISABLE_FILE_LOG="${DISABLE_FILE_LOG:-1}"
QUALITY_CKPT_DIR="${QUALITY_CKPT_DIR:-}"
EPOCHS="${EPOCHS:-}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-}"

if [[ -x "/.conda/envs/mmrs/bin/python" ]]; then
    PYTHON_BIN="/.conda/envs/mmrs/bin/python"
else
    PYTHON_BIN="python"
fi

declare -A RESULTS
declare -a JOB_TAGS
declare -a JOB_PIDS

log() { echo "$@" | tee -a "${SUMMARY_FILE}"; }

parse_best_result() {
    local logfile="$1"
    local best_line=""
    local result=""

    best_line="$(grep "best test:" "${logfile}" | tail -1 || true)"
    if [[ -n "${best_line}" ]]; then
        result="$(printf '%s\n' "${best_line}" | \
                 grep -oP 'recall@10: \K[0-9.]+|ndcg@10: \K[0-9.]+' | \
                 paste -sd '/' - || true)"
    fi

    printf '%s' "${result:-N/A}"
}

get_gpu_for_condition() {
    local ctype="$1"
    case "${ctype}" in
        noisy) printf '%s' "${NOISY_GPU}" ;;
        missing) printf '%s' "${MISSING_GPU}" ;;
        mixed) printf '%s' "${MIXED_GPU}" ;;
        *) printf '%s' "${GPU_ID}" ;;
    esac
}

record_result() {
    local tag="$1"
    local status="$2"
    local value="$3"
    local logfile="$4"
    printf '%s\t%s\t%s\n' "${status}" "${value}" "${logfile}" > "${RESULT_TMP_DIR}/${tag}.tsv"
}

collect_result() {
    local tag="$1"
    local result_file="${RESULT_TMP_DIR}/${tag}.tsv"
    local status=""
    local value=""
    local logfile=""

    if [[ ! -f "${result_file}" ]]; then
        RESULTS["${tag}"]="FAIL"
        log "[WARN] 결과 파일 없음: ${tag}"
        return
    fi

    IFS=$'\t' read -r status value logfile < "${result_file}"
    case "${status}" in
        OK)
            RESULTS["${tag}"]="${value}"
            log "[${MODEL_NAME}][${tag}] recall@10/ndcg@10 = ${value}"
            ;;
        SKIP)
            RESULTS["${tag}"]="SKIP"
            log "[WARN] ${tag} skip: ${value}"
            ;;
        FAIL)
            RESULTS["${tag}"]="FAIL"
            log "[WARN] 실행 실패: ${tag} (log: ${logfile})"
            ;;
        *)
            RESULTS["${tag}"]="FAIL"
            log "[WARN] 알 수 없는 결과 상태: ${tag} (${status})"
            ;;
    esac
}

run_clean_async() {
    local gpu_id="$1"
    local tag="clean"
    local logfile="${LOG_DIR}/${MODEL_NAME}-${DATASET_NAME}-${tag}-$(date +%b-%d-%Y-%H-%M-%S).log"
    local result=""

    cd "${SRC_DIR}"
    if ! "${PYTHON_BIN}" - "${gpu_id}" "${DATASET_NAME}" "${DATA_PATH}" \
                         "${MODEL_CONFIG_NAME}" \
                         "${LOG_DIR}" "${GRAPH_BUILD_DEVICE}" "${GRAPH_BUILD_CHUNK_SIZE}" \
                         "${CLEAN_IMG_FILE}" "${CLEAN_TXT_FILE}" "${USE_GPU}" \
                         "${DISABLE_FILE_LOG}" "${QUALITY_CKPT_DIR}" "${EPOCHS}" \
                         "${TRAIN_BATCH_SIZE}" "${EVAL_BATCH_SIZE}" <<'PY' > "${logfile}" 2>&1
import sys
from utils.quick_start import quick_start

(
    gpu_id,
    dataset_name,
    data_path,
    model_config_name,
    log_dir,
    graph_build_device,
    graph_build_chunk_size,
    clean_img_file,
    clean_txt_file,
    use_gpu,
    disable_file_log,
    quality_ckpt_dir,
    epochs,
    train_batch_size,
    eval_batch_size,
) = sys.argv[1:16]

config_dict = {
    "model_config_name": model_config_name,
    "gpu_id": gpu_id,
    "data_path": data_path,
    "split_mode": "recbole_ls",
    "missing_modal": 0,
    "missing_ratio": 0.0,
    "use_gpu": str(use_gpu).lower() in {"1", "true", "yes", "y", "on"},
    "disable_file_log": str(disable_file_log).lower() in {"1", "true", "yes", "y", "on"},
    "log_dir": log_dir,
    "graph_build_device": graph_build_device,
    "graph_build_chunk_size": int(graph_build_chunk_size),
    "repair_target_vision_feature_file": clean_img_file,
    "repair_target_text_feature_file": clean_txt_file,
}
if quality_ckpt_dir:
    config_dict["quality_ckpt_dir"] = quality_ckpt_dir
if epochs:
    config_dict["epochs"] = int(epochs)
if train_batch_size:
    config_dict["train_batch_size"] = int(train_batch_size)
if eval_batch_size:
    config_dict["eval_batch_size"] = int(eval_batch_size)

quick_start(model="StateMoE", dataset=dataset_name, config_dict=config_dict, save_model=False)
PY
    then
        record_result "${tag}" "FAIL" "train failed" "${logfile}"
        return 1
    fi

    result="$(parse_best_result "${logfile}")"
    record_result "${tag}" "OK" "${result}" "${logfile}"
}

run_one_async() {
    local ctype="$1"
    local ratio="$2"
    local gpu_id="$3"
    local tag="${ctype}${ratio}"
    local img_file="single/${ctype}/image_feat_single_${ctype}${ratio}.npy"
    local txt_file="single/${ctype}/text_feat_single_${ctype}${ratio}.npy"
    local gt_state_file="single/${ctype}/gt_state_single_${ctype}${ratio}.npy"
    local logfile="${LOG_DIR}/${MODEL_NAME}-${DATASET_NAME}-${tag}-$(date +%b-%d-%Y-%H-%M-%S).log"
    local result=""

    if [[ ! -f "${DATASET_DIR}/${img_file}" ]]; then
        record_result "${tag}" "SKIP" "missing file: ${DATASET_DIR}/${img_file}" "-"
        return 0
    fi
    if [[ ! -f "${DATASET_DIR}/${txt_file}" ]]; then
        record_result "${tag}" "SKIP" "missing file: ${DATASET_DIR}/${txt_file}" "-"
        return 0
    fi

    cd "${SRC_DIR}"
    if ! "${PYTHON_BIN}" - "${gpu_id}" "${DATASET_NAME}" "${DATA_PATH}" \
                         "${MODEL_CONFIG_NAME}" \
                         "${img_file}" "${txt_file}" "${gt_state_file}" "${LOG_DIR}" \
                         "${GRAPH_BUILD_DEVICE}" "${GRAPH_BUILD_CHUNK_SIZE}" \
                         "${CLEAN_IMG_FILE}" "${CLEAN_TXT_FILE}" "${USE_GPU}" \
                         "${DISABLE_FILE_LOG}" "${QUALITY_CKPT_DIR}" "${EPOCHS}" \
                         "${TRAIN_BATCH_SIZE}" "${EVAL_BATCH_SIZE}" <<'PY' > "${logfile}" 2>&1
import sys
from utils.quick_start import quick_start

(
    gpu_id,
    dataset_name,
    data_path,
    model_config_name,
    img_file,
    txt_file,
    gt_state_file,
    log_dir,
    graph_build_device,
    graph_build_chunk_size,
    clean_img_file,
    clean_txt_file,
    use_gpu,
    disable_file_log,
    quality_ckpt_dir,
    epochs,
    train_batch_size,
    eval_batch_size,
) = sys.argv[1:19]

config_dict = {
    "model_config_name": model_config_name,
    "gpu_id": gpu_id,
    "data_path": data_path,
    "split_mode": "recbole_ls",
    "missing_modal": 0,
    "missing_ratio": 0.0,
    "vision_feature_file": img_file,
    "text_feature_file": txt_file,
    "gt_state_file": gt_state_file,
    "use_gpu": str(use_gpu).lower() in {"1", "true", "yes", "y", "on"},
    "disable_file_log": str(disable_file_log).lower() in {"1", "true", "yes", "y", "on"},
    "log_dir": log_dir,
    "graph_build_device": graph_build_device,
    "graph_build_chunk_size": int(graph_build_chunk_size),
    "repair_target_vision_feature_file": clean_img_file,
    "repair_target_text_feature_file": clean_txt_file,
}
if quality_ckpt_dir:
    config_dict["quality_ckpt_dir"] = quality_ckpt_dir
if epochs:
    config_dict["epochs"] = int(epochs)
if train_batch_size:
    config_dict["train_batch_size"] = int(train_batch_size)
if eval_batch_size:
    config_dict["eval_batch_size"] = int(eval_batch_size)

quick_start(model="StateMoE", dataset=dataset_name, config_dict=config_dict, save_model=False)
PY
    then
        record_result "${tag}" "FAIL" "train failed" "${logfile}"
        return 1
    fi

    result="$(parse_best_result "${logfile}")"
    record_result "${tag}" "OK" "${result}" "${logfile}"
}

CORR_TYPES=""
for ctype in ${COND_TYPES_RAW}; do
    if [[ "${ctype}" == "clean" ]]; then
        continue
    fi
    CORR_TYPES+=" ${ctype}"
done
COND_TYPES="${CORR_TYPES# }"

log "================================================================"
log "  ${MODEL_NAME} 로버스트니스 평가 시작: $(date)"
log "  Dataset: ${DATASET_DIR}"
log "  Conditions: clean + ${COND_TYPES:-<none>} × ratios ${RATIOS}"
log "  clean GPU: ${GPU_ID} | noisy GPU: ${NOISY_GPU} | missing GPU: ${MISSING_GPU} | mixed GPU: ${MIXED_GPU}"
log "  Model config: ${MODEL_CONFIG_NAME}"
log "  Graph build: device=${GRAPH_BUILD_DEVICE}, chunk=${GRAPH_BUILD_CHUNK_SIZE}"
log "  Repair targets: image=${CLEAN_IMG_FILE}, text=${CLEAN_TXT_FILE}"
log "================================================================"

log "[${MODEL_NAME}][clean] 시작... gpu=${GPU_ID}"
run_clean_async "${GPU_ID}" &
JOB_TAGS+=("clean")
JOB_PIDS+=("$!")

if [[ -n "${COND_TYPES}" ]]; then
    for ctype in ${COND_TYPES}; do
        for ratio in ${RATIOS}; do
            tag="${ctype}${ratio}"
            assigned_gpu="$(get_gpu_for_condition "${ctype}")"
            log "[${MODEL_NAME}][${tag}] 시작... gpu=${assigned_gpu}"
            run_one_async "${ctype}" "${ratio}" "${assigned_gpu}" &
            JOB_TAGS+=("${tag}")
            JOB_PIDS+=("$!")
        done
    done
fi

for idx in "${!JOB_TAGS[@]}"; do
    if ! wait "${JOB_PIDS[$idx]}"; then
        :
    fi
    collect_result "${JOB_TAGS[$idx]}"
done

log ""
log "═══════════════════════════════════════════════════"
log "  ${MODEL_NAME} 결과 요약 (recall@10 / ndcg@10)"
log "  Dataset: ${DATASET_NAME}  |  $(date)"
log "═══════════════════════════════════════════════════"
log "$(printf '%-15s  %s' 'Condition' 'recall@10 / ndcg@10')"
log "$(printf '%-15s  %s' '---------' '--------------------')"
log "$(printf '%-15s  %s' 'clean' "${RESULTS["clean"]:-N/A}")"
if [[ -n "${COND_TYPES}" ]]; then
    for ctype in ${COND_TYPES}; do
        for ratio in ${RATIOS}; do
            log "$(printf '%-15s  %s' "${ctype}${ratio}" "${RESULTS["${ctype}${ratio}"]:-N/A}")"
        done
    done
fi
log "═══════════════════════════════════════════════════"
log "결과 파일: ${SUMMARY_FILE}"
