#!/usr/bin/env bash
# Full pipeline: generate -> validate -> screen
# Usage: ./pipeline.sh "Li,Ni,Ti,O" 100

set -u

ELEMENTS=${1:-"Li,Ni,Ti,O"}
N_SAMPLES=${2:-100}
MAX_STOICH=6
MAX_NATOMS=20
MAX_FACTOR=6

RUN_TS=$(date +%Y%m%d_%H%M%S)
SAVE_DIR="results/navigate_${RUN_TS}"
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/pipeline_${RUN_TS}.log"

CYAN="\033[1;36m"
GREEN="\033[1;32m"
RED="\033[1;31m"
DIM="\033[2m"
RESET="\033[0m"

PIPELINE_START_EPOCH=0
HEARTBEAT_PID=""

timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

epoch_now() {
    date +%s
}

format_duration() {
    local total=$1
    local hours=$((total / 3600))
    local mins=$(((total % 3600) / 60))
    local secs=$((total % 60))
    printf "%dh %02dm %02ds" "$hours" "$mins" "$secs"
}

log_line() {
    echo -e "$*"
}

start_heartbeat() {
    while true; do
        sleep 300
        log_line "${DIM}[HEARTBEAT $(date +%H:%M:%S)] Pipeline still running — PID $$${RESET}"
    done
}

stop_heartbeat() {
    if [[ -n "$HEARTBEAT_PID" ]]; then
        kill "$HEARTBEAT_PID" >/dev/null 2>&1 || true
    fi
}

fail_step() {
    local step_idx=$1
    local exit_code=$2
    local fail_time
    fail_time=$(timestamp)
    log_line "${RED}[ERROR] STEP ${step_idx} failed at ${fail_time} with exit code ${exit_code}${RESET}"
    log_line "${RED}[ERROR] Check log for details above${RESET}"
    log_line "═══════════════════════════════════════════════════"
    log_line "PIPELINE FAILED AT STEP ${step_idx}"
    log_line "═══════════════════════════════════════════════════"
    log_line "Failed at     : ${fail_time}"
    log_line "Error         : Step ${step_idx} exited with code ${exit_code}"
    log_line "Partial log   : ${LOG_FILE}"
    log_line "═══════════════════════════════════════════════════"
    stop_heartbeat
    exit 1
}

count_cifs() {
    local dir=$1
    if [[ -d "$dir" ]]; then
        find "$dir" -type f -name "*.cif" 2>/dev/null | wc -l | tr -d " "
    else
        echo "0"
    fi
}

run_pipeline() {
    mkdir -p "$LOG_DIR"

    log_line "To monitor progress from another terminal run:"
    log_line "  tail -f ${LOG_FILE}"
    log_line "TIP: To run detached so SSH disconnect won't kill it:"
    log_line "  nohup ./pipeline.sh ${ELEMENTS} ${N_SAMPLES} &"
    log_line "  tail -f logs/pipeline_*.log"
    log_line ""

    PIPELINE_START_EPOCH=$(epoch_now)
    local start_time
    start_time=$(timestamp)
    local host
    host=$(hostname)
    local python_path
    python_path=$(command -v python)
    local workdir
    workdir=$(pwd)

    log_line "═══════════════════════════════════════════════════"
    log_line "CATHODE DISCOVERY PIPELINE — RUN LOG"
    log_line "═══════════════════════════════════════════════════"
    log_line "Start time    : ${start_time}"
    log_line "Elements      : ${ELEMENTS}"
    log_line "N samples     : ${N_SAMPLES}"
    log_line "Max stoich    : ${MAX_STOICH}"
    log_line "Max natoms    : ${MAX_NATOMS}"
    log_line "Max factor    : ${MAX_FACTOR}"
    log_line "Save dir      : ${SAVE_DIR}"
    log_line "Hostname      : ${host}"
    log_line "Python        : ${python_path}"
    log_line "Working dir   : ${workdir}"
    log_line "═══════════════════════════════════════════════════"

    start_heartbeat &
    HEARTBEAT_PID=$!

    local step_start
    local step_end
    local duration
    local cif_count
    local passed
    local total

    log_line "───────────────────────────────────────────────────"
    log_line "${CYAN}[STEP 1/3] Structure Generation${RESET}"
    log_line "Started  : $(timestamp)"
    log_line "───────────────────────────────────────────────────"
    step_start=$(epoch_now)
    python -m chemeleon.cli navigate system \
        --elements "$ELEMENTS" \
        --max-stoich "$MAX_STOICH" \
        --n-samples "$N_SAMPLES" \
        --max-natoms "$MAX_NATOMS" \
        --max-factor "$MAX_FACTOR" \
        --save-dir "$SAVE_DIR"
    if [[ $? -ne 0 ]]; then
        fail_step 1 $?
    fi
    step_end=$(epoch_now)
    duration=$(format_duration $((step_end - step_start)))
    cif_count=$(count_cifs "$SAVE_DIR")
    log_line "───────────────────────────────────────────────────"
    log_line "${GREEN}[STEP 1/3] COMPLETED${RESET}"
    log_line "Finished : $(timestamp)"
    log_line "Duration : ${duration}"
    log_line "CIF files generated: ${cif_count}"
    log_line "───────────────────────────────────────────────────"

    log_line "───────────────────────────────────────────────────"
    log_line "${CYAN}[STEP 2/3] Structural Validation${RESET}"
    log_line "Started  : $(timestamp)"
    log_line "───────────────────────────────────────────────────"
    step_start=$(epoch_now)
    python validator.py
    if [[ $? -ne 0 ]]; then
        fail_step 2 $?
    fi
    step_end=$(epoch_now)
    duration=$(format_duration $((step_end - step_start)))
    total=$cif_count
    passed="?"
    if [[ -f "$LOG_FILE" ]]; then
        local pass_line
        pass_line=$(grep "Passed:" "$LOG_FILE" | tail -1 || true)
        if [[ -n "$pass_line" ]]; then
            passed=$(echo "$pass_line" | awk '{print $2}')
            total=$(echo "$pass_line" | awk '{print $4}')
        fi
    fi
    log_line "───────────────────────────────────────────────────"
    log_line "${GREEN}[STEP 2/3] COMPLETED${RESET}"
    log_line "Finished : $(timestamp)"
    log_line "Duration : ${duration}"
    log_line "Structures passed validation: ${passed} / ${total}"
    log_line "───────────────────────────────────────────────────"

    log_line "───────────────────────────────────────────────────"
    log_line "${CYAN}[STEP 3/3] CHGNet Ehull Screening${RESET}"
    log_line "Started  : $(timestamp)"
    log_line "───────────────────────────────────────────────────"
    step_start=$(epoch_now)
    python chgnet_layer.py
    if [[ $? -ne 0 ]]; then
        fail_step 3 $?
    fi
    step_end=$(epoch_now)
    duration=$(format_duration $((step_end - step_start)))
    local ehull_pass
    local ehull_total
    ehull_pass="?"
    ehull_total="?"
    if [[ -f results/chgnet_screening.json ]]; then
        ehull_pass=$(grep -c '"is_metastable": true' results/chgnet_screening.json || true)
        ehull_total=$(grep -c '"is_metastable"' results/chgnet_screening.json || true)
    fi
    log_line "───────────────────────────────────────────────────"
    log_line "${GREEN}[STEP 3/3] COMPLETED${RESET}"
    log_line "Finished : $(timestamp)"
    log_line "Duration : ${duration}"
    log_line "Structures passed Ehull screening: ${ehull_pass} / ${ehull_total}"
    log_line "───────────────────────────────────────────────────"

    stop_heartbeat

    local end_time
    end_time=$(timestamp)
    local total_duration
    total_duration=$(format_duration $(( $(epoch_now) - PIPELINE_START_EPOCH )))

    log_line "═══════════════════════════════════════════════════"
    log_line "PIPELINE COMPLETED SUCCESSFULLY"
    log_line "═══════════════════════════════════════════════════"
    log_line "End time      : ${end_time}"
    log_line "Total duration: ${total_duration}"
    log_line "Structures generated : ${cif_count}"
    log_line "Passed validation    : ${passed}"
    log_line "Passed Ehull screen  : ${ehull_pass}"
    log_line "Results saved to     : ${SAVE_DIR}"
    log_line "Log saved to         : ${LOG_FILE}"
    log_line "═══════════════════════════════════════════════════"
}

run_pipeline 2>&1 | tee >(sed 's/\x1b\[[0-9;]*m//g' >> "$LOG_FILE")