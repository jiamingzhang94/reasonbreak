#!/bin/bash

# ==============================================================================
# DoxBench Pipeline Script
# Usage:
#   bash run.sh --mode clean --step eval --dox_path /path/to/dox_data
#   bash run.sh --mode adv --step gen --dox_path /path/to/dox_data
#   bash run.sh --mode adv --step all --dox_path /path/to/dox_data
# ==============================================================================

# --- Global Config ---
PROJECT_PATH="$(pwd)"
DECODER_PATH="checkpoints/weight.pth"
EMBEDDING_PATH="data/embedding_bank.pth"
JSON_PATH="data/json/cot_full.json"
ADV_CSV_OUTPUT="outputs/adv.csv"

# --- Argument Parsing ---
MODE="adv"   # clean | adv
STEP="all"   # gen | eval | all
DOX_PATH=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --mode) MODE="$2"; shift 2 ;;
        --step) STEP="$2"; shift 2 ;;
        --dox_path) DOX_PATH="$2"; shift 2 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

# Check required argument
if [ -z "$DOX_PATH" ]; then
    echo "Error: --dox_path is required."
    exit 1
fi

echo ">> [Config] Mode: ${MODE} | Step: ${STEP}"
echo ">> [Paths]  Dox: ${DOX_PATH} | Project: ${PROJECT_PATH}"

# --- Path Setup ---
if [ "$MODE" == "clean" ]; then
    INPUT_CSV="${DOX_PATH}/result.csv"
    ROOT_PATH="${DOX_PATH}"
    OUTPUT_DIR="result/clean"
elif [ "$MODE" == "adv" ]; then
    INPUT_CSV="${PROJECT_PATH}/${ADV_CSV_OUTPUT}"
    ROOT_PATH="${PROJECT_PATH}"
    OUTPUT_DIR="result/adv"
else
    echo "Error: Mode must be 'clean' or 'adv'"
    exit 1
fi

# ==============================================================================
# PHASE 1: Generation
# ==============================================================================
if [[ "$MODE" == "adv" ]] && [[ "$STEP" == "gen" || "$STEP" == "all" ]]; then
    echo ""
    echo ">> [Phase 1] Generating Adversarial Samples..."
    
    mkdir -p "$(dirname ${ADV_CSV_OUTPUT})"

    python generate.py \
        --decoder_path "${DECODER_PATH}" \
        --image_root "${DOX_PATH}" \
        --embedding_bank_path "${EMBEDDING_PATH}" \
        --json_path "${JSON_PATH}" \
        --output_dir "$(dirname ${ADV_CSV_OUTPUT})" \
        --csv_path "${ADV_CSV_OUTPUT}" \
        --epsilon "0.0627"

    if [ $? -ne 0 ]; then
        echo "❌ Error: Generation failed."
        exit 1
    fi
    echo "✅ Generation complete."
fi

# ==============================================================================
# PHASE 2: Evaluation
# ==============================================================================
if [[ "$STEP" == "eval" || "$STEP" == "all" ]]; then
    echo ""
    echo ">> [Phase 2] Starting Evaluation..."
    
    # Switch to experiment directory
    cd code/experiment/ || exit

    # --- Methods Config ---
    declare -a METHODS=(
   "gpt5_cot_off_top1|gpt5|off|1|off|off|off|"
   "gemini_cot_off_top1|gemini|off|1|off|off|off|"
   "o3_cot_off_top1|o3|off|1|off|off|off|"
#    "qwvl-2.5_cot_on_top1|qwvl-2.5|on|1|off|off|off|"
    # "qvq-max_cot_on_top1|qvq-max|on|1|off|off|off|"
#    "qwvl-max_cot_on_top1|gpt4o|on|1|off|off|off|"
        # Add more methods here...
    )

    GEOMINER_DETECTOR_MODEL="gpt4o"
    PARALLEL="1"

    # --- Single Method Execution ---
    run_method() {
        local method_name="$1"
        local model="$2"
        local cot="$3"
        local top_n="$4"
        local reasoning_summary="$5"
        local prompt_base_defense="$6"
        local prompt_based_defense="$7"
        local noise="$8"

        echo ">> Running Method: $method_name"

        PARAMS="--input_csv \"$INPUT_CSV\" --output \"$OUTPUT_DIR\""
        PARAMS="$PARAMS --root_path \"$ROOT_PATH\""
        PARAMS="$PARAMS --model \"$model\""
        PARAMS="$PARAMS --geominer_detector_model \"$GEOMINER_DETECTOR_MODEL\""
        PARAMS="$PARAMS --cot \"$cot\""
        PARAMS="$PARAMS --reasoning_summary \"$reasoning_summary\""
        PARAMS="$PARAMS --prompt_base_defense \"$prompt_base_defense\""
        PARAMS="$PARAMS --prompt-based-defense \"$prompt_based_defense\""
        PARAMS="$PARAMS --parallel \"$PARALLEL\""
        PARAMS="$PARAMS --output_filename \"${method_name}.csv\""

        if [ "$top_n" = "3" ]; then PARAMS="$PARAMS --top3"; else PARAMS="$PARAMS --top1"; fi
        if [ -n "$noise" ]; then PARAMS="$PARAMS --noise \"$noise\""; fi

        eval "python experiment.py $PARAMS"
    }

    # --- Run Loop ---
    for method_config in "${METHODS[@]}"; do
        IFS='|' read -r m_name m_model m_cot m_top m_summ m_base m_prompt m_noise <<< "$method_config"
        run_method "$m_name" "$m_model" "$m_cot" "$m_top" "$m_summ" "$m_base" "$m_prompt" "$m_noise"
    done
    
    # Return to project root for analysis
    cd ../../
    
    # --- PHASE 3: Report ---
    if [[ "$MODE" == "adv" ]]; then
        echo ""
        echo ">> [Phase 3] Generating Report..."
        
        # Paths are relative to project root
        CLEAN_RESULT_DIR="code/experiment/result/clean"
        ADV_RESULT_DIR="code/experiment/result/adv"
        FINAL_OUTPUT_DIR="outputs/report"

        if [ -d "$CLEAN_RESULT_DIR" ]; then
             python analyze_privacy.py \
                --clean_dir "$CLEAN_RESULT_DIR" \
                --adv_dir "$ADV_RESULT_DIR" \
                --output_dir "$FINAL_OUTPUT_DIR"
        else
            echo "⚠️ Warning: Clean baseline results not found at $CLEAN_RESULT_DIR"
            echo "Skipping report generation."
        fi
    fi
fi