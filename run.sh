#!/bin/bash
set -e

# Always run from the project root directory
cd "$(dirname "$0")"

mkdir -p results

echo "=========================================================="
echo "🩺 EndoClean: Endoscopy Denoising & Classification Pipeline"
echo "=========================================================="

MODE="${1:-run}"

# Function to start background training
start_background_training() {
    PID_FILE="results/training.pid"
    LOG_FILE="results/training.log"

    if [ -f "$PID_FILE" ]; then
        OLD_PID=$(cat "$PID_FILE" 2>/dev/null || echo "")
        if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
            echo "ℹ️  A training process is already running in the background (PID: $OLD_PID)."
            echo "📄 Track progress with: tail -f $LOG_FILE"
            return
        else
            rm -f "$PID_FILE"
        fi
    fi

    echo "🚀 Starting model training in the BACKGROUND..."
    echo "📄 Output log: $LOG_FILE"

    # Launch background job detached
    nohup bash -c '
        echo "=== Training session started: $(date) ===" > results/training.log
        ./run_experiments.sh >> results/training.log 2>&1
        rm -f results/training.pid
        echo "=== Training session completed: $(date) ===" >> results/training.log
    ' > /dev/null 2>&1 &

    TRAIN_PID=$!
    echo "$TRAIN_PID" > "$PID_FILE"
    echo "✓ Training job dispatched with PID: $TRAIN_PID"
    echo "  (To monitor in terminal: tail -f $LOG_FILE)"
    echo ""
}

# Check if retraining was requested
if [[ "$MODE" == "--all" || "$MODE" == "all" || "$MODE" == "--train" || "$MODE" == "train" ]]; then
    start_background_training
else
    RAW_MODEL="results/checkpoints/raw_seed42/best_model.pth"
    PDE_MODEL="results/checkpoints/pde_denoised_seed42/best_model.pth"

    if [[ ! -f "$RAW_MODEL" || ! -f "$PDE_MODEL" ]]; then
        echo "⚠️ Trained checkpoints missing. Launching training in background..."
        start_background_training
    else
        echo "✓ Existing trained checkpoints found."
    fi
fi

# Make sure aggregated metrics file exists so dashboard does not fail
if [[ ! -f "results/test_metrics_aggregated.csv" ]]; then
    echo "▶ Initializing baseline test metrics..."
    python3 src/evaluate.py --metadata_csv "data/metadata.csv" > /dev/null 2>&1 || true
fi

echo "=========================================================="
echo "▶ Launching EndoClean Web Dashboard in the foreground..."
echo "  The app is fully functional while training runs in background!"
echo "=========================================================="

streamlit run demo/app.py
