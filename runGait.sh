#!/bin/bash

# Unified gait simulation runner
#
# Usage:
#   ./runGait.sh --model=gait3d_pelvis213.osim --metmodel=houdijk --min=0.73 --max=1.63
#   ./runGait.sh --model=sipp_generic_runmad.osim --metmodel=0 --min=0.73 --max=1.63
#   ./runGait.sh --model=gait3d_pelvis213.osim    # metmodel defaults to 0

min_speed="0.73"
max_speed="1.63"
model_file=""
metmodel="0"

for arg in "$@"; do
    case $arg in
        --min=*)
            min_speed="${arg#*=}"
            shift
            ;;
        --max=*)
            max_speed="${arg#*=}"
            shift
            ;;
        --model=*)
            model_file="${arg#*=}"
            shift
            ;;
        --metmodel=*)
            metmodel="${arg#*=}"
            shift
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 --model=<model.osim> [--metmodel=<name|0>] [--min=SPEED] [--max=SPEED]"
            exit 1
            ;;
    esac
done

# --model is required
if [ -z "$model_file" ]; then
    echo "Error: --model is required."
    echo "Usage: $0 --model=<model.osim> [--metmodel=<name|0>] [--min=SPEED] [--max=SPEED]"
    exit 1
fi

# Derive a short prefix from the model filename (strip .osim extension)
model_stem="${model_file%.osim}"
# Replace any dots or spaces with underscores for safety
model_stem="${model_stem//[. ]/_}"

if [ "$metmodel" = "0" ]; then
    prefix="${model_stem}"
    echo "Running gait simulations: model=${model_file}, metmodel=0 (none)"
else
    prefix="${model_stem}_${metmodel}"
    echo "Running gait simulations: model=${model_file}, metmodel=${metmodel}"
fi

# Format min and max speeds
min_formatted=$(printf "%.2f" "$min_speed" 2>/dev/null || echo "$min_speed")
max_formatted=$(printf "%.2f" "$max_speed" 2>/dev/null || echo "$max_speed")

# Generate the historical valid speed bins
valid_speeds=()
# add 0 speed (e.g. free speed walking)
valid_speeds+=("0.00")
# 0.73 to 3.53 in steps of 0.1
for s in $(seq 0.73 0.1 3.53); do
    valid_speeds+=($(printf "%.2f" "$s"))
done
# 3.73 to 5.63 in steps of 0.2
for s in $(seq 3.73 0.2 5.63); do
    valid_speeds+=($(printf "%.2f" "$s"))
done
for s in $(seq 6 1 7); do
    valid_speeds+=($(printf "%.2f" "$s"))
done
# Filter valid speeds to range [min_speed, max_speed]
speeds_to_run=()
for s in "${valid_speeds[@]}"; do
    if (( $(echo "$s >= $min_formatted" | bc -l) )) && (( $(echo "$s <= $max_formatted" | bc -l) )); then
        speeds_to_run+=("$s")
    fi
done

if [ ${#speeds_to_run[@]} -eq 0 ]; then
    echo "Error: No valid gait speed bins found in the range [$min_formatted, $max_formatted]."
    echo "Valid bins are: 0.73 - 3.53 in 0.1 steps, and 3.73 - 5.63 in 0.2 steps."
    exit 1
fi

echo "Speeds to run: ${speeds_to_run[*]}"

# Loop over valid speeds in range
for speed_formatted in "${speeds_to_run[@]}"; do
    echo "================================================================="
    echo "Starting simulations for speed: $speed_formatted"
    echo "================================================================="

    while true; do
        LM_PROJECT=iwse matlab -batch "addpath('scripts/func'); done = runGaitSingleStep($speed_formatted, '$prefix', '$model_file', '$metmodel'); exit(double(done));"
        exit_code=$?

        if [ $exit_code -eq 1 ]; then
            echo "Speed $speed_formatted: Successfully reached 10 converged runs."
            break
        elif [ $exit_code -eq 0 ]; then
            echo "Restarting MATLAB process to clear memory..."
        else
            echo "MATLAB exited with error code $exit_code. Aborting."
            exit $exit_code
        fi
    done
done

echo "All simulations completed successfully!"
