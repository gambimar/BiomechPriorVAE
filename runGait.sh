#!/bin/bash

# Unified gait simulation runner
#
# Usage:
#   ./runGait.sh --model=gait3d_pelvis213.osim --metmodel=houdijk --min=0.73 --max=1.63
#   ./runGait.sh --model=sipp_generic_runmad.osim --metmodel=0 --min=0.73 --max=1.63
#   ./runGait.sh --model=gait3d_pelvis213.osim    # metmodel defaults to 0

export LC_ALL=C
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || exit 2
cd -- "$repo_dir" || exit 2

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
model_stem="${model_file//\\//}"
model_stem="${model_stem##*/}"
model_stem="${model_stem%.osim}"
# Replace any dots or spaces with underscores for safety
model_stem="${model_stem//[. ]/_}"

if [ "$metmodel" = "0" ]; then
    prefix="${model_stem}"
    echo "Running gait simulations: model=${model_file}, metmodel=0 (none)"
else
    prefix="${model_stem}_${metmodel}"
    echo "Running gait simulations: model=${model_file}, metmodel=${metmodel}"
fi

number_pattern='^([0-9]+([.][0-9]*)?|[.][0-9]+)$'
if [[ ! $min_speed =~ $number_pattern || ! $max_speed =~ $number_pattern ]]; then
    echo "Error: --min and --max must be nonnegative decimal speeds."
    exit 1
fi

# Generate integer hundredths to avoid accumulating floating-point steps.
selected_speeds=$(awk -v min="$min_speed" -v max="$max_speed" '
    function emit(s) { if (s / 100 >= min && s / 100 <= max) printf "%.2f\n", s / 100 }
    BEGIN {
        emit(0)
        for (s = 73; s <= 353; s += 10) emit(s)
        for (s = 373; s <= 563; s += 20) emit(s)
        emit(600)
        emit(700)
    }') || exit 2
if [ -z "$selected_speeds" ]; then
    echo "Error: No valid gait speed bins found in the range [$min_speed, $max_speed]."
    echo "Valid bins are: 0.00, 0.73 - 3.53 in 0.1 steps, 3.73 - 5.53 in 0.2 steps, 6.00, 7.00."
    exit 1
fi
mapfile -t speeds_to_run <<< "$selected_speeds"

# A quote inside a MATLAB character vector is represented by two quotes.
matlab_prefix="${prefix//\'/\'\'}"
matlab_model="${model_file//\'/\'\'}"
matlab_metmodel="${metmodel//\'/\'\'}"

echo "Speeds to run: ${speeds_to_run[*]}"

# Loop over valid speeds in range
for speed_formatted in "${speeds_to_run[@]}"; do
    echo "================================================================="
    echo "Starting simulations for speed: $speed_formatted"
    echo "================================================================="

    while true; do
        LM_PROJECT=iwse matlab -batch "addpath('scripts/func'); done = runGaitSingleStep($speed_formatted, '$matlab_prefix', '$matlab_model', '$matlab_metmodel'); exit(10 * double(done));"
        exit_code=$?

        # MATLAB -batch uses 1 for unhandled errors; reserve 10 for completion.
        if [ $exit_code -eq 10 ]; then
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
