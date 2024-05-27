#!/bin/bash

# Check if an experiment name was provided
if [ "$#" -ne 4 ] && [ "$#" -ne 5 ]; then
    echo "Usage: $0 SAMPLETYPE MODEL_TYPE MODEL_WEIGHTS BATCH_SIZE [VAE_PRIOR]"
    exit 1
fi

# The first argument is the experiment name
SAMPLETYPE="$1"
MODEL_TYPE="$2"
MODEL_WEIGHTS="$3"
BATCH_SIZE="$4"
# Check if SAMPLETYPE is either "sample" or "sample_cond"
if [[ "${SAMPLETYPE}" != "sample" ]] && [[ "${SAMPLETYPE}" != "sample-cond" ]]; then
    echo "Usage: $0 SAMPLETYPE MODEL_TYPE MODEL_WEIGHTS BATCH_SIZE [VAE_PRIOR]"
    echo "SAMPLETYPE must be either 'sample' or 'sample_cond'"
    exit 1
fi

# Source your environment variables
source ./env_vars.sh

# Path to the job script template
JOB_SCRIPT_TEMPLATE="./jobscript_template_sample.sh"

# Temporary job script file that will be populated with environment variables and the experiment name
TEMP_JOB_SCRIPT="./jobscript_populated_sample.sh"


# Optionally provide two additional arguments for the loss_type and the model_type
if [ "$#" -eq 4 ]; then
    VAE_PRIOR="$4"
    RUN_EXPERIMENTS_ARGS="--model-type ${MODEL_TYPE} --load-weights ${MODEL_WEIGHTS} --batch-size ${BATCH_SIZE} --p-uncond 0.1 --num-samples 10000 --vae-prior ${VAE_PRIOR}"
else 
    RUN_EXPERIMENTS_ARGS="--model-type ${MODEL_TYPE} --load-weights ${MODEL_WEIGHTS} --batch-size ${BATCH_SIZE} --p-uncond 0.1 --num-samples 10000"
fi

# Replace placeholders in the template with actual environment variable values and the experiment name
sed -e "s|\${VENV_PATH}|$VENV_PATH|g" \
    -e "s|\${WORKING_DIR}|$WORKING_DIR|g" \
    -e "s|\${EMAIL}|$EMAIL|g" \
    -e "s|\${QUEUE}|$QUEUE|g" \
    -e "s|\${WALLTIME}|$WALLTIME|g" \
    -e "s|\${GPU_MODE}|$GPU_MODE|g" \
    -e "s|\${NUM_CORES}|$NUM_CORES|g" \
    -e "s|\${MEM_GB}|$MEM_GB|g" \
    -e "s|\${EXPERIMENT_NAME}|$MODEL|g" \
    -e "s|\${RUN_EXPERIMENTS_ARGS}|$RUN_EXPERIMENTS_ARGS|g" \
    -e "s|\${SAMPLETYPE}|$SAMPLETYPE|g" \
    "$JOB_SCRIPT_TEMPLATE" > "$TEMP_JOB_SCRIPT"

# Submit the job
bsub < "$TEMP_JOB_SCRIPT"

# Optionally, remove the temporary job script after submission
rm "$TEMP_JOB_SCRIPT"