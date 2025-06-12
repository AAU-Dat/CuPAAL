#!/bin/bash

PRISM_FILE_NAME="${PRISM_FILE_PATH##*/}"
SAVE_PATH="../results/${PRISM_FILE_NAME}_${OBSERVATION_COUNT}-${OBSERVATION_LENGTH}"

python jajapy_part.py "$PRISM_FILE_PATH" "$SAVE_PATH" "$ITERATIONS" "$EPSILON" "$OBSERVATION_COUNT" "$OBSERVATION_LENGTH"
./CuPAAL -m "$SAVE_PATH"_cupaal_model.txt -s "$SAVE_PATH"_cupaal_training_set.txt -o "$SAVE_PATH"_cupaal_model_learned.txt -r "$SAVE_PATH"_cupaal_results.csv -i "$ITERATIONS" -e "$EPSILON" -t "$TIME"
./CuPAAL -m "$SAVE_PATH"_cupaal_model-semi.txt -s "$SAVE_PATH"_cupaal_training_set.txt -o "$SAVE_PATH"_cupaal_model_learned-semi.txt -r "$SAVE_PATH"_cupaal_results-semi.csv -i "$ITERATIONS" -e "$EPSILON" -t "$TIME"
