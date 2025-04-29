#!/bin/bash

python jajapy_part.py "$PRISM_FILE" "$ITERATIONS" "$EPSILON"
./CuPAAL -m ../results/cupaal_model.txt -s ../results/cupaal_training_set.txt -o ../results/cupaal_model_learned.txt -r ../results/cupaal_results.csv -i "$ITERATIONS" -e "$EPSILON" -t "$TIME"
