#!/bin/bash

cupaal_postfix="_cupaal_run"
models=(oscillators1 oscillators2 oscillators3)

#for model in "${models[@]}"
#do
#  for i in {0..9}
#  do
#    curl -s --create-dirs -o "experiments/$model/$model$cupaal_postfix$i.txt" -L "https://raw.githubusercontent.com/AAU-Dat/P10-Thesis/refs/heads/experiments-models/experiments/initial-models/$model$cupaal_postfix$i.txt" &
#  done
#  curl -s --create-dirs -o "experiments/$model/observations.txt" -L "https://raw.githubusercontent.com/AAU-Dat/P10-Thesis/refs/heads/experiments-models/experiments/observations/$model""_observations_cupaal.txt" &
#done
#echo "Downloading experiment files"
#wait

for model in "${models[@]}"
do
#  start=$EPOCHREALTIME
  for i in {0..9}
  do
    ./cmake-build-release/CuPAAL -m "experiments/$model/$model$cupaal_postfix$i.txt" \
    -s "experiments/$model/observations.txt" \
    -o "experiments/$model/$model$cupaal_postfix$i""_bw.txt" \
    -r "experiments/$model/$model$cupaal_postfix$i""_bw_result.csv"
  done
#  stop=$EPOCHREALTIME
#  echo "Time spent on the $model model: $(bc -l <<< "$stop - $start")"
done
