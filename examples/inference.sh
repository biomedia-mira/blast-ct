#!/usr/bin/env bash

python ./inference.py \
--job-dir=./jobs/inference_job \
--test-csv-path=path-to-test-csv \
--config-file=./congfig_file.json \
--device=0 \
--saved-model-paths="./saved_models/model_1.pt
./saved_models/model_2.pt
./saved_models/model_3.pt
./saved_models/model_4.pt
./saved_models/model_5.pt
./saved_models/model_6.pt
./saved_models/model_7.pt
./saved_models/model_8.pt
./saved_models/model_9.pt
./saved_models/model_10.pt
./saved_models/model_11.pt
./saved_models/model_12.pt"
