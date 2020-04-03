#!/usr/bin/env bash

python ../train.py \
--job-dir=./jobs \
--train-csv-path=path-to-train-cs \
--valid-csv-path=path-to-valid-csv \
--config-file=./config.json \
--device=0 \
--num-epochs=1200 \
--random-seeds="1"



