#!/bin/bash

date >> results.txt

for i in {0..3}; do
    for hierarchical in 0 1; do
        for pack_gru in 0 1; do
            for connect_hidden in 0 1; do
                for aggregate_mode in attention last_hidden average max; do
                    export LOAD_CHECKPOINT=0
                    export REMARK=${i}_hierarchical_${hierarchical}_pack_gru_${pack_gru}_connect_hidden_${connect_hidden}_aggregate_mode_${aggregate_mode}
                    export HIERARCHICAL=$hierarchical
                    export PACK_GRU=$pack_gru
                    export CONNECT_HIDDEN=$connect_hidden
                    export AGGREGATE_MODE=$aggregate_mode
                    echo $REMARK >> results.txt
                    python3 src/train.py
                    python3 src/evaluate.py >> results.txt
                    echo "" >> results.txt
                done
            done
        done
    done
done
