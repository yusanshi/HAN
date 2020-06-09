#!/bin/bash

date >> results.txt
echo "" >> results.txt

for i in {1..3}; do
    for hierarchical in 0 1; do
        for pack_gru in 0 1; do
            for connect_hidden in 0 1; do
                for aggregate_mode in attention last_hidden average max; do
                    export REMARK=round_${i}-hierarchical_${hierarchical}-pack_gru_${pack_gru}-connect_hidden_${connect_hidden}-aggregate_mode_${aggregate_mode}
                    export HIERARCHICAL=$hierarchical
                    export PACK_GRU=$pack_gru
                    export CONNECT_HIDDEN=$connect_hidden
                    export AGGREGATE_MODE=$aggregate_mode
                    echo $REMARK >> results.txt
                    rm -rf checkpoint/*
                    python3 src/train.py
                    python3 src/evaluate.py >> results.txt
                    echo "" >> results.txt
                done
            done
        done
    done
done
