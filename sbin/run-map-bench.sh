#!/bin/bash

BENCH_CMD="numactl -N 0 -m 0 ./tools/benchmark/map_benchmark"

NUM_KEYS=5000000
KEY_SOURCE=polynomial
KEY_POLY=13,3,7
SEED=1337
PRINT_HEADER=1

KEY_SETUP="--num_keys $NUM_KEYS --key_source $KEY_SOURCE --key_poly $KEY_POLY --seed $SEED"
MAP_SETUP="--print_header $PRINT_HEADER"


# Insert
NUM_INSERT_TRIALS=5
NUM_FIND_TRIALS=0

SETUP="$KEY_SETUP"
SETUP="$SETUP --num_insert_trials $NUM_INSERT_TRIALS --num_find_trials $NUM_FIND_TRIALS"
SETUP="$SETUP --insert_queue_type shift --max_find_queue_len 0"

$BENCH_CMD --map_type "nvhm_map" --num_workers 1 $SETUP\
  --max_insert_queue_len 3 --print_header $PRINT_HEADER
PRINT_HEADER=0

for map_type in "nvhm_std_map_shim" "std_unordered_map" "absl_flat_hash_map" "folly_f14_value_map" "phmap_flat_hash_map"; do
  $BENCH_CMD --map_type $map_type --num_workers 1 $SETUP\
    --max_insert_queue_len 0 --print_header $PRINT_HEADER
  PRINT_HEADER=0
done


# Find
NUM_INSERT_TRIALS=0
NUM_FIND_TRIALS=5

SETUP="$KEY_SETUP"
SETUP="$SETUP --num_insert_trials $NUM_INSERT_TRIALS --num_find_trials $NUM_FIND_TRIALS"
SETUP="$SETUP --min_insert_queue_len 0 --max_insert_queue_len 0"
SETUP="$SETUP --find_queue_type ring"

# 1 worker.
for perc_qlen in "100 6" "50 12" "10 24" "1 28"; do
  set -- $perc_qlen
  perc=$1
  qlen=$2
  $BENCH_CMD --map_type nvhm_map --num_workers 1 $SETUP\
    --find_hit_perc $perc --max_find_queue_len $qlen --print_header $PRINT_HEADER
  PRINT_HEADER=0

  for map_type in "nvhm_std_map_shim" "std_unordered_map" "absl_flat_hash_map" "folly_f14_value_map" "phmap_flat_hash_map"; do
    $BENCH_CMD --map_type $map_type --num_workers 1 $SETUP\
      --find_hit_perc $perc --max_find_queue_len 0 --print_header $PRINT_HEADER
    PRINT_HEADER=0
  done
done

# 8 worker.
for perc_qlen in "100 4" "50 6" "10 16" "1 20"; do
  set -- $perc_qlen
  perc=$1
  qlen=$2
  $BENCH_CMD --map_type nvhm_map --num_workers 8 $SETUP\
    --find_hit_perc $perc --max_find_queue_len $qlen --print_header $PRINT_HEADER
  PRINT_HEADER=0

  for map_type in "nvhm_std_map_shim" "std_unordered_map" "absl_flat_hash_map" "folly_f14_value_map" "phmap_flat_hash_map"; do
    $BENCH_CMD --map_type $map_type --num_workers 8 $SETUP\
      --find_hit_perc $perc --max_find_queue_len 0 --print_header $PRINT_HEADER
    PRINT_HEADER=0
  done
done
