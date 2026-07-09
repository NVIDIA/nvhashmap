#!/bin/bash

BENCH_CMD="numactl -N 0 -m 0 ./tools/benchmark/benchmark"

NUM_KEYS=50000000
NUM_TRIALS=10

# Fastest Insert on Grace
$BENCH_CMD --num_workers 1\
  --insert_queue_type shift --max_insert_queue_len 3\
  --find_keep_perc 100 --find_queue_type ring --max_find_queue_len 0\
  --map_type nvhm_map --kernel_type default

# Fastest Find on Grace
$BENCH_CMD --num_workers 1 --num_keys $NUM_KEYS\
  --insert_queue_type shift --max_insert_queue_len 0\
  --find_keep_perc 100 --find_queue_type ring --max_find_queue_len 8\
  --map_type nvhm_map --kernel_type default
$BENCH_CMD --num_workers 1 --num_keys $NUM_KEYS\
  --insert_queue_type shift --max_insert_queue_len 0\
  --find_keep_perc 50 --find_queue_type ring --max_find_queue_len 12\
  --map_type nvhm_map --kernel_type default
$BENCH_CMD --num_workers 1 --num_keys $NUM_KEYS\
   --insert_queue_type shift --max_insert_queue_len 0\
   --find_keep_perc 10 --find_queue_type ring --max_find_queue_len 24\
   --map_type nvhm_map --kernel_type default
$BENCH_CMD --num_workers 1 --num_keys $NUM_KEYS\
  --insert_queue_type shift --max_insert_queue_len 0\
  --find_keep_perc 1 --find_queue_type ring --max_find_queue_len 48\
  --map_type nvhm_map --kernel_type default

$BENCH_CMD --num_workers 8 --num_keys $NUM_KEYS\
  --insert_queue_type shift --max_insert_queue_len 0\
  --find_keep_perc 100 --find_queue_type ring --max_find_queue_len 4\
  --map_type nvhm_map --kernel_type default
$BENCH_CMD --num_workers 8 --num_keys $NUM_KEYS\
  --insert_queue_type shift --max_insert_queue_len 0\
  --find_keep_perc 50 --find_queue_type ring --max_find_queue_len 8\
  --map_type nvhm_map --kernel_type default
$BENCH_CMD --num_workers 8 --num_keys $NUM_KEYS\
   --insert_queue_type shift --max_insert_queue_len 0\
   --find_keep_perc 10 --find_queue_type ring --max_find_queue_len 16\
   --map_type nvhm_map --kernel_type default
$BENCH_CMD --num_workers 8 --num_keys $NUM_KEYS\
  --insert_queue_type shift --max_insert_queue_len 0\
  --find_keep_perc 1 --find_queue_type ring --max_find_queue_len 0\
  --map_type nvhm_map --kernel_type default

for NUM_WORKERS in 1 8; do
  for FIND_KEEP_PERC in 100 50 10 1; do
    for MAP_TYPE in "nvhm_std_map_shim" "std_unordered_map" "absl_flat_hash_map" "folly_f14_value_map" "phmap_flat_hash_map"; do
    $BENCH_CMD --num_trials $NUM_TRIALS --num_workers $NUM_WORKERS --num_keys $NUM_KEYS\
      --insert_queue_type shift --max_insert_queue_len 0\
      --find_keep_perc $FIND_KEEP_PERC --find_queue_type ring --max_find_queue_len 0\
      --map_type $MAP_TYPE --kernel_type default
    done
  done
done
