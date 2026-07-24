#!/bin/bash

BENCH_CMD="numactl -N 0 -m 0 ./tools/benchmark/mutex_benchmark"

MUTEX_TYPE="nvhm_spin_wait_mutex"
PRINT_HEADER=1

for scenario in "high_contention" "low_contention" "read_mostly"; do
  for use_wait in true false; do
    for spin_count in 0 1 2 4 8 16 32 64 128; do
      $BENCH_CMD --scenario $scenario --duration_ms 1000 --num_workers 1,2,4,8,16,32,64,128 --mutex_type $MUTEX_TYPE --use_wait $use_wait --spin_count $spin_count --print_header $PRINT_HEADER
      PRINT_HEADER=0
    done
  done
done
