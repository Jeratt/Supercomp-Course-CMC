#!/bin/bash

set -e

if [ ! -f ./wave3d_combo ]; then
    echo "Error: ./wave3d_combo not found. Run 'make' first."
    exit 1
fi

declare -a mpi_procs=(4 8)
declare -a omp_threads=(1 2 4 8)
declare -a grid_sizes=(128 256)
declare -a domain_types=("1" "pi")

for N in "${grid_sizes[@]}"; do
    for np in "${mpi_procs[@]}"; do
        for nt in "${omp_threads[@]}"; do
            for type in "${domain_types[@]}"; do
                if [ "$type" == "1" ]; then
                    L_ARGS=("1.0" "1.0" "1.0")
                else
                    L_ARGS=("pi" "pi" "pi")
                fi
                
                OUT_FILE="stats_mpi_omp_${N}_${np}_${nt}_${type}.out"
                ERR_FILE="stats_mpi_omp_${N}_${np}_${nt}_${type}.err"
                
                echo "Submitting: N=$N, MPI processes=$np, OpenMP threads=$nt, L=$type"
                
                mpisubmit.pl \
                    -p "$np" \
                    -t "$nt" \
                    -w 00:30 \
                    --stdout "$OUT_FILE" \
                    --stderr "$ERR_FILE" \
                    ./wave3d_combo "$N" "$nt" "${L_ARGS[@]}"
            done
        done
    done
done

echo "All MPI+OpenMP jobs submitted to Polus via mpisubmit.pl"