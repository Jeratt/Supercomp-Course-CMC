#!/bin/bash

set -e

if [ ! -f ./wave3d_mpi ]; then
    echo "Error: ./wave3d_mpi not found. Run 'make compile_polus' first."
    exit 1
fi

declare -a procs=(1 4 8 16 32)

for N in 128 256 512; do
    for type in "1.0_1.0_1.0" "pi_pi_pi"; do
        if [[ "$type" == "1.0_1.0_1.0" ]]; then
            L_ARGS="1.0 1.0 1.0"
            L_LABEL="1"
        else
            L_ARGS="pi pi pi"
            L_LABEL="pi"
        fi

        for np in "${procs[@]}"; do
            JOB_NAME="mpi_job_${N}_${np}_${L_LABEL}"
            OUT_FILE="stats_mpi_${N}_${np}_${L_LABEL}.out"
            ERR_FILE="stats_mpi_${N}_${np}_${L_LABEL}.err"

            echo "Submitting: N=$N, np=$np, L=$L_LABEL"
            bsub -n $np \
                 -q short \
                 -W 00:30 \
                 -J "$JOB_NAME" \
                 -o "$OUT_FILE" \
                 -e "$ERR_FILE" \
                 -R "span[ptile=32]" \
                 mpirun -np $np ./wave3d_mpi $N $L_ARGS
        done
    done
done

echo "All MPI jobs submitted to Polus."