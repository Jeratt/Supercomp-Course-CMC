#!/bin/bash

set -e

if [ ! -f ./wave3d ]; then
    echo "Error: ./wave3d not found. Run make first."
    exit 1
fi

for N in 128 256 512; do
    for type in "1.0_1.0_1.0" "pi_pi_pi"; do
        if [[ "$type" == "1.0_1.0_1.0" ]]; then
            L_ARGS="1.0 1.0 1.0"
            L_LABEL="1"
        else
            L_ARGS="pi pi pi"
            L_LABEL="pi"
        fi

        for threads in 1 2 4 8 16 32; do
            JOB_NAME="omp_job_${N}_${threads}_${L_LABEL}"
            OUT_FILE="stats_${N}_${threads}_${L_LABEL}.out"
            ERR_FILE="stats_${N}_${threads}_${L_LABEL}.err"

            echo "Submitting: N=$N, threads=$threads, L=$L_LABEL"
            bsub -n 1 -q short -W 00:30 -J "$JOB_NAME" -o "$OUT_FILE" -e "$ERR_FILE" \
                 -R "affinity[core(10,same=socket,exclusive=(socket,alljobs)):membind=localonly:distribute=pack(socket=1)]" \
                 ./wave3d $N $threads $L_ARGS
        done
    done
done

echo "All Polus jobs submitted."