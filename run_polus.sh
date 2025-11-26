#!/bin/bash
# Запуск MPI-версии на IBM Polus через mpisubmit.pl
# Требования: N = 128/256/512, L=1 и L=π, np = 1,4,8,16,32

set -e

if [ ! -f ./wave3d_mpi ]; then
    echo "Error: ./wave3d_mpi not found. Run 'make compile_polus' first."
    exit 1
fi

declare -a procs=(1 4 8 16 32)

for N in 128 256 512; do
    for type in "1.0_1.0_1.0" "pi_pi_pi"; do
        if [[ "$type" == "1.0_1.0_1.0" ]]; then
            L_ARGS=("1.0" "1.0" "1.0")
            L_LABEL="1"
        else
            L_ARGS=("pi" "pi" "pi")
            L_LABEL="pi"
        fi

        for np in "${procs[@]}"; do
            OUT_FILE="stats_mpi_${N}_${np}_${L_LABEL}.out"
            ERR_FILE="stats_mpi_${N}_${np}_${L_LABEL}.err"
            JOB_NAME="mpi_job_${N}_${np}_${L_LABEL}"

            echo "Submitting: N=$N, np=$np, L=$L_LABEL"

            mpisubmit.pl \
                -p "$np" \
                -w 00:30 \
                --stdout "$OUT_FILE" \
                --stderr "$ERR_FILE" \
                --jobname "$JOB_NAME" \
                ./wave3d_mpi "$N" "${L_ARGS[@]}"
        done
    done
done

echo "All MPI jobs submitted to Polus via mpisubmit.pl"