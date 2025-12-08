#!/bin/bash
# Запуск гибридной MPI+OpenMP версии на IBM Polus через mpisubmit.pl
# Требуемые конфигурации согласно Таблице 4 из задания:
# - Число MPI процессов: 4, 8
# - Число OpenMP нитей: 1, 2, 4, 8
# - Размеры сетки: 128^3, 256^3
# - Типы области: L=1 и L=pi

set -e

if [ ! -f ./wave3d_combo ]; then
    echo "Error: ./equation not found. Run 'make compile_polus' first."
    exit 1
fi

# Конфигурации для запуска согласно Таблице 4
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
                
                # Формируем имя файла для вывода
                OUT_FILE="stats_mpi_omp_${N}_${np}_${nt}_${type}.out"
                ERR_FILE="stats_mpi_omp_${N}_${np}_${nt}_${type}.err"
                
                echo "Submitting: N=$N, MPI processes=$np, OpenMP threads=$nt, L=$type"
                
                # Запускаем задачу через mpisubmit.pl
                mpisubmit.pl \
                    -p "$np" \
                    -t "$nt" \
                    -w 00:30 \
                    --stdout "$OUT_FILE" \
                    --stderr "$ERR_FILE" \
                    ./equation "$N" "$nt" "$type" "${L_ARGS[@]}"
            done
        done
    done
done

echo "All MPI+OpenMP jobs submitted to Polus via mpisubmit.pl"