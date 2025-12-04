#define _USE_MATH_DEFINES
#include "equation.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <mpi.h>
using namespace std;

double parse_length(const string& arg, string& label_part) {
    if (arg == "pi") {
        label_part = "pi";
        return M_PI;
    } else {
        label_part = arg;
        return stod(arg);
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, np;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    
    if (argc != 5) {
        if (rank == 0) {
            cerr << "Usage: mpirun -np NPROC ./wave3d_mpi N Lx Ly Lz\n"
                 << "  Lx, Ly, Lz: numbers (e.g. 1.0) or 'pi'\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    int N = stoi(argv[1]);
    if (N <= 0) {
        if (rank == 0) cerr << "N must be positive.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    string lx_label, ly_label, lz_label;
    double Lx = parse_length(argv[2], lx_label);
    double Ly = parse_length(argv[3], ly_label);
    double Lz = parse_length(argv[4], lz_label);
    string domain_label = lx_label + "_" + ly_label + "_" + lz_label;
    
    Grid grid(N, Lx, Ly, Lz, domain_label);
    
    if (rank == 0) {
        cout << "MPI run: np=" << np << "\n"
             << "  N = " << grid.N << "\n"
             << "  Lx = " << grid.Lx << "\n"
             << "  Ly = " << grid.Ly << "\n"
             << "  Lz = " << grid.Lz << "\n"
             << "  Domain label = " << grid.domain_label << endl;
    }
    
    // Автоматическое создание сбалансированной топологии
    int dims[3] = {0, 0, 0};
    MPI_Dims_create(np, 3, dims);
    
    // Периодичность: только по оси y (индекс 1)
    int periods[3] = {0, 1, 0};
    
    if (rank == 0) {
        cout << "Chosen 3D topology: (" << dims[0] << ", " << dims[1] << ", " << dims[2] << ")\n"
             << "Periodicity: x=" << periods[0] << ", y=" << periods[1] << ", z=" << periods[2] << endl;
    }
    
    // Создание декартова коммуникатора
    MPI_Comm comm_cart;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, true, &comm_cart);
    
    // Определение координат процесса в декартовой топологии
    int coords[3];
    MPI_Cart_coords(comm_cart, rank, 3, coords);
    
    // Определение соседей для обмена гало-ячейками
    int neighbors[6];
    int dummy;
    
    // X direction neighbors
    MPI_Cart_shift(comm_cart, 0, -1, &neighbors[0], &dummy);
    MPI_Cart_shift(comm_cart, 0, 1, &dummy, &neighbors[1]);
    
    // Y direction neighbors (periodic)
    MPI_Cart_shift(comm_cart, 1, -1, &neighbors[2], &dummy);
    MPI_Cart_shift(comm_cart, 1, 1, &dummy, &neighbors[3]);
    
    // Z direction neighbors
    MPI_Cart_shift(comm_cart, 2, -1, &neighbors[4], &dummy);
    MPI_Cart_shift(comm_cart, 2, 1, &dummy, &neighbors[5]);
    
    // Преобразуем в vector<int> и заменяем MPI_PROC_NULL на -1
    vector<int> neighbors_vec(6);
    for (int i = 0; i < 6; ++i) {
        if (neighbors[i] == MPI_PROC_NULL) 
            neighbors_vec[i] = MPI_PROC_NULL;
        else 
            neighbors_vec[i] = neighbors[i];
    }
    
    // Создание блока данных для текущего процесса
    Block block(grid, neighbors_vec, coords, dims[0], dims[1], dims[2], rank);
    
    // Результаты
    VDOUB result;
    double time = 0, max_inacc = 0, first_inacc = 0, last_inacc = 0;
    
    // Решение задачи
    solve_mpi(grid, block, dims[0], dims[1], dims[2], comm_cart,
              time, max_inacc, first_inacc, last_inacc, result);
    
    if (rank == 0) {
        cout << "=== RESULT ===\n"
             << "Time: " << time << " s\n"
             << "Max inaccuracy: " << max_inacc << "\n"
             << "First step inaccuracy: " << first_inacc << "\n"
             << "Last step inaccuracy: " << last_inacc << endl;
    }
    
    MPI_Finalize();
    return 0;
}