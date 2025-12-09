#define _USE_MATH_DEFINES
#include "equation.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <mpi.h>
#include <omp.h>
#include <algorithm>
using namespace std;

double parse_length(const string& arg, string& label_part) {
    // Приводим к нижнему регистру для надежности сравнения
    string lower_arg = arg;
    transform(lower_arg.begin(), lower_arg.end(), lower_arg.begin(), ::tolower);
    
    if (lower_arg == "pi") {
        label_part = "pi";
        return M_PI;
    } else {
        label_part = arg;
        try {
            return stod(arg);
        } catch (...) {
            cerr << "Invalid length parameter: " << arg << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1.0; // never reached
        }
    }
}

void determine_dimensions(int proc_num, int dims[3]) {
    // Для варианта 3 приоритет отдается разбиению по Y (периодическое направление)
    dims[0] = 1; // X dimension
    dims[1] = 1; // Y dimension (periodic)
    dims[2] = 1; // Z dimension
    
    // Находим оптимальное разбиение с приоритетом по Y
    int best_diff = 1000;
    int best_dims[3] = {1, 1, proc_num}; // По умолчанию все процессы в Z
    
    for (int i = 1; i <= proc_num; ++i) {
        if (proc_num % i != 0) continue;
        int remaining = proc_num / i;
        for (int j = 1; j <= remaining; ++j) {
            if (remaining % j != 0) continue;
            int k = remaining / j;
            
            int current_diff = abs(i - 1) + abs(k - 1); // Стремимся к балансу по X и Z
            if (current_diff < best_diff) {
                best_diff = current_diff;
                best_dims[0] = i;
                best_dims[1] = j;
                best_dims[2] = k;
            }
        }
    }
    
    dims[0] = best_dims[0];
    dims[1] = best_dims[1];
    dims[2] = best_dims[2];
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, np;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    
    // argc должно быть равно 7: имя программы, N, num_threads, type, Lx, Ly, Lz
    if (argc != 6) {
        if (rank == 0) {
            cerr << "Usage: mpirun -np NPROC ./wave3d_combo N num_threads Lx Ly Lz" << endl
                 << "  Lx, Ly, Lz: numbers (e.g. 1.0) or 'pi'" << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    int N = stoi(argv[1]);
    if (N <= 0) {
        if (rank == 0) cerr << "N must be positive." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Устанавливаем число потоков OpenMP из аргумента командной строки
    int num_threads = stoi(argv[2]);
    if (num_threads <= 0) {
        if (rank == 0) cerr << "Number of threads must be positive." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    omp_set_num_threads(num_threads);
    
    string lx_label, ly_label, lz_label;
    double Lx = parse_length(argv[3], lx_label);
    double Ly = parse_length(argv[4], ly_label);
    double Lz = parse_length(argv[5], lz_label);
    
    string domain_label = lx_label + "_" + ly_label + "_" + lz_label;
    Grid grid(N, Lx, Ly, Lz, domain_label);
    
    if (rank == 0) {
        cout << "MPI+OpenMP run: np=" << np << "  OMP threads=" << num_threads << endl;
        cout << "  N = " << grid.N << endl
             << "  Lx = " << grid.Lx << endl
             << "  Ly = " << grid.Ly << endl
             << "  Lz = " << grid.Lz << endl
             << "  Domain label = " << grid.domain_label << endl;
    }
    
    int dims[3];
    determine_dimensions(np, dims);
    int periods[3] = {0, 1, 0}; // Периодичность только по Y
    if (rank == 0) {
        cout << "Chosen 3D topology: (" << dims[0] << ", " << dims[1] << ", " << dims[2] << ")" << endl
             << "Periodicity: x=" << periods[0] << ", y=" << periods[1] << ", z=" << periods[2] << endl;
    }
    
    MPI_Comm comm_cart;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, true, &comm_cart);
    int coords[3];
    MPI_Cart_coords(comm_cart, rank, 3, coords);
    
    vector<int> neighbors(6, -1);
    MPI_Cart_shift(comm_cart, 0, -1, &neighbors[0], &neighbors[1]); // x- / x+
    MPI_Cart_shift(comm_cart, 1, -1, &neighbors[2], &neighbors[3]); // y- / y+
    MPI_Cart_shift(comm_cart, 2, -1, &neighbors[4], &neighbors[5]); // z- / z+
    
    Block block(grid, neighbors, coords, dims[0], dims[1], dims[2], rank);
    
    VDOUB result;
    double time = 0, max_inacc = 0, first_inacc = 0, last_inacc = 0;
    solve_mpi(grid, block, dims[0], dims[1], dims[2], comm_cart,
              time, max_inacc, first_inacc, last_inacc, result);
    
    if (rank == 0) {
        cout << "=== RESULT ===" << endl
             << "Time: " << time << " s" << endl
             << "Max inaccuracy: " << max_inacc << endl
             << "First step inaccuracy: " << first_inacc << endl
             << "Last step inaccuracy: " << last_inacc << endl;
    }
    
    MPI_Finalize();
    return 0;
}