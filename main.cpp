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

void determine_dimensions(int proc_num, int dims[3]) {
    // Простой подход: разбиваем на кубические или близкие к кубическим блоки
    dims[0] = 1;
    dims[1] = 1;
    dims[2] = 1;
    
    // Находим множители, стараясь получить примерно равные размеры
    for (int i = static_cast<int>(cbrt(proc_num)); i >= 1; --i) {
        if (proc_num % i == 0) {
            dims[0] = i;
            int remaining = proc_num / i;
            
            // Находим множители для оставшегося числа
            for (int j = static_cast<int>(sqrt(remaining)); j >= 1; --j) {
                if (remaining % j == 0) {
                    dims[1] = j;
                    dims[2] = remaining / j;
                    break;
                }
            }
            break;
        }
    }
    
    // Гарантируем, что произведение равно числу процессов
    if (dims[0] * dims[1] * dims[2] != proc_num) {
        dims[0] = 1;
        dims[1] = 1;
        dims[2] = proc_num;
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, np;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    
    if (argc != 5) {
        if (rank == 0) {
            cerr << "Usage: mpirun -np NPROC ./wave3d_mpi N Lx Ly Lz" << endl
                 << "  Lx, Ly, Lz: numbers (e.g. 1.0) or 'pi'" << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    int N = stoi(argv[1]);
    if (N <= 0) {
        if (rank == 0) cerr << "N must be positive." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    string lx_label, ly_label, lz_label;
    double Lx = parse_length(argv[2], lx_label);
    double Ly = parse_length(argv[3], ly_label);
    double Lz = parse_length(argv[4], lz_label);
    string domain_label = lx_label + "_" + ly_label + "_" + lz_label;
    Grid grid(N, Lx, Ly, Lz, domain_label);
    
    if (rank == 0) {
        cout << "MPI run: np=" << np << endl
             << "  N = " << grid.N << endl
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