#define _USE_MATH_DEFINES
#include "equation.hpp"
#include <iostream>
#include <fstream>
#include <cstring>
#include <cmath>
#include <mpi.h>
using namespace std;

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
    
    int N = atoi(argv[1]);
    if (N <= 0) {
        if (rank == 0) cerr << "N must be positive.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Parse Lx, Ly, Lz
    double Lx, Ly, Lz;
    char L_type_str[10];
    
    if (strcmp(argv[2], "pi") == 0 && strcmp(argv[3], "pi") == 0 && strcmp(argv[4], "pi") == 0) {
        Lx = Ly = Lz = M_PI;
        strcpy(L_type_str, "pi");
    } else {
        Lx = atof(argv[2]);
        Ly = atof(argv[3]);
        Lz = atof(argv[4]);
        strcpy(L_type_str, "1.0");
    }
    
    Grid grid(N, L_type_str, Lx, Ly, Lz);
    
    if (rank == 0) {
        cout << "Input values:\n"
             << "\tN = " << grid.N << "\n"
             << "\tProcesses = " << np << "\n"
             << "\tL_type = " << grid.L_type << "\n"
             << "\tLx = " << grid.Lx << "\n"
             << "\tLy = " << grid.Ly << "\n"
             << "\tLz = " << grid.Lz << endl;
    }
    
    // Create cartesian topology
    int dims[3] = {0, 0, 0};
    MPI_Dims_create(np, 3, dims);
    
    // Periodicity: only along Y axis (index 1) for variant 3 (1P 1R 1P)
    int periods[3] = {0, 1, 0};
    
    if (rank == 0) {
        cout << "Dims topology: (" << dims[0] << ", " << dims[1] << ", " << dims[2] << ")\n"
             << "Periodicity: x=" << periods[0] << ", y=" << periods[1] << ", z=" << periods[2] << endl;
    }
    
    // Create cartesian communicator
    MPI_Comm comm_cart;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, true, &comm_cart);
    
    // Get coordinates in cartesian topology
    int coords[3];
    MPI_Cart_coords(comm_cart, rank, 3, coords);
    
    // Find neighbors in 6 directions
    VINT neighbors(6);
    int dummy;
    
    // X direction neighbors (Dirichlet)
    MPI_Cart_shift(comm_cart, 0, -1, &neighbors[0], &dummy); // left (x-)
    MPI_Cart_shift(comm_cart, 0, +1, &dummy, &neighbors[1]); // right (x+)
    
    // Y direction neighbors (Periodic)
    MPI_Cart_shift(comm_cart, 1, -1, &neighbors[2], &dummy); // bottom (y-)
    MPI_Cart_shift(comm_cart, 1, +1, &dummy, &neighbors[3]); // top (y+)
    
    // Z direction neighbors (Dirichlet)
    MPI_Cart_shift(comm_cart, 2, -1, &neighbors[4], &dummy); // front (z-)
    MPI_Cart_shift(comm_cart, 2, +1, &dummy, &neighbors[5]); // back (z+)
    
    // Replace MPI_PROC_NULL with -1 for compatibility
    for (int i = 0; i < 6; i++) {
        if (neighbors[i] == MPI_PROC_NULL)
            neighbors[i] = MPI_PROC_NULL;
    }
    
    // Create block for current process
    Block block(grid, neighbors, coords, dims[0], dims[1], dims[2], rank);
    
    // Result vectors and metrics
    VDOUB result_vec;
    double time = 0.0;
    double max_inacc = 0.0;
    double first_inacc = 0.0;
    double last_inacc = 0.0;
    
    // Solve the problem
    solve_mpi(grid, block, dims[0], dims[1], dims[2], comm_cart,
              time, max_inacc, first_inacc, last_inacc, result_vec);
    
    // Output results
    if (rank == 0) {
        cout << "Result:\n"
             << "\tTime = " << time << "\n"
             << "\tMax inaccuracy = " << max_inacc << "\n"
             << "\tFirst step inaccuracy = " << first_inacc << "\n"
             << "\tLast step inaccuracy = " << last_inacc << endl;
    }
    
    MPI_Finalize();
    return 0;
}