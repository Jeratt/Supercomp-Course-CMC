#define _USE_MATH_DEFINES
#include "equation.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <mpi.h>

#define INACCURACY "INACCURACY"
#define NUMERICAL "NUMERICAL"
#define ANALITICAL "ANALITICAL"

void dump_block_to_CSV(const Grid& g, const Block& b, const VDOUB& u_local, const std::string& filename, std::string matrix_type) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file '" << filename << "' for writing." << std::endl;
        return;
    }
    
    for (int i = 1; i < b.Nx + 1; ++i) {
        double global_x = (b.x_start + i - 1) * g.h_x;
        for (int j = 1; j < b.Ny + 1; ++j) {
            double global_y = (b.y_start + j - 1) * g.h_y;
            for (int k = 1; k < b.Nz + 1; ++k) {
                double global_z = (b.z_start + k - 1) * g.h_z;
                double analytical_value = u_analytical(g, global_x, global_y, global_z, (TIME_STEPS - 1) * g.tau);
                
                if (matrix_type == NUMERICAL)
                    file << u_local[b.index(i, j, k)];
                else if (matrix_type == ANALITICAL)
                    file << analytical_value;
                else if (matrix_type == INACCURACY)
                    file << fabs(u_local[b.index(i, j, k)] - analytical_value);
                else
                    std::cerr << "Unknown matrix type: " << matrix_type << std::endl;
                
                if (k < b.Nz)
                    file << ",";
            }
            file << "\n";
        }
        if (j < b.Ny)
            file << "\n";
    }
    file.close();
}

void save_statistics(const Grid& g, double& time, double& max_inaccuracy, double& first_step_inaccuracy, double& last_step_inaccuracy, int& proc_num) {
    std::string filename = "results/statistics/" + std::to_string(g.N) + "_" + std::to_string(proc_num) + "_" + g.L_type + "_" + "statistics.txt";
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file '" << filename << "' for writing." << std::endl;
        return;
    }
    file << "Time = " << time << "\n" 
         << "Max inaccuracy = " << max_inaccuracy << "\n" 
         << "First step inaccuracy = " << first_step_inaccuracy << "\n" 
         << "Last step inaccuracy = " << last_step_inaccuracy << std::endl;
}

// Function to determine dimensions for cartesian topology
void determine_dimensions(int proc_num, int dims[3]) {
    // For variant 3, we want to prioritize splitting along Y dimension (periodic)
    dims[0] = 1; // Y dimension (periodic)
    dims[1] = 1; // X dimension (Dirichlet)
    dims[2] = 1; // Z dimension (Dirichlet)
    
    // Try to find factors of proc_num
    int remaining = proc_num;
    
    // First, try to set Y dimension (periodic) - make it as large as possible
    for (int i = static_cast<int>(sqrt(proc_num)); i >= 1; --i) {
        if (proc_num % i == 0) {
            dims[0] = i;
            remaining = proc_num / i;
            break;
        }
    }
    
    // Then, try to set X dimension
    for (int i = static_cast<int>(sqrt(remaining)); i >= 1; --i) {
        if (remaining % i == 0) {
            dims[1] = i;
            dims[2] = remaining / i;
            break;
        }
    }
    
    // Ensure the product equals proc_num
    if (dims[0] * dims[1] * dims[2] != proc_num) {
        dims[0] = 1;
        dims[1] = 1;
        dims[2] = proc_num;
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, proc_num;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
    
    if ((!strcmp(argv[2], "custom") && argc != 6) || (strcmp(argv[2], "custom") && argc != 3)) {
        if (rank == 0)
            std::cerr << "Invalid number of arguments. You must specify 3 or 6.\n" 
                      << "Syntax: N, L_type (1 -> Lx=Ly=Lz=1, pi -> Lx=Ly=Lz=Pi, custom -> specify 3 extra values for Lx, Ly, Lz), [Lx, Ly, Lz]" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    int N = atoi(argv[1]);
    if (N < 0) {
        if (rank == 0)
            std::cerr << "Invalid N: must be > 0" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    Grid grid = Grid();
    if (!strcmp(argv[2], "pi"))
        grid = Grid(N, argv[2], M_PI);
    else if (!strcmp(argv[2], "1"))
        grid = Grid(N, argv[2], 1.0);
    else if (!strcmp(argv[2], "custom"))
        grid = Grid(N, argv[2], atof(argv[3]), atof(argv[4]), atof(argv[5]));
    else {
        if (rank == 0)
            std::cerr << "Invalid L_type: must be '1', 'pi' or 'custom'" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    if (rank == 0)
        std::cout << "Input values:\n\tN = " << grid.N << "\n\tProcesses = " << proc_num << "\n\tL_type = " << grid.L_type <<
            "\n\tLx = " << grid.Lx << "\n\tLy = " << grid.Ly << "\n\tLz = " << grid.Lz << std::endl;
    
    // Create cartesian topology
    int dims[3] = {0, 0, 0};
    determine_dimensions(proc_num, dims);
    
    if (rank == 0)
        std::cout << "Dims topology: (" << dims[0] << ", " << dims[1] << ", " << dims[2] << ")" << std::endl;
    
    // Periodicity: only Y dimension has periodic boundary conditions
    int periods[3] = {1, 0, 0}; // Y is periodic, X and Z are not
    
    MPI_Comm comm_cart;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &comm_cart);
    
    int coords[3];
    MPI_Cart_coords(comm_cart, rank, 3, coords);
    
    // Neighbors in 6 directions
    VINT neighbors(6);
    MPI_Cart_shift(comm_cart, 0, 1, &neighbors[0], &neighbors[1]); // Y direction (periodic)
    MPI_Cart_shift(comm_cart, 1, 1, &neighbors[2], &neighbors[3]); // X direction (Dirichlet)
    MPI_Cart_shift(comm_cart, 2, 1, &neighbors[4], &neighbors[5]); // Z direction (Dirichlet)
    
    VDOUB result_vec;
    double time = 0.0, first_step_inaccuracy = 0.0, last_step_inaccuracy = 0.0, max_inaccuracy = -1.0;
    
    Block block = Block(grid, neighbors, coords, dims[0], dims[1], dims[2], rank);
    // block.print_block_info();
    
    solve_equation(grid, block, dims[0], dims[1], dims[2], comm_cart, time, max_inaccuracy, first_step_inaccuracy, last_step_inaccuracy, result_vec);
    
    if (rank == 0) {
        std::cout << "Result:\n\tTime = " << time << "\n\tMax inaccuracy = " << max_inaccuracy 
                  << "\n\tFirst step inaccuracy = " << first_step_inaccuracy 
                  << "\n\tLast step inaccuracy = " << last_step_inaccuracy << std::endl;
        save_statistics(grid, time, max_inaccuracy, first_step_inaccuracy, last_step_inaccuracy, proc_num);
    }
    
    // Dump results for each process (uncomment if needed)
    // std::string prefix = "results/grid/" + std::to_string(N) + "_" + std::to_string(proc_num) + "_" + grid.L_type + "_";
    // dump_block_to_CSV(grid, block, result_vec, prefix + NUMERICAL + "_" + std::to_string(rank) + ".csv", NUMERICAL);
    // dump_block_to_CSV(grid, block, result_vec, prefix + ANALITICAL + "_" + std::to_string(rank) + ".csv", ANALITICAL);
    // dump_block_to_CSV(grid, block, result_vec, prefix + INACCURACY + "_" + std::to_string(rank) + ".csv", INACCURACY);
    
    MPI_Finalize();
    return 0;
}