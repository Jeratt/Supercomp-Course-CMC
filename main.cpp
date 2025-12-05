#define _USE_MATH_DEFINES
#include "equation.hpp"
#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <mpi.h>

int get_0_dim_size(const int& proc_num) {
    if (proc_num > 10 and proc_num % 4 == 0)
        return 4;
    else if (proc_num % 2 == 0)
        return 2;
    else
        return 1;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, proc_num;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_num);

    // Проверка аргументов командной строки
    if (argc != 6) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " N PROCESSES Lx Ly Lz\n"
                      << "  Lx, Ly, Lz: numbers (e.g. 1.0) or 'pi'\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int N = atoi(argv[1]);
    int procs_num = atoi(argv[2]); // Не используем, размер определяется из MPI_Comm_size
    if (N <= 0) {
        if (rank == 0) {
            std::cerr << "N must be a positive integer.\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Парсим длины сторон
    auto parse_length = [](const char* arg) -> double {
        if (strcmp(arg, "pi") == 0) {
            return M_PI;
        } else {
            return atof(arg);
        }
    };

    double Lx = parse_length(argv[3]);
    double Ly = parse_length(argv[4]);
    double Lz = parse_length(argv[5]);

    std::string domain_label = "mpi_custom"; // или можно сформировать из argv[3-5]

    Grid grid(N, Lx, Ly, Lz, domain_label);

    if (rank == 0) {
        std::cout << "Input values:\n"
                  << "\tN = " << grid.N << "\n"
                  << "\tProcesses = " << proc_num << "\n"
                  << "\tLx = " << grid.Lx << "\n"
                  << "\tLy = " << grid.Ly << "\n"
                  << "\tLz = " << grid.Lz << "\n"
                  << "\tDomain label = " << grid.domain_label << std::endl;
    }

    // create cartesian topology
    int dims[3] = {get_0_dim_size(proc_num), 0, 0};
    MPI_Dims_create(proc_num, 3, dims);
    if (rank == 0)
        std::cout << "Dims topology: (" << dims[0] << ", " << dims[1] << ", " << dims[2] << ")" << std::endl;
    int periods[3] = {1, 1, 1}; // periodic boundaries - NOTE: This is specific to variant 8
    MPI_Comm comm_cart;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &comm_cart);
    int coords[3];
    MPI_Cart_coords(comm_cart, rank, 3, coords);

    // neighbors in 6 directions
    VINT neighbors(6);
    MPI_Cart_shift(comm_cart, 0, 1, &neighbors[0], &neighbors[1]); // (left and right)
    MPI_Cart_shift(comm_cart, 1, 1, &neighbors[2], &neighbors[3]); // (bottom and top)
    MPI_Cart_shift(comm_cart, 2, 1, &neighbors[4], &neighbors[5]); // (front and back)

    VDOUB result_vec;
    double time, first_step_inaccuracy, last_step_inaccuracy, max_inaccuracy = -1;
    Block block = Block(grid, neighbors, coords, dims[0], dims[1], dims[2], rank);
//    block.print_block_info();
    solve_equation(grid, block, dims[0], dims[1], dims[2], comm_cart, time, max_inaccuracy, first_step_inaccuracy, last_step_inaccuracy, result_vec);

    if (rank == 0) {
        std::cout << "Result:\n"
                  << "\tTime = " << time << "\n"
                  << "\tMax inaccuracy = " << max_inaccuracy << "\n"
                  << "\tFirst step inaccuracy = " << first_step_inaccuracy << "\n"
                  << "\tLast step inaccuracy = " << last_step_inaccuracy << std::endl;
    }

    MPI_Finalize();
    return 0;
}