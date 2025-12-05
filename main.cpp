#define _USE_MATH_DEFINES
#include "equation.hpp"
#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <mpi.h>

// Функция для парсинга длины (аналогично OMP-версии)
double parse_length(const char* arg, std::string& label_part) {
    if (strcmp(arg, "pi") == 0) {
        label_part = "pi";
        return M_PI;
    } else {
        label_part = std::string(arg);
        return atof(arg);
    }
}

int get_0_dim_size(const int& proc_num) {
    if (proc_num > 10 && proc_num % 4 == 0)
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

    // Проверка аргументов командной строки (теперь как у оригинального MPI)
    if (argc != 4 && argc != 6) { // custom требует 6, другие 4
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " N L_type [Lx Ly Lz if L_type=custom]\n"
                      << "  L_type: '1', 'pi', or 'custom'\n"
                      << "  Lx, Ly, Lz: numbers (e.g. 1.0) or 'pi' (only with 'custom')\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int N = atoi(argv[1]);
    std::string l_type = argv[2];

    if (N <= 0) {
        if (rank == 0) {
            std::cerr << "N must be a positive integer.\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    double Lx, Ly, Lz;
    std::string lx_label, ly_label, lz_label;

    if (l_type == "pi") {
        Lx = Ly = Lz = M_PI;
        lx_label = ly_label = lz_label = "pi";
    } else if (l_type == "1") {
        Lx = Ly = Lz = 1.0;
        lx_label = ly_label = lz_label = "1";
    } else if (l_type == "custom") {
        if (argc != 6) {
             if (rank == 0) {
                std::cerr << "Usage: " << argv[0] << " N L_type [Lx Ly Lz if L_type=custom]\n"
                          << "  L_type: 'custom' requires Lx, Ly, Lz.\n";
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        Lx = parse_length(argv[3], lx_label);
        Ly = parse_length(argv[4], ly_label);
        Lz = parse_length(argv[5], lz_label);
    } else {
        if (rank == 0) {
            std::cerr << "Invalid L_type: must be '1', 'pi', or 'custom'.\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::string domain_label = l_type; // или можно использовать комбинацию Lx_Ly_Lz

    Grid grid(N, Lx, Ly, Lz, domain_label);

    if (rank == 0) {
        std::cout << "Input values:\n"
                  << "\tN = " << grid.N << "\n"
                  << "\tProcesses = " << proc_num << "\n" // proc_num определено через MPI
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
    // NOTE: Периодические границы (1, 1, 1) соответствуют варианту 8.
    // Для варианта 3 должны быть 0, 1, 0.
    // Это будет изменено позже при адаптации под вариант 3.
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