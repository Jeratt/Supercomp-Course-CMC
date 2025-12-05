#define _USE_MATH_DEFINES
#include "equation.hpp" // Убедитесь, что подключаете вашу версию
#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <mpi.h>

double parse_length(const char* arg) {
    if (strcmp(arg, "pi") == 0)
        return M_PI;
    else
        return atof(arg);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, proc_num;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_num);

    if (argc != 5) {
        if (rank == 0)
            std::cerr << "Usage: " << argv[0] << " N Lx Ly Lz\n"
                      << "  Lx, Ly, Lz: numbers (e.g. 1.0) or 'pi'\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int N = atoi(argv[1]);
    if (N <= 0) {
        if (rank == 0)
            std::cerr << "Invalid N: must be > 0" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    double Lx = parse_length(argv[2]);
    double Ly = parse_length(argv[3]);
    double Lz = parse_length(argv[4]);
    
    // Создаем метки для L_type (аналогично вашей OMP версии)
    std::string lx_label, ly_label, lz_label;
    const char* arg_lx = argv[2];
    const char* arg_ly = argv[3];
    const char* arg_lz = argv[4];
    
    if (strcmp(arg_lx, "pi") == 0) lx_label = "pi"; else lx_label = std::string(arg_lx);
    if (strcmp(arg_ly, "pi") == 0) ly_label = "pi"; else ly_label = std::string(arg_ly);
    if (strcmp(arg_lz, "pi") == 0) lz_label = "pi"; else lz_label = std::string(arg_lz);
    
    // Grid grid = Grid(N, const_cast<char*>(L_type), Lx, Ly, Lz); // Старый вызов
    Grid grid = Grid(N, Lx, Ly, Lz, lx_label, ly_label, lz_label); // Новый вызов

    if (rank == 0)
        std::cout << "Input values:\n\tN = " << grid.N << "\n\tProcesses = " << proc_num 
                  //<< "\n\tL_type = " << grid.L_type << "\n\tLx = " << grid.Lx // Старый вывод
                  << "\n\tDomain label = " << grid.domain_label << "\n\tLx = " << grid.Lx // Новый вывод
                  << "\n\tLy = " << grid.Ly << "\n\tLz = " << grid.Lz << std::endl;

    // Создание топологии
    int dims[3] = {0, 0, 0};
    MPI_Dims_create(proc_num, 3, dims);
    
    if (rank == 0)
        std::cout << "Dims topology: (" << dims[0] << ", " << dims[1] << ", " << dims[2] << ")" << std::endl;

    // Periodicity: variant 3 = 1P П 1P (Dirichlet on x,z, Periodic on y)
    int periods[3] = {0, 1, 0};  // Only Y direction is periodic
    
    MPI_Comm comm_cart;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &comm_cart);

    int coords[3];
    MPI_Cart_coords(comm_cart, rank, 3, coords);

    // Find neighbors in 6 directions
    VINT neighbors(6);
    int dummy;
    
    // X direction (Dirichlet - no periodic wrap)
    MPI_Cart_shift(comm_cart, 0, -1, &neighbors[0], &dummy);  // left (x-)
    MPI_Cart_shift(comm_cart, 0, +1, &dummy, &neighbors[1]);  // right (x+)
    
    // Y direction (Periodic)
    MPI_Cart_shift(comm_cart, 1, -1, &neighbors[2], &dummy);  // bottom (y-)
    MPI_Cart_shift(comm_cart, 1, +1, &dummy, &neighbors[3]);  // top (y+)
    
    // Z direction (Dirichlet - no periodic wrap)
    MPI_Cart_shift(comm_cart, 2, -1, &neighbors[4], &dummy);  // front (z-)
    MPI_Cart_shift(comm_cart, 2, +1, &dummy, &neighbors[5]);  // back (z+)

    VDOUB result_vec;
    double time, first_step_inaccuracy, last_step_inaccuracy, max_inaccuracy = -1;
    
    Block block = Block(grid, neighbors, coords, dims[0], dims[1], dims[2], rank);
    
    // Uncomment for debugging:
    // block.print_block_info();
    
    solve_equation(grid, block, dims[0], dims[1], dims[2], comm_cart, 
                   time, max_inaccuracy, first_step_inaccuracy, last_step_inaccuracy, result_vec);

    if (rank == 0) {
        std::cout << "Result:\n\tTime = " << time << "\n\tMax inaccuracy = " << max_inaccuracy 
                  << "\n\tFirst step inaccuracy = " << first_step_inaccuracy 
                  << "\n\tLast step inaccuracy = " << last_step_inaccuracy << std::endl;
    }

    MPI_Finalize();
    return 0;
}