#define _USE_MATH_DEFINES
#include "equation.hpp"
#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <cmath>
#include <mpi.h>

double parse_length(const char* arg, std::string& label_part) {
    if (strcmp(arg, "pi") == 0) {
        label_part = "pi";
        return M_PI;
    } else {
        label_part = std::string(arg);
        return atof(arg);
    }
}

int get_optimal_dimensions(int proc_num, int dims[3]) {
    // Подбираем оптимальные размеры сетки процессов
    dims[0] = 1; // y-направление (периодическое)
    dims[1] = 1; // x-направление (условие первого рода)
    dims[2] = 1; // z-направление (условие первого рода)
    
    // Пытаемся распределить процессы поровну по всем направлениям
    for (int i = 1; i <= proc_num; ++i) {
        if (proc_num % i == 0) {
            int rest = proc_num / i;
            for (int j = 1; j <= rest; ++j) {
                if (rest % j == 0) {
                    int k = rest / j;
                    // Предпочитаем большее количество процессов в периодическом направлении (y)
                    if (j > dims[0] || (j == dims[0] && i > dims[1]) || (j == dims[0] && i == dims[1] && k > dims[2])) {
                        dims[0] = j; // y
                        dims[1] = i; // x
                        dims[2] = k; // z
                    }
                }
            }
        }
    }
    
    return 0;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, proc_num;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
    
    if (argc != 4) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " N Lx Ly Lz\n"
                      << "  Lx, Ly, Lz: numbers (e.g. 1.0) or 'pi'\n";
        }
        MPI_Finalize();
        return 1;
    }
    
    int N = atoi(argv[1]);
    if (N <= 0) {
        if (rank == 0)
            std::cerr << "N must be a positive integer." << std::endl;
        MPI_Finalize();
        return 1;
    }
    
    std::string lx_label, ly_label, lz_label;
    double Lx = parse_length(argv[2], lx_label);
    double Ly = parse_length(argv[3], ly_label);
    double Lz = parse_length(argv[4], lz_label);
    
    std::string domain_label = lx_label + "_" + ly_label + "_" + lz_label;
    
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
    
    // Создание декартовой топологии
    int dims[3];
    get_optimal_dimensions(proc_num, dims);
    
    // Периодические условия только по y-направлению (индекс 0)
    int periods[3] = {1, 0, 0};
    
    if (rank == 0) {
        std::cout << "Processors grid: (" << dims[0] << ", " << dims[1] << ", " << dims[2] << ")" << std::endl;
        std::cout << "Periodic directions: y" << std::endl;
    }
    
    MPI_Comm comm_cart;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &comm_cart);
    
    int coords[3];
    MPI_Cart_coords(comm_cart, rank, 3, coords);
    
    // Определение соседей в 6 направлениях
    VINT neighbors(6, MPI_PROC_NULL);
    MPI_Cart_shift(comm_cart, 0, 1, &neighbors[0], &neighbors[1]); // y- и y+
    MPI_Cart_shift(comm_cart, 1, 1, &neighbors[2], &neighbors[3]); // x- и x+
    MPI_Cart_shift(comm_cart, 2, 1, &neighbors[4], &neighbors[5]); // z- и z+
    
    Block block(grid, neighbors, coords, dims[0], dims[1], dims[2], rank);
    
    // Инициализация переменных для результатов
    VDOUB result_vec;
    double time = 0, max_inaccuracy = 0, first_step_inaccuracy = 0, last_step_inaccuracy = 0;
    
    // Решение задачи
    solve_equation(grid, block, dims[0], dims[1], dims[2], comm_cart, time, max_inaccuracy, 
                  first_step_inaccuracy, last_step_inaccuracy, result_vec);
    
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