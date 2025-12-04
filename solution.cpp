#define _USE_MATH_DEFINES
#include "equation.hpp"
#include <algorithm>
#include <iostream>
#include <cmath>
#include <mpi.h>

// Функция для обмена призрачными слоями
void exchange_ghost_layers(Block& b, VDOUB& ui_local, MPI_Comm& comm_cart) {
    MPI_Request reqs[12];
    int req_count = 0;
    
    // 0: left (y-), 1: right (y+), 2: bottom (x-), 3: top (x+), 4: front (z-), 5: back (z+)
    for (int dir = 0; dir < 6; ++dir) {
        int send_neighbor = b.neighbors[dir];
        int recv_neighbor = b.neighbors[(dir % 2 == 0) ? (dir + 1) : (dir - 1)];
        
        if (send_neighbor != MPI_PROC_NULL) {
            int data_size;
            switch(dir) {
                case 0: // left (y-)
                case 1: // right (y+)
                    data_size = b.Nx * b.Nz;
                    break;
                case 2: // bottom (x-)
                case 3: // top (x+)
                    data_size = b.Ny * b.Nz;
                    break;
                case 4: // front (z-)
                case 5: // back (z+)
                    data_size = b.Nx * b.Ny;
                    break;
                default:
                    data_size = 0;
            }
            
            // Отправка данных
            if (dir % 2 == 0) { // Отправка с нижней границы
                switch(dir) {
                    case 0: // y-
                        MPI_Isend(&ui_local[b.index(1, 1, 1)], data_size, MPI_DOUBLE, send_neighbor, 0, comm_cart, &reqs[req_count++]);
                        break;
                    case 2: // x-
                        MPI_Isend(&ui_local[b.index(1, 1, 1)], data_size, MPI_DOUBLE, send_neighbor, 2, comm_cart, &reqs[req_count++]);
                        break;
                    case 4: // z-
                        MPI_Isend(&ui_local[b.index(1, 1, 1)], data_size, MPI_DOUBLE, send_neighbor, 4, comm_cart, &reqs[req_count++]);
                        break;
                }
            } else { // Отправка с верхней границы
                switch(dir) {
                    case 1: // y+
                        MPI_Isend(&ui_local[b.index(1, b.Ny, 1)], data_size, MPI_DOUBLE, send_neighbor, 1, comm_cart, &reqs[req_count++]);
                        break;
                    case 3: // x+
                        MPI_Isend(&ui_local[b.index(b.Nx, 1, 1)], data_size, MPI_DOUBLE, send_neighbor, 3, comm_cart, &reqs[req_count++]);
                        break;
                    case 5: // z+
                        MPI_Isend(&ui_local[b.index(1, 1, b.Nz)], data_size, MPI_DOUBLE, send_neighbor, 5, comm_cart, &reqs[req_count++]);
                        break;
                }
            }
            
            // Прием данных
            if (recv_neighbor != MPI_PROC_NULL) {
                if (dir % 2 == 0) { // Прием на верхний призрачный слой
                    switch(dir) {
                        case 0: // y-
                            MPI_Irecv(&ui_local[b.index(1, b.Ny+1, 1)], data_size, MPI_DOUBLE, recv_neighbor, 1, comm_cart, &reqs[req_count++]);
                            break;
                        case 2: // x-
                            MPI_Irecv(&ui_local[b.index(b.Nx+1, 1, 1)], data_size, MPI_DOUBLE, recv_neighbor, 3, comm_cart, &reqs[req_count++]);
                            break;
                        case 4: // z-
                            MPI_Irecv(&ui_local[b.index(1, 1, b.Nz+1)], data_size, MPI_DOUBLE, recv_neighbor, 5, comm_cart, &reqs[req_count++]);
                            break;
                    }
                } else { // Прием на нижний призрачный слой
                    switch(dir) {
                        case 1: // y+
                            MPI_Irecv(&ui_local[b.index(1, 0, 1)], data_size, MPI_DOUBLE, recv_neighbor, 0, comm_cart, &reqs[req_count++]);
                            break;
                        case 3: // x+
                            MPI_Irecv(&ui_local[b.index(0, 1, 1)], data_size, MPI_DOUBLE, recv_neighbor, 2, comm_cart, &reqs[req_count++]);
                            break;
                        case 5: // z+
                            MPI_Irecv(&ui_local[b.index(1, 1, 0)], data_size, MPI_DOUBLE, recv_neighbor, 4, comm_cart, &reqs[req_count++]);
                            break;
                    }
                }
            }
        }
    }
    
    if (req_count > 0)
        MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
}

// Заполнение буферов для обмена данными
void fill_send_buffers(Block& b, VDOUB& ui_local, int dir) {
    int idx = 0;
    
    switch(dir) {
        case 0: // left (y-)
            for (int i = 1; i <= b.Nx; ++i) {
                for (int k = 1; k <= b.Nz; ++k) {
                    b.left_send[idx++] = ui_local[b.index(i, 1, k)];
                }
            }
            break;
        case 1: // right (y+)
            for (int i = 1; i <= b.Nx; ++i) {
                for (int k = 1; k <= b.Nz; ++k) {
                    b.right_send[idx++] = ui_local[b.index(i, b.Ny, k)];
                }
            }
            break;
        case 2: // bottom (x-)
            for (int j = 1; j <= b.Ny; ++j) {
                for (int k = 1; k <= b.Nz; ++k) {
                    b.bottom_send[idx++] = ui_local[b.index(1, j, k)];
                }
            }
            break;
        case 3: // top (x+)
            for (int j = 1; j <= b.Ny; ++j) {
                for (int k = 1; k <= b.Nz; ++k) {
                    b.top_send[idx++] = ui_local[b.index(b.Nx, j, k)];
                }
            }
            break;
        case 4: // front (z-)
            for (int i = 1; i <= b.Nx; ++i) {
                for (int j = 1; j <= b.Ny; ++j) {
                    b.front_send[idx++] = ui_local[b.index(i, j, 1)];
                }
            }
            break;
        case 5: // back (z+)
            for (int i = 1; i <= b.Nx; ++i) {
                for (int j = 1; j <= b.Ny; ++j) {
                    b.back_send[idx++] = ui_local[b.index(i, j, b.Nz)];
                }
            }
            break;
    }
}

// Заполнение призрачных слоев из полученных данных
void fill_ghost_layers(Block& b, VDOUB& ui_local, int dir) {
    int idx = 0;
    
    switch(dir) {
        case 0: // left (y-)
            for (int i = 1; i <= b.Nx; ++i) {
                for (int k = 1; k <= b.Nz; ++k) {
                    ui_local[b.index(i, 0, k)] = b.left_recieve[idx++];
                }
            }
            break;
        case 1: // right (y+)
            for (int i = 1; i <= b.Nx; ++i) {
                for (int k = 1; k <= b.Nz; ++k) {
                    ui_local[b.index(i, b.Ny+1, k)] = b.right_recieve[idx++];
                }
            }
            break;
        case 2: // bottom (x-)
            for (int j = 1; j <= b.Ny; ++j) {
                for (int k = 1; k <= b.Nz; ++k) {
                    ui_local[b.index(0, j, k)] = b.bottom_recieve[idx++];
                }
            }
            break;
        case 3: // top (x+)
            for (int j = 1; j <= b.Ny; ++j) {
                for (int k = 1; k <= b.Nz; ++k) {
                    ui_local[b.index(b.Nx+1, j, k)] = b.top_recieve[idx++];
                }
            }
            break;
        case 4: // front (z-)
            for (int i = 1; i <= b.Nx; ++i) {
                for (int j = 1; j <= b.Ny; ++j) {
                    ui_local[b.index(i, j, 0)] = b.front_recieve[idx++];
                }
            }
            break;
        case 5: // back (z+)
            for (int i = 1; i <= b.Nx; ++i) {
                for (int j = 1; j <= b.Ny; ++j) {
                    ui_local[b.index(i, j, b.Nz+1)] = b.back_recieve[idx++];
                }
            }
            break;
    }
}

// Корректный оператор Лапласа для локального блока
inline double laplace_operator(const Grid& g, const Block& b, const VDOUB& ui_local, const int& i, const int& j, const int& k) {
    return (ui_local[b.index(i-1, j, k)] - 2 * ui_local[b.index(i, j, k)] + ui_local[b.index(i+1, j, k)]) / (g.h_x * g.h_x) +
           (ui_local[b.index(i, j-1, k)] - 2 * ui_local[b.index(i, j, k)] + ui_local[b.index(i, j+1, k)]) / (g.h_y * g.h_y) +
           (ui_local[b.index(i, j, k-1)] - 2 * ui_local[b.index(i, j, k)] + ui_local[b.index(i, j, k+1)]) / (g.h_z * g.h_z);
}

// Инициализация начальных условий
void init(const Grid& g, Block& b, VVEC& u_local, const int& dim0_n, const int& dim1_n, const int& dim2_n, 
         MPI_Comm& comm_cart, double& max_inaccuracy, double& first_step_inaccuracy) {
    
    // Инициализация всех точек u^0 аналитическим решением
    for (int i = 1; i <= b.Nx; ++i) {
        for (int j = 1; j <= b.Ny; ++j) {
            for (int k = 1; k <= b.Nz; ++k) {
                double x = (b.x_start + i - 1) * g.h_x;
                double y = (b.y_start + j - 1) * g.h_y;
                double z = (b.z_start + k - 1) * g.h_z;
                u_local[0][b.index(i, j, k)] = u_analytical(g, x, y, z, 0.0);
            }
        }
    }
    
    // Граничные условия для u^0
    // Условия первого рода на x=0 и x=Lx
    
    // Установка значений для границ x=0 и x=Lx
    if (b.dim1_rank == 0) {
        for (int j = 1; j <= b.Ny; ++j) {
            for (int k = 1; k <= b.Nz; ++k) {
                u_local[0][b.index(0, j, k)] = 0.0;
            }
        }
    }
    
    if (b.dim1_rank == dim1_n - 1) {
        for (int j = 1; j <= b.Ny; ++j) {
            for (int k = 1; k <= b.Nz; ++k) {
                u_local[0][b.index(b.Nx+1, j, k)] = 0.0;
            }
        }
    }
    
    // Условия первого рода на z=0 и z=Lz
    if (b.dim2_rank == 0) {
        for (int i = 1; i <= b.Nx; ++i) {
            for (int j = 1; j <= b.Ny; ++j) {
                u_local[0][b.index(i, j, 0)] = 0.0;
            }
        }
    }
    
    if (b.dim2_rank == dim2_n - 1) {
        for (int i = 1; i <= b.Nx; ++i) {
            for (int j = 1; j <= b.Ny; ++j) {
                u_local[0][b.index(i, j, b.Nz+1)] = 0.0;
            }
        }
    }
    
    // Периодические условия по y
    if (b.dim0_rank == 0) {
        for (int i = 1; i <= b.Nx; ++i) {
            for (int k = 1; k <= b.Nz; ++k) {
                double x = (b.x_start + i - 1) * g.h_x;
                double z = (b.z_start + k - 1) * g.h_z;
                u_local[0][b.index(i, 0, k)] = u_analytical(g, x, 0.0, z, 0.0);
            }
        }
    }
    
    if (b.dim0_rank == dim0_n - 1) {
        for (int i = 1; i <= b.Nx; ++i) {
            for (int k = 1; k <= b.Nz; ++k) {
                double x = (b.x_start + i - 1) * g.h_x;
                double z = (b.z_start + k - 1) * g.h_z;
                u_local[0][b.index(i, b.Ny+1, k)] = u_analytical(g, x, g.Ly, z, 0.0);
            }
        }
    }
    
    // Обмен призрачными слоями для u^0
    exchange_ghost_layers(b, u_local[0], comm_cart);
    
    // Вычисление u^1 по формуле (12)
    for (int i = 1; i <= b.Nx; ++i) {
        for (int j = 1; j <= b.Ny; ++j) {
            for (int k = 1; k <= b.Nz; ++k) {
                u_local[1][b.index(i, j, k)] = u_local[0][b.index(i, j, k)] + 
                    0.5 * g.a2 * g.tau * g.tau * laplace_operator(g, b, u_local[0], i, j, k);
            }
        }
    }
    
    // Граничные условия для u^1
    if (b.dim1_rank == 0) {
        for (int j = 0; j <= b.Ny+1; ++j) {
            for (int k = 0; k <= b.Nz+1; ++k) {
                u_local[1][b.index(0, j, k)] = 0.0;
            }
        }
    }
    
    if (b.dim1_rank == dim1_n - 1) {
        for (int j = 0; j <= b.Ny+1; ++j) {
            for (int k = 0; k <= b.Nz+1; ++k) {
                u_local[1][b.index(b.Nx+1, j, k)] = 0.0;
            }
        }
    }
    
    if (b.dim2_rank == 0) {
        for (int i = 0; i <= b.Nx+1; ++i) {
            for (int j = 0; j <= b.Ny+1; ++j) {
                u_local[1][b.index(i, j, 0)] = 0.0;
            }
        }
    }
    
    if (b.dim2_rank == dim2_n - 1) {
        for (int i = 0; i <= b.Nx+1; ++i) {
            for (int j = 0; j <= b.Ny+1; ++j) {
                u_local[1][b.index(i, j, b.Nz+1)] = 0.0;
            }
        }
    }
    
    // Периодические условия по y
    if (b.dim0_rank == 0) {
        for (int i = 1; i <= b.Nx; ++i) {
            for (int k = 1; k <= b.Nz; ++k) {
                double x = (b.x_start + i - 1) * g.h_x;
                double z = (b.z_start + k - 1) * g.h_z;
                u_local[1][b.index(i, 0, k)] = u_analytical(g, x, 0.0, z, g.tau);
            }
        }
    }
    
    if (b.dim0_rank == dim0_n - 1) {
        for (int i = 1; i <= b.Nx; ++i) {
            for (int k = 1; k <= b.Nz; ++k) {
                double x = (b.x_start + i - 1) * g.h_x;
                double z = (b.z_start + k - 1) * g.h_z;
                u_local[1][b.index(i, b.Ny+1, k)] = u_analytical(g, x, g.Ly, z, g.tau);
            }
        }
    }
    
    // Обмен призрачными слоями для u^1
    exchange_ghost_layers(b, u_local[1], comm_cart);
    
    // Расчет погрешности на первом шаге
    double local_max_error = 0.0;
    for (int i = 1; i <= b.Nx; ++i) {
        for (int j = 1; j <= b.Ny; ++j) {
            for (int k = 1; k <= b.Nz; ++k) {
                double x = (b.x_start + i - 1) * g.h_x;
                double y = (b.y_start + j - 1) * g.h_y;
                double z = (b.z_start + k - 1) * g.h_z;
                double exact = u_analytical(g, x, y, z, g.tau);
                double err = std::abs(u_local[1][b.index(i, j, k)] - exact);
                if (err > local_max_error) local_max_error = err;
            }
        }
    }
    
    double global_max_error;
    MPI_Reduce(&local_max_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX, 0, comm_cart);
    
    if (b.rank == 0) {
        max_inaccuracy = std::max(max_inaccuracy, global_max_error);
        first_step_inaccuracy = global_max_error;
        std::cout << "Max inaccuracy on step 1: " << global_max_error << std::endl;
    }
}

// Основной алгоритм решения
void run_algo(const Grid& g, Block& b, VVEC& u_local, const int& dim0_n, const int& dim1_n, const int& dim2_n, 
             MPI_Comm& comm_cart, double& max_inaccuracy, double& last_step_inaccuracy) {
    
    for (int s = 2; s < TIME_STEPS; ++s) {
        int prev = (s - 2) % 3;
        int curr = (s - 1) % 3;
        int next = s % 3;
        
        // Обмен призрачными слоями для текущего временного слоя
        exchange_ghost_layers(b, u_local[curr], comm_cart);
        
        // Вычисление значений для внутренних точек
        for (int i = 1; i <= b.Nx; ++i) {
            for (int j = 1; j <= b.Ny; ++j) {
                for (int k = 1; k <= b.Nz; ++k) {
                    u_local[next][b.index(i, j, k)] = 2.0 * u_local[curr][b.index(i, j, k)] 
                        - u_local[prev][b.index(i, j, k)]
                        + g.a2 * g.tau * g.tau * laplace_operator(g, b, u_local[curr], i, j, k);
                }
            }
        }
        
        // Граничные условия для следующего временного слоя
        
        // Условия первого рода на x=0 и x=Lx
        if (b.dim1_rank == 0) {
            for (int j = 0; j <= b.Ny+1; ++j) {
                for (int k = 0; k <= b.Nz+1; ++k) {
                    u_local[next][b.index(0, j, k)] = 0.0;
                }
            }
        }
        
        if (b.dim1_rank == dim1_n - 1) {
            for (int j = 0; j <= b.Ny+1; ++j) {
                for (int k = 0; k <= b.Nz+1; ++k) {
                    u_local[next][b.index(b.Nx+1, j, k)] = 0.0;
                }
            }
        }
        
        // Условия первого рода на z=0 и z=Lz
        if (b.dim2_rank == 0) {
            for (int i = 0; i <= b.Nx+1; ++i) {
                for (int j = 0; j <= b.Ny+1; ++j) {
                    u_local[next][b.index(i, j, 0)] = 0.0;
                }
            }
        }
        
        if (b.dim2_rank == dim2_n - 1) {
            for (int i = 0; i <= b.Nx+1; ++i) {
                for (int j = 0; j <= b.Ny+1; ++j) {
                    u_local[next][b.index(i, j, b.Nz+1)] = 0.0;
                }
            }
        }
        
        // Периодические условия по y
        if (b.dim0_rank == 0) {
            for (int i = 1; i <= b.Nx; ++i) {
                for (int k = 1; k <= b.Nz; ++k) {
                    double x = (b.x_start + i - 1) * g.h_x;
                    double z = (b.z_start + k - 1) * g.h_z;
                    double y = 0.0;
                    u_local[next][b.index(i, 0, k)] = u_analytical(g, x, y, z, s * g.tau);
                }
            }
        }
        
        if (b.dim0_rank == dim0_n - 1) {
            for (int i = 1; i <= b.Nx; ++i) {
                for (int k = 1; k <= b.Nz; ++k) {
                    double x = (b.x_start + i - 1) * g.h_x;
                    double z = (b.z_start + k - 1) * g.h_z;
                    double y = g.Ly;
                    u_local[next][b.index(i, b.Ny+1, k)] = u_analytical(g, x, y, z, s * g.tau);
                }
            }
        }
        
        // Обмен призрачными слоями для следующего шага
        exchange_ghost_layers(b, u_local[next], comm_cart);
        
        // Расчет погрешности
        double local_max_error = 0.0;
        for (int i = 1; i <= b.Nx; ++i) {
            for (int j = 1; j <= b.Ny; ++j) {
                for (int k = 1; k <= b.Nz; ++k) {
                    double x = (b.x_start + i - 1) * g.h_x;
                    double y = (b.y_start + j - 1) * g.h_y;
                    double z = (b.z_start + k - 1) * g.h_z;
                    double exact = u_analytical(g, x, y, z, s * g.tau);
                    double err = std::abs(u_local[next][b.index(i, j, k)] - exact);
                    if (err > local_max_error) local_max_error = err;
                }
            }
        }
        
        double global_max_error;
        MPI_Reduce(&local_max_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX, 0, comm_cart);
        
        if (b.rank == 0) {
            if (global_max_error > max_inaccuracy) 
                max_inaccuracy = global_max_error;
            
            if (s == TIME_STEPS - 1)
                last_step_inaccuracy = global_max_error;
            
            std::cout << "Max inaccuracy on step " << s << ": " << global_max_error << std::endl;
        }
    }
}

// Основная функция решения
void solve_equation(const Grid& grid, Block& block, const int& dim0_n, const int& dim1_n, const int& dim2_n, 
                   MPI_Comm& comm_cart, double& time, double& max_inaccuracy, 
                   double& first_step_inaccuracy, double& last_step_inaccuracy, VDOUB& result) {
    
    // Выделение памяти для трех временных слоев
    VDOUB u0_local(block.N), u1_local(block.N), u2_local(block.N);
    VVEC u_local{u0_local, u1_local, u2_local};
    
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    
    // Инициализация и выполнение алгоритма
    init(grid, block, u_local, dim0_n, dim1_n, dim2_n, comm_cart, max_inaccuracy, first_step_inaccuracy);
    run_algo(grid, block, u_local, dim0_n, dim1_n, dim2_n, comm_cart, max_inaccuracy, last_step_inaccuracy);
    
    double end_time = MPI_Wtime();
    double local_time = end_time - start_time;
    
    // Сбор времени выполнения
    MPI_Reduce(&local_time, &time, 1, MPI_DOUBLE, MPI_MAX, 0, comm_cart);
    
    // Копирование результата последнего временного шага
    result = u_local[(TIME_STEPS - 1) % 3];
}