#define _USE_MATH_DEFINES
#include "equation.hpp"
#include <algorithm>
#include <iostream>
#include <cmath>
#include <mpi.h>

// Обмен призрачными слоями между соседними процессами
void exchange_ghost_layers(Block& b, VDOUB& ui_local, MPI_Comm& comm_cart) {
    MPI_Request reqs[12];
    int req_count = 0;
    
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
            
            if (dir % 2 == 0) { // send from lower boundary
                MPI_Isend(&ui_local[b.index(1, 1, 1)], data_size, MPI_DOUBLE, send_neighbor, b.rank * 6 + dir, comm_cart, &reqs[req_count++]);
            } else { // send from upper boundary
                MPI_Isend(&ui_local[b.index(b.padded_Nx-2, 1, 1)], data_size, MPI_DOUBLE, send_neighbor, b.rank * 6 + dir, comm_cart, &reqs[req_count++]);
            }
            
            if (recv_neighbor != MPI_PROC_NULL) {
                if (dir % 2 == 0) { // receive to upper ghost layer
                    MPI_Irecv(&ui_local[b.index(b.padded_Nx-1, 1, 1)], data_size, MPI_DOUBLE, recv_neighbor, recv_neighbor * 6 + (dir+1), comm_cart, &reqs[req_count++]);
                } else { // receive to lower ghost layer
                    MPI_Irecv(&ui_local[b.index(0, 1, 1)], data_size, MPI_DOUBLE, recv_neighbor, recv_neighbor * 6 + (dir-1), comm_cart, &reqs[req_count++]);
                }
            }
        }
    }
    
    if (req_count > 0)
        MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
}

// Заполнение буферов для обмена данными
void fill_send_buffers(Block& b, VDOUB& ui_local, VDOUB& send_buf, int dir) {
    int idx = 0;
    
    switch(dir) {
        case 0: // left (y-)
            for (int i = 1; i <= b.Nx; ++i) {
                for (int k = 1; k <= b.Nz; ++k) {
                    send_buf[idx++] = ui_local[b.index(i, 1, k)];
                }
            }
            break;
        case 1: // right (y+)
            for (int i = 1; i <= b.Nx; ++i) {
                for (int k = 1; k <= b.Nz; ++k) {
                    send_buf[idx++] = ui_local[b.index(i, b.Ny, k)];
                }
            }
            break;
        case 2: // bottom (x-)
            for (int j = 1; j <= b.Ny; ++j) {
                for (int k = 1; k <= b.Nz; ++k) {
                    send_buf[idx++] = ui_local[b.index(1, j, k)];
                }
            }
            break;
        case 3: // top (x+)
            for (int j = 1; j <= b.Ny; ++j) {
                for (int k = 1; k <= b.Nz; ++k) {
                    send_buf[idx++] = ui_local[b.index(b.Nx, j, k)];
                }
            }
            break;
        case 4: // front (z-)
            for (int i = 1; i <= b.Nx; ++i) {
                for (int j = 1; j <= b.Ny; ++j) {
                    send_buf[idx++] = ui_local[b.index(i, j, 1)];
                }
            }
            break;
        case 5: // back (z+)
            for (int i = 1; i <= b.Nx; ++i) {
                for (int j = 1; j <= b.Ny; ++j) {
                    send_buf[idx++] = ui_local[b.index(i, j, b.Nz)];
                }
            }
            break;
    }
}

// Заполнение призрачных слоев из полученных данных
void fill_ghost_layers(Block& b, VDOUB& ui_local, VDOUB& recv_buf, int dir) {
    int idx = 0;
    
    switch(dir) {
        case 0: // left (y-)
            for (int i = 1; i <= b.Nx; ++i) {
                for (int k = 1; k <= b.Nz; ++k) {
                    ui_local[b.index(i, 0, k)] = recv_buf[idx++];
                }
            }
            break;
        case 1: // right (y+)
            for (int i = 1; i <= b.Nx; ++i) {
                for (int k = 1; k <= b.Nz; ++k) {
                    ui_local[b.index(i, b.Ny+1, k)] = recv_buf[idx++];
                }
            }
            break;
        case 2: // bottom (x-)
            for (int j = 1; j <= b.Ny; ++j) {
                for (int k = 1; k <= b.Nz; ++k) {
                    ui_local[b.index(0, j, k)] = recv_buf[idx++];
                }
            }
            break;
        case 3: // top (x+)
            for (int j = 1; j <= b.Ny; ++j) {
                for (int k = 1; k <= b.Nz; ++k) {
                    ui_local[b.index(b.Nx+1, j, k)] = recv_buf[idx++];
                }
            }
            break;
        case 4: // front (z-)
            for (int i = 1; i <= b.Nx; ++i) {
                for (int j = 1; j <= b.Ny; ++j) {
                    ui_local[b.index(i, j, 0)] = recv_buf[idx++];
                }
            }
            break;
        case 5: // back (z+)
            for (int i = 1; i <= b.Nx; ++i) {
                for (int j = 1; j <= b.Ny; ++j) {
                    ui_local[b.index(i, j, b.Nz+1)] = recv_buf[idx++];
                }
            }
            break;
    }
}

// Адаптированный оператор Лапласа для локального блока
inline double laplace_operator(const Grid& g, const Block& b, const VDOUB& ui_local, const int& i, const int& j, const int& k) {
    return (ui_local[b.index(i-1, j, k)] - 2 * ui_local[b.index(i, j, k)] + ui_local[b.index(i+1, j, k)]) / (g.h_x * g.h_x) +
           (ui_local[b.index(i, j-1, k)] - 2 * ui_local[b.index(i, j, k)] + ui_local[b.index(i, j+1, k)]) / (g.h_y * g.h_y) +
           (ui_local[b.index(i, j, k-1)] - 2 * ui_local[b.index(i, j, k)] + ui_local[b.index(i, j, k+1)]) / (g.h_z * g.h_z);
}

// Инициализация начальных условий
void init(const Grid& g, Block& b, VVEC& u_local, const int& dim0_n, const int& dim1_n, const int& dim2_n, 
         MPI_Comm& comm_cart, double& max_inaccuracy, double& first_step_inaccuracy) {
    
    // Установка значений в призрачных слоях для периодических условий по y
    if (b.dim0_rank == 0) {
        // Нижняя граница по y (y=0)
        for (int i = 1; i <= b.Nx; ++i) {
            for (int k = 1; k <= b.Nz; ++k) {
                double x = (b.x_start + i - 1) * g.h_x;
                double z = (b.z_start + k - 1) * g.h_z;
                u_local[0][b.index(i, 0, k)] = u_analytical(g, x, 0.0, z, 0.0);
                u_local[1][b.index(i, 0, k)] = u_analytical(g, x, 0.0, z, g.tau);
            }
        }
    }
    
    if (b.dim0_rank == dim0_n - 1) {
        // Верхняя граница по y (y=Ly)
        for (int i = 1; i <= b.Nx; ++i) {
            for (int k = 1; k <= b.Nz; ++k) {
                double x = (b.x_start + i - 1) * g.h_x;
                double z = (b.z_start + k - 1) * g.h_z;
                u_local[0][b.index(i, b.Ny+1, k)] = u_analytical(g, x, g.Ly, z, 0.0);
                u_local[1][b.index(i, b.Ny+1, k)] = u_analytical(g, x, g.Ly, z, g.tau);
            }
        }
    }
    
    // Условия первого рода для x=0 и x=Lx
    if (b.dim1_rank == 0) {
        // x=0 граница
        for (int j = 0; j <= b.Ny+1; ++j) {
            for (int k = 0; k <= b.Nz+1; ++k) {
                u_local[0][b.index(0, j, k)] = 0.0;
                u_local[1][b.index(0, j, k)] = 0.0;
            }
        }
    }
    
    if (b.dim1_rank == dim1_n - 1) {
        // x=Lx граница
        for (int j = 0; j <= b.Ny+1; ++j) {
            for (int k = 0; k <= b.Nz+1; ++k) {
                u_local[0][b.index(b.Nx+1, j, k)] = 0.0;
                u_local[1][b.index(b.Nx+1, j, k)] = 0.0;
            }
        }
    }
    
    // Условия первого рода для z=0 и z=Lz
    if (b.dim2_rank == 0) {
        // z=0 граница
        for (int i = 0; i <= b.Nx+1; ++i) {
            for (int j = 0; j <= b.Ny+1; ++j) {
                u_local[0][b.index(i, j, 0)] = 0.0;
                u_local[1][b.index(i, j, 0)] = 0.0;
            }
        }
    }
    
    if (b.dim2_rank == dim2_n - 1) {
        // z=Lz граница
        for (int i = 0; i <= b.Nx+1; ++i) {
            for (int j = 0; j <= b.Ny+1; ++j) {
                u_local[0][b.index(i, j, b.Nz+1)] = 0.0;
                u_local[1][b.index(i, j, b.Nz+1)] = 0.0;
            }
        }
    }
    
    // Внутренние точки для u^0 (начальное условие)
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
    
    // Граничные точки для u^1
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
        
        // Обработка внутренних точек
        for (int i = 1; i <= b.Nx; ++i) {
            for (int j = 1; j <= b.Ny; ++j) {
                for (int k = 1; k <= b.Nz; ++k) {
                    u_local[next][b.index(i, j, k)] = 2.0 * u_local[curr][b.index(i, j, k)] 
                        - u_local[prev][b.index(i, j, k)]
                        + g.a2 * g.tau * g.tau * laplace_operator(g, b, u_local[curr], i, j, k);
                }
            }
        }
        
        // Граничные условия
        
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
        
        // Условия первого рода для x
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
        
        // Условия первого рода для z
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
        
        // Расчет погрешности на текущем шаге
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