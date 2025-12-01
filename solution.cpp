#define _USE_MATH_DEFINES
#include "equation.hpp"
#include <mpi.h>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;

inline double laplace_operator(const Grid& g, Block& b, const VDOUB& u, int i, int j, int k) {
    double d2x = (u[b.local_index(i - 1, j, k)] - 2.0 * u[b.local_index(i, j, k)] + u[b.local_index(i + 1, j, k)]) 
                / (g.h_x * g.h_x);
    
    // Для периодических условий по y уже учтены корректные значения в гало-ячейках
    double d2y = (u[b.local_index(i, j - 1, k)] - 2.0 * u[b.local_index(i, j, k)] + u[b.local_index(i, j + 1, k)]) 
                / (g.h_y * g.h_y);
    
    double d2z = (u[b.local_index(i, j, k - 1)] - 2.0 * u[b.local_index(i, j, k)] + u[b.local_index(i, j, k + 1)]) 
                / (g.h_z * g.h_z);
    
    return d2x + d2y + d2z;
}

void exchange_halos(Block& b, VDOUB& u) {
    const int tag_left   = 1, tag_right  = 2,
              tag_bottom = 3, tag_top    = 4,
              tag_front  = 5, tag_back   = 6;
              
    MPI_Request req[12];
    int nreq = 0;
    
    // X axis -> first order (Dirichlet)
    if (b.neighbors[0] != -1) { // left neighbor exists (x-)
        for (int idx = 0, j = 1; j <= b.Ny; ++j)
            for (int k = 1; k <= b.Nz; ++k, ++idx)
                b.left_send[idx] = u[b.local_index(1, j, k)];
                
        MPI_Irecv(b.left_recv.data(), b.Ny * b.Nz, MPI_DOUBLE, b.neighbors[0], tag_right,  MPI_COMM_WORLD, &req[nreq++]);
        MPI_Isend(b.left_send.data(), b.Ny * b.Nz, MPI_DOUBLE, b.neighbors[0], tag_left,   MPI_COMM_WORLD, &req[nreq++]);
    }
    
    if (b.neighbors[1] != -1) { // right neighbor exists (x+)
        for (int idx = 0, j = 1; j <= b.Ny; ++j)
            for (int k = 1; k <= b.Nz; ++k, ++idx)
                b.right_send[idx] = u[b.local_index(b.Nx, j, k)];
                
        MPI_Irecv(b.right_recv.data(), b.Ny * b.Nz, MPI_DOUBLE, b.neighbors[1], tag_left,   MPI_COMM_WORLD, &req[nreq++]);
        MPI_Isend(b.right_send.data(), b.Ny * b.Nz, MPI_DOUBLE, b.neighbors[1], tag_right,  MPI_COMM_WORLD, &req[nreq++]);
    }
    
    // Y axis -> periodic
    if (b.neighbors[2] != -1) { // bottom neighbor (y-)
        for (int idx = 0, i = 1; i <= b.Nx; ++i)
            for (int k = 1; k <= b.Nz; ++k, ++idx)
                b.bottom_send[idx] = u[b.local_index(i, 1, k)];
                
        MPI_Irecv(b.bottom_recv.data(), b.Nx * b.Nz, MPI_DOUBLE, b.neighbors[2], tag_top,    MPI_COMM_WORLD, &req[nreq++]);
        MPI_Isend(b.bottom_send.data(), b.Nx * b.Nz, MPI_DOUBLE, b.neighbors[2], tag_bottom, MPI_COMM_WORLD, &req[nreq++]);
    }
    
    if (b.neighbors[3] != -1) { // top neighbor (y+)
        for (int idx = 0, i = 1; i <= b.Nx; ++i)
            for (int k = 1; k <= b.Nz; ++k, ++idx)
                b.top_send[idx] = u[b.local_index(i, b.Ny, k)];
                
        MPI_Irecv(b.top_recv.data(), b.Nx * b.Nz, MPI_DOUBLE, b.neighbors[3], tag_bottom, MPI_COMM_WORLD, &req[nreq++]);
        MPI_Isend(b.top_send.data(), b.Nx * b.Nz, MPI_DOUBLE, b.neighbors[3], tag_top,    MPI_COMM_WORLD, &req[nreq++]);
    }
    
    // Z axis -> first order (Dirichlet)
    if (b.neighbors[4] != -1) { // front neighbor (z-)
        for (int idx = 0, i = 1; i <= b.Nx; ++i)
            for (int j = 1; j <= b.Ny; ++j, ++idx)
                b.front_send[idx] = u[b.local_index(i, j, 1)];
                
        MPI_Irecv(b.front_recv.data(), b.Nx * b.Ny, MPI_DOUBLE, b.neighbors[4], tag_back,  MPI_COMM_WORLD, &req[nreq++]);
        MPI_Isend(b.front_send.data(), b.Nx * b.Ny, MPI_DOUBLE, b.neighbors[4], tag_front, MPI_COMM_WORLD, &req[nreq++]);
    }
    
    if (b.neighbors[5] != -1) { // back neighbor (z+)
        for (int idx = 0, i = 1; i <= b.Nx; ++i)
            for (int j = 1; j <= b.Ny; ++j, ++idx)
                b.back_send[idx] = u[b.local_index(i, j, b.Nz)];
                
        MPI_Irecv(b.back_recv.data(), b.Nx * b.Ny, MPI_DOUBLE, b.neighbors[5], tag_front, MPI_COMM_WORLD, &req[nreq++]);
        MPI_Isend(b.back_send.data(), b.Nx * b.Ny, MPI_DOUBLE, b.neighbors[5], tag_back,  MPI_COMM_WORLD, &req[nreq++]);
    }
    
    MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);
    
    // Обновление гало-ячейки x
    if (b.neighbors[0] != -1) {
        for (int idx = 0, j = 1; j <= b.Ny; ++j)
            for (int k = 1; k <= b.Nz; ++k, ++idx)
                u[b.local_index(0, j, k)] = b.left_recv[idx];
    }
    
    if (b.neighbors[1] != -1) {
        for (int idx = 0, j = 1; j <= b.Ny; ++j)
            for (int k = 1; k <= b.Nz; ++k, ++idx)
                u[b.local_index(b.Nx + 1, j, k)] = b.right_recv[idx];
    }
    
    // Обновление гало-ячейки y
    if (b.neighbors[2] != -1) {
        for (int idx = 0, i = 1; i <= b.Nx; ++i)
            for (int k = 1; k <= b.Nz; ++k, ++idx)
                u[b.local_index(i, 0, k)] = b.bottom_recv[idx];
    }
    
    if (b.neighbors[3] != -1) {
        for (int idx = 0, i = 1; i <= b.Nx; ++i)
            for (int k = 1; k <= b.Nz; ++k, ++idx)
                u[b.local_index(i, b.Ny + 1, k)] = b.top_recv[idx];
    }
    
    // Обновление гало-ячейки z
    if (b.neighbors[4] != -1) {
        for (int idx = 0, i = 1; i <= b.Nx; ++i)
            for (int j = 1; j <= b.Ny; ++j, ++idx)
                u[b.local_index(i, j, 0)] = b.front_recv[idx];
    }
    
    if (b.neighbors[5] != -1) {
        for (int idx = 0, i = 1; i <= b.Nx; ++i)
            for (int j = 1; j <= b.Ny; ++j, ++idx)
                u[b.local_index(i, j, b.Nz + 1)] = b.back_recv[idx];
    }
}

void apply_boundary_conditions(const Grid& g, Block& b, VDOUB& u) {
    // X axis - Dirichlet conditions (u=0)
    if (b.x_start == 0) { // left boundary (x=0)
        for (int j = 0; j <= b.padded_Ny - 1; ++j) // включаем гало-ячейки
            for (int k = 0; k <= b.padded_Nz - 1; ++k)
                u[b.local_index(0, j, k)] = 0.0;
    }
    
    if (b.x_end == g.N) { // right boundary (x=Lx)
        for (int j = 0; j <= b.padded_Ny - 1; ++j)
            for (int k = 0; k <= b.padded_Nz - 1; ++k)
                u[b.local_index(b.Nx + 1, j, k)] = 0.0;
    }
    
    // Z axis - Dirichlet conditions (u=0)
    if (b.z_start == 0) { // front boundary (z=0)
        for (int i = 0; i <= b.padded_Nx - 1; ++i)
            for (int j = 0; j <= b.padded_Ny - 1; ++j)
                u[b.local_index(i, j, 0)] = 0.0;
    }
    
    if (b.z_end == g.N) { // back boundary (z=Lz)
        for (int i = 0; i <= b.padded_Nx - 1; ++i)
            for (int j = 0; j <= b.padded_Ny - 1; ++j)
                u[b.local_index(i, j, b.Nz + 1)] = 0.0;
    }
    
    // Y axis - Periodic conditions are handled via halo exchange, 
    // no additional action needed here for interior points
}

void enforce_periodicity_y(const Grid& /*g*/, Block& b, VDOUB& u, MPI_Comm comm_cart) {
    // Периодические условия по y: u(x,0,z) = u(x,Ly,z)
    // Обработка специальных случаев для процессов на границах y=0 и y=Ly
    
    MPI_Status status;
    const int TAG_PERIODIC = 999;
    
    // Случай 1: один процесс содержит обе границы (редко при большом числе процессов)
    if (b.dimy == 1) {
        // Если один процесс отвечает за всю ось y, просто копируем значения
        for (int i = 0; i <= b.Nx + 1; ++i) {
            for (int k = 0; k <= b.Nz + 1; ++k) {
                u[b.local_index(i, b.Ny + 1, k)] = u[b.local_index(i, 1, k)];
                u[b.local_index(i, 0, k)] = u[b.local_index(i, b.Ny, k)];
            }
        }
        return;
    }
    
    // Случай 2: разные процессы отвечают за границы y=0 и y=Ly
    bool is_bottom_proc = (b.coord_y == 0);       // y = 0
    bool is_top_proc = (b.coord_y == b.dimy - 1);  // y = Ly
    
    if (is_bottom_proc && is_top_proc) {
        // Этот процесс содержит обе границы
        for (int i = 1; i <= b.Nx; ++i) {
            for (int k = 1; k <= b.Nz; ++k) {
                // y = Ly (верхняя граница блока) = y = 0 (нижняя граница блока)
                u[b.local_index(i, b.Ny + 1, k)] = u[b.local_index(i, 1, k)];
                // y = 0 (нижняя граница блока) = y = Ly (верхняя граница блока)
                u[b.local_index(i, 0, k)] = u[b.local_index(i, b.Ny, k)];
            }
        }
    } 
    else if (is_bottom_proc) {
        // Этот процесс содержит границу y = 0
        // 1. Отправляем данные с y = 1 (первая внутренняя строка) процессу с y = Ly
        std::vector<double> send_buf(b.Nx * b.Nz);
        int idx = 0;
        for (int i = 1; i <= b.Nx; ++i)
            for (int k = 1; k <= b.Nz; ++k)
                send_buf[idx++] = u[b.local_index(i, 1, k)];
        
        MPI_Send(send_buf.data(), b.Nx * b.Nz, MPI_DOUBLE, b.neighbors[3], TAG_PERIODIC, comm_cart);
        
        // 2. Принимаем данные для y = 0 (гало-ячейка) от процесса с y = Ly
        std::vector<double> recv_buf(b.Nx * b.Nz);
        MPI_Recv(recv_buf.data(), b.Nx * b.Nz, MPI_DOUBLE, b.neighbors[3], TAG_PERIODIC, comm_cart, &status);
        
        // 3. Заполняем гало-ячейку y = 0
        idx = 0;
        for (int i = 1; i <= b.Nx; ++i)
            for (int k = 1; k <= b.Nz; ++k)
                u[b.local_index(i, 0, k)] = recv_buf[idx++];
    }
    else if (is_top_proc) {
        // Этот процесс содержит границу y = Ly
        // 1. Принимаем данные для y = Ly+1 (гало-ячейка) от процесса с y = 0
        std::vector<double> recv_buf(b.Nx * b.Nz);
        MPI_Recv(recv_buf.data(), b.Nx * b.Nz, MPI_DOUBLE, b.neighbors[2], TAG_PERIODIC, comm_cart, &status);
        
        // 2. Заполняем гало-ячейку y = Ly+1
        int idx = 0;
        for (int i = 1; i <= b.Nx; ++i)
            for (int k = 1; k <= b.Nz; ++k)
                u[b.local_index(i, b.Ny + 1, k)] = recv_buf[idx++];
        
        // 3. Отправляем данные с y = Ny (последняя внутренняя строка) процессу с y = 0
        std::vector<double> send_buf(b.Nx * b.Nz);
        idx = 0;
        for (int i = 1; i <= b.Nx; ++i)
            for (int k = 1; k <= b.Nz; ++k)
                send_buf[idx++] = u[b.local_index(i, b.Ny, k)];
        
        MPI_Send(send_buf.data(), b.Nx * b.Nz, MPI_DOUBLE, b.neighbors[2], TAG_PERIODIC, comm_cart);
    }
}

void init(const Grid& g, Block& b, VVEC& u, MPI_Comm comm_cart, 
          double& max_inaccuracy, double& first_step_inaccuracy) {
    // Шаг 1: Инициализация u^0 аналитическим решением для всех точек (включая границы)
    for (int i = 0; i <= b.Nx + 1; ++i) {
        int i_global = b.x_start + i - 1;
        if (i_global < 0 || i_global > g.N) continue;
        double x = g.x_coord(i_global);
        
        for (int j = 0; j <= b.Ny + 1; ++j) {
            int j_global = b.y_start_global + j - 1;
            if (j_global < 0 || j_global > g.N) continue;
            double y = g.y_coord(j_global);
            
            for (int k = 0; k <= b.Nz + 1; ++k) {
                int k_global = b.z_start + k - 1;
                if (k_global < 0 || k_global > g.N) continue;
                double z = g.z_coord(k_global);
                
                u[0][b.local_index(i, j, k)] = u_analytical(g, x, y, z, 0.0);
            }
        }
    }
    
    // Шаг 2: Обмен гало-ячейками для u^0
    exchange_halos(b, u[0]);
    
    // Шаг 3: Применение граничных условий для u^0
    apply_boundary_conditions(g, b, u[0]);
    
    // Шаг 4: Обеспечение периодичности по y для u^0
    enforce_periodicity_y(g, b, u[0], comm_cart);
    
    // Шаг 5: Вычисление u^1 для внутренних точек по формуле второго порядка
    for (int i = 1; i <= b.Nx; ++i) {
        int i_global = b.x_start + i - 1;
        double x = g.x_coord(i_global);
        
        for (int j = 1; j <= b.Ny; ++j) {
            int j_global = b.y_start_global + j - 1;
            double y = g.y_coord(j_global);
            
            for (int k = 1; k <= b.Nz; ++k) {
                int k_global = b.z_start + k - 1;
                double z = g.z_coord(k_global);
                
                // Внутренние точки вычисляем по формуле
                if (i_global > 0 && i_global < g.N && 
                    j_global > 0 && j_global < g.N && 
                    k_global > 0 && k_global < g.N) {
                    u[1][b.local_index(i, j, k)] = u[0][b.local_index(i, j, k)]
                        + 0.5 * g.a2 * g.tau * g.tau * laplace_operator(g, b, u[0], i, j, k);
                } 
                // Граничные точки берем из аналитического решения
                else {
                    u[1][b.local_index(i, j, k)] = u_analytical(g, x, y, z, g.tau);
                }
            }
        }
    }
    
    // Шаг 6: Обмен гало-ячейками для u^1
    exchange_halos(b, u[1]);
    
    // Шаг 7: Применение граничных условий для u^1
    apply_boundary_conditions(g, b, u[1]);
    
    // Шаг 8: Обеспечение периодичности по y для u^1
    enforce_periodicity_y(g, b, u[1], comm_cart);
    
    // Шаг 9: Вычисление погрешности на первом шаге
    double local_max_err = 0.0;
    double t = g.tau;
    
    for (int i = 1; i <= b.Nx; ++i) {
        int i_global = b.x_start + i - 1;
        double x = g.x_coord(i_global);
        
        for (int j = 1; j <= b.Ny; ++j) {
            int j_global = b.y_start_global + j - 1;
            double y = g.y_coord(j_global);
            
            for (int k = 1; k <= b.Nz; ++k) {
                int k_global = b.z_start + k - 1;
                double z = g.z_coord(k_global);
                
                double exact = u_analytical(g, x, y, z, t);
                double err = fabs(u[1][b.local_index(i, j, k)] - exact);
                if (err > local_max_err) local_max_err = err;
            }
        }
    }
    
    double global_max_err;
    MPI_Allreduce(&local_max_err, &global_max_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    
    max_inaccuracy = global_max_err;
    first_step_inaccuracy = global_max_err;
    
    if (b.rank == 0)
        cout << "Max start inaccuracy: " << global_max_err << endl;
}

void run_algo(const Grid& g, Block& b, VVEC& u, MPI_Comm comm_cart,
              double& max_inaccuracy, double& last_step_inaccuracy) {
    for (int step = 2; step < TIME_STEPS; ++step) {
        int prev = (step - 2) % 3;
        int curr = (step - 1) % 3;
        int next = step % 3;
        double t = step * g.tau;
        
        // Шаг 1: Вычисление внутренних точек для следующего временного слоя
        for (int i = 1; i <= b.Nx; ++i) {
            int i_global = b.x_start + i - 1;
            double x = g.x_coord(i_global);
            
            for (int j = 1; j <= b.Ny; ++j) {
                int j_global = b.y_start_global + j - 1;
                double y = g.y_coord(j_global);
                
                for (int k = 1; k <= b.Nz; ++k) {
                    int k_global = b.z_start + k - 1;
                    double z = g.z_coord(k_global);
                    
                    // Внутренние точки вычисляем по явной схеме
                    if (i_global > 0 && i_global < g.N && 
                        j_global > 0 && j_global < g.N && 
                        k_global > 0 && k_global < g.N) {
                        u[next][b.local_index(i, j, k)] = 2.0 * u[curr][b.local_index(i, j, k)]
                            - u[prev][b.local_index(i, j, k)]
                            + g.a2 * g.tau * g.tau * laplace_operator(g, b, u[curr], i, j, k);
                    } 
                    // Граничные точки берем из аналитического решения для точности
                    else {
                        u[next][b.local_index(i, j, k)] = u_analytical(g, x, y, z, t);
                    }
                }
            }
        }
        
        // Шаг 2: Обмен гало-ячейками
        exchange_halos(b, u[next]);
        
        // Шаг 3: Применение граничных условий для точек x=0, x=Lx, z=0, z=Lz
        apply_boundary_conditions(g, b, u[next]);
        
        // Шаг 4: Обеспечение периодичности по y
        enforce_periodicity_y(g, b, u[next], comm_cart);
        
        // Шаг 5: Вычисление погрешности
        double local_max_err = 0.0;
        
        for (int i = 1; i <= b.Nx; ++i) {
            int i_global = b.x_start + i - 1;
            double x = g.x_coord(i_global);
            
            for (int j = 1; j <= b.Ny; ++j) {
                int j_global = b.y_start_global + j - 1;
                double y = g.y_coord(j_global);
                
                for (int k = 1; k <= b.Nz; ++k) {
                    int k_global = b.z_start + k - 1;
                    double z = g.z_coord(k_global);
                    
                    double exact = u_analytical(g, x, y, z, t);
                    double err = fabs(u[next][b.local_index(i, j, k)] - exact);
                    if (err > local_max_err) local_max_err = err;
                }
            }
        }
        
        double global_max_err;
        MPI_Allreduce(&local_max_err, &global_max_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        
        if (global_max_err > max_inaccuracy)
            max_inaccuracy = global_max_err;
            
        if (step == TIME_STEPS - 1)
            last_step_inaccuracy = global_max_err;
            
        if (b.rank == 0)
            cout << "Step " << step << ": max inaccuracy = " << global_max_err << endl;
    }
}

void solve_mpi(const Grid& g, Block& b,
               int /*dimx*/, int /*dimy*/, int /*dimz*/,
               MPI_Comm comm_cart,
               double& time,
               double& max_inaccuracy,
               double& first_step_inaccuracy,
               double& last_step_inaccuracy,
               VDOUB& result) {
    int total_size = b.padded_Nx * b.padded_Ny * b.padded_Nz;
    VDOUB u0(total_size), u1(total_size), u2(total_size);
    VVEC u = {u0, u1, u2};
    
    double start = MPI_Wtime();
    
    // Инициализация и первый шаг
    init(g, b, u, comm_cart, max_inaccuracy, first_step_inaccuracy);
    
    // Основной цикл по времени
    run_algo(g, b, u, comm_cart, max_inaccuracy, last_step_inaccuracy);
    
    double end = MPI_Wtime();
    time = end - start;
    
    // Сбор результатов с последнего временного шага
    result.resize(b.Nx * b.Ny * b.Nz);
    int next = (TIME_STEPS - 1) % 3;
    
    for (int i = 1; i <= b.Nx; ++i)
        for (int j = 1; j <= b.Ny; ++j)
            for (int k = 1; k <= b.Nz; ++k) {
                int idx = (i - 1) * b.Ny * b.Nz + (j - 1) * b.Nz + (k - 1);
                result[idx] = u[next][b.local_index(i, j, k)];
            }
}