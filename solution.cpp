#define _USE_MATH_DEFINES
#include "equation.hpp"
#include <algorithm>
#include <iostream>
#include <chrono>
#include <cmath>
#include <mpi.h>

// --- Обмен ghost-слоями ---
void exchange_ghost_layers(Block& b, VDOUB& ui_padded, const MPI_Comm& comm_cart) {
    MPI_Request reqs[12];
    int req_count = 0;

    // --- Обмен по X ---
    if (b.neighbors[0] != MPI_PROC_NULL) { // Есть сосед слева
        int data_size = b.local_Ny * b.local_Nz;
        MPI_Isend(b.send_x_low.data(), data_size, MPI_DOUBLE, b.neighbors[0], b.rank * 6 + 0, comm_cart, &reqs[req_count++]);
        MPI_Irecv(b.recv_x_low.data(), data_size, MPI_DOUBLE, b.neighbors[0], b.neighbors[0] * 6 + 1, comm_cart, &reqs[req_count++]);
    }
    if (b.neighbors[1] != MPI_PROC_NULL) { // Есть сосед справа
        int data_size = b.local_Ny * b.local_Nz;
        MPI_Isend(b.send_x_high.data(), data_size, MPI_DOUBLE, b.neighbors[1], b.rank * 6 + 1, comm_cart, &reqs[req_count++]);
        MPI_Irecv(b.recv_x_high.data(), data_size, MPI_DOUBLE, b.neighbors[1], b.neighbors[1] * 6 + 0, comm_cart, &reqs[req_count++]);
    }

    // --- Обмен по Y ---
    if (b.neighbors[2] != MPI_PROC_NULL) {
        int data_size = b.local_Nx * b.local_Nz;
        MPI_Isend(b.send_y_low.data(), data_size, MPI_DOUBLE, b.neighbors[2], b.rank * 6 + 2, comm_cart, &reqs[req_count++]);
        MPI_Irecv(b.recv_y_low.data(), data_size, MPI_DOUBLE, b.neighbors[2], b.neighbors[2] * 6 + 3, comm_cart, &reqs[req_count++]);
    }
    if (b.neighbors[3] != MPI_PROC_NULL) {
        int data_size = b.local_Nx * b.local_Nz;
        MPI_Isend(b.send_y_high.data(), data_size, MPI_DOUBLE, b.neighbors[3], b.rank * 6 + 3, comm_cart, &reqs[req_count++]);
        MPI_Irecv(b.recv_y_high.data(), data_size, MPI_DOUBLE, b.neighbors[3], b.neighbors[3] * 6 + 2, comm_cart, &reqs[req_count++]);
    }

    // --- Обмен по Z ---
    if (b.neighbors[4] != MPI_PROC_NULL) {
        int data_size = b.local_Nx * b.local_Ny;
        MPI_Isend(b.send_z_low.data(), data_size, MPI_DOUBLE, b.neighbors[4], b.rank * 6 + 4, comm_cart, &reqs[req_count++]);
        MPI_Irecv(b.recv_z_low.data(), data_size, MPI_DOUBLE, b.neighbors[4], b.neighbors[4] * 6 + 5, comm_cart, &reqs[req_count++]);
    }
    if (b.neighbors[5] != MPI_PROC_NULL) {
        int data_size = b.local_Nx * b.local_Ny;
        MPI_Isend(b.send_z_high.data(), data_size, MPI_DOUBLE, b.neighbors[5], b.rank * 6 + 5, comm_cart, &reqs[req_count++]);
        MPI_Irecv(b.recv_z_high.data(), data_size, MPI_DOUBLE, b.neighbors[5], b.neighbors[5] * 6 + 4, comm_cart, &reqs[req_count++]);
    }

    MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);

    // --- Копирование полученных данных в ghost-слои ---
    // Ghost X=0 (low)
    if (b.neighbors[0] != MPI_PROC_NULL) {
        for (int j = 0; j < b.local_Ny; ++j) {
            for (int k = 0; k < b.local_Nz; ++k) {
                ui_padded[b.padded_index(0, j + 1, k + 1)] = b.recv_x_low[j * b.local_Nz + k];
            }
        }
    }
    // Ghost X=Nx+1 (high)
    if (b.neighbors[1] != MPI_PROC_NULL) {
        for (int j = 0; j < b.local_Ny; ++j) {
            for (int k = 0; k < b.local_Nz; ++k) {
                ui_padded[b.padded_index(b.padded_Nx - 1, j + 1, k + 1)] = b.recv_x_high[j * b.local_Nz + k];
            }
        }
    }

    // Ghost Y=0 (low)
    if (b.neighbors[2] != MPI_PROC_NULL) {
        for (int i = 0; i < b.local_Nx; ++i) {
            for (int k = 0; k < b.local_Nz; ++k) {
                ui_padded[b.padded_index(i + 1, 0, k + 1)] = b.recv_y_low[i * b.local_Nz + k];
            }
        }
    }
    // Ghost Y=Ny+1 (high)
    if (b.neighbors[3] != MPI_PROC_NULL) {
        for (int i = 0; i < b.local_Nx; ++i) {
            for (int k = 0; k < b.local_Nz; ++k) {
                ui_padded[b.padded_index(i + 1, b.padded_Ny - 1, k + 1)] = b.recv_y_high[i * b.local_Nz + k];
            }
        }
    }

    // Ghost Z=0 (low)
    if (b.neighbors[4] != MPI_PROC_NULL) {
        for (int i = 0; i < b.local_Nx; ++i) {
            for (int j = 0; j < b.local_Ny; ++j) {
                ui_padded[b.padded_index(i + 1, j + 1, 0)] = b.recv_z_low[i * b.local_Ny + j];
            }
        }
    }
    // Ghost Z=Nz+1 (high)
    if (b.neighbors[5] != MPI_PROC_NULL) {
        for (int i = 0; i < b.local_Nx; ++i) {
            for (int j = 0; j < b.local_Ny; ++j) {
                ui_padded[b.padded_index(i + 1, j + 1, b.padded_Nz - 1)] = b.recv_z_high[i * b.local_Ny + j];
            }
        }
    }
}


// --- Заполнение буферов для отправки ---
void fill_send_buffers(Block& b, const VDOUB& ui_padded) {
    // X-направление: заполняем из внутренних точек, прилегающих к границе
    if (b.neighbors[0] != MPI_PROC_NULL) { // Отправляем влево (low)
        for (int j = 0; j < b.local_Ny; ++j) {
            for (int k = 0; k < b.local_Nz; ++k) {
                b.send_x_low[j * b.local_Nz + k] = ui_padded[b.local_index(0, j, k)]; // i=0 в local -> i=1 в padded
            }
        }
    }
    if (b.neighbors[1] != MPI_PROC_NULL) { // Отправляем вправо (high)
        for (int j = 0; j < b.local_Ny; ++j) {
            for (int k = 0; k < b.local_Nz; ++k) {
                b.send_x_high[j * b.local_Nz + k] = ui_padded[b.local_index(b.local_Nx - 1, j, k)]; // i=local_Nx-1 в local -> i=local_Nx в padded
            }
        }
    }

    // Y-направление
    if (b.neighbors[2] != MPI_PROC_NULL) { // Отправляем вниз (low)
        for (int i = 0; i < b.local_Nx; ++i) {
            for (int k = 0; k < b.local_Nz; ++k) {
                b.send_y_low[i * b.local_Nz + k] = ui_padded[b.local_index(i, 0, k)]; // j=0 в local -> j=1 в padded
            }
        }
    }
    if (b.neighbors[3] != MPI_PROC_NULL) { // Отправляем вверх (high)
        for (int i = 0; i < b.local_Nx; ++i) {
            for (int k = 0; k < b.local_Nz; ++k) {
                b.send_y_high[i * b.local_Nz + k] = ui_padded[b.local_index(i, b.local_Ny - 1, k)]; // j=local_Ny-1 в local -> j=local_Ny в padded
            }
        }
    }

    // Z-направление
    if (b.neighbors[4] != MPI_PROC_NULL) { // Отправляем ближе (low)
        for (int i = 0; i < b.local_Nx; ++i) {
            for (int j = 0; j < b.local_Ny; ++j) {
                b.send_z_low[i * b.local_Ny + j] = ui_padded[b.local_index(i, j, 0)]; // k=0 в local -> k=1 в padded
            }
        }
    }
    if (b.neighbors[5] != MPI_PROC_NULL) { // Отправляем дальше (high)
        for (int i = 0; i < b.local_Nx; ++i) {
            for (int j = 0; j < b.local_Ny; ++j) {
                b.send_z_high[i * b.local_Ny + j] = ui_padded[b.local_index(i, j, b.local_Nz - 1)]; // k=local_Nz-1 в local -> k=local_Nz в padded
            }
        }
    }
}


// --- Оператор Лапласа ---
inline double laplace_operator(const Grid& g, const Block& b, const VDOUB& ui_padded, const int& i_local, const int& j_local, const int& k_local) {
    // Индексы в padded сетке
    int i_pad = i_local + 1;
    int j_pad = j_local + 1;
    int k_pad = k_local + 1;

    // x-направление: граничные условия 1-го рода (обрабатываются отдельно, внутри лапласа используем ghost)
    double d2x = (ui_padded[b.padded_index(i_pad - 1, j_pad, k_pad)] - 2.0 * ui_padded[b.padded_index(i_pad, j_pad, k_pad)] + ui_padded[b.padded_index(i_pad + 1, j_pad, k_pad)]) / (g.h_x * g.h_x);

    // y-направление: периодические условия (обрабатываются через ghost или принудительно в цикле)
    double d2y = (ui_padded[b.padded_index(i_pad, j_pad - 1, k_pad)] - 2.0 * ui_padded[b.padded_index(i_pad, j_pad, k_pad)] + ui_padded[b.padded_index(i_pad, j_pad + 1, k_pad)]) / (g.h_y * g.h_y);

    // z-направление: граничные условия 1-го рода
    double d2z = (ui_padded[b.padded_index(i_pad, j_pad, k_pad - 1)] - 2.0 * ui_padded[b.padded_index(i_pad, j_pad, k_pad)] + ui_padded[b.padded_index(i_pad, j_pad, k_pad + 1)]) / (g.h_z * g.h_z);

    return d2x + d2y + d2z;
}


// --- Инициализация u^0 и u^1 ---
void init(const Grid& g, Block& b, VVEC& u_padded, const int& dim0_n, const int& dim1_n, const int& dim2_n, MPI_Comm& comm_cart, double& max_inaccuracy, double& first_step_inaccuracy) {
    // --- Шаг 1: Заполняем ВСЕ локальные точки u^0 аналитически ---
    for (int i_loc = 0; i_loc < b.local_Nx; ++i_loc) {
        for (int j_loc = 0; j_loc < b.local_Ny; ++j_loc) {
            for (int k_loc = 0; k_loc < b.local_Nz; ++k_loc) {
                int global_i = b.x_start + i_loc;
                int global_j = b.y_start + j_loc;
                int global_k = b.z_start + k_loc;
                double x = global_i * g.h_x;
                double y = global_j * g.h_y;
                double z = global_k * g.h_z;
                u_padded[0][b.local_index(i_loc, j_loc, k_loc)] = u_analytical(g, x, y, z, 0.0);
            }
        }
    }

    // --- Заполнение ghost-слоев u^0 по аналитическому решению или соседям ---
    // Глобальные границы x и z (1-го рода) -> устанавливаем 0.0 или аналитическое значение в ghost
    // Это делается *до* обмена, чтобы ghost-значения на глобальных границах были корректны
    if (b.neighbors[0] == MPI_PROC_NULL) { // Глобальная граница X=0
        for (int j_loc = 0; j_loc < b.local_Ny; ++j_loc) {
            for (int k_loc = 0; k_loc < b.local_Nz; ++k_loc) {
                 // Ghost индекс (i_pad = 0)
                 u_padded[0][b.padded_index(0, j_loc + 1, k_loc + 1)] = 0.0; // u=0 на x=0
            }
        }
    }
    if (b.neighbors[1] == MPI_PROC_NULL) { // Глобальная граница X=Lx
        for (int j_loc = 0; j_loc < b.local_Ny; ++j_loc) {
            for (int k_loc = 0; k_loc < b.local_Nz; ++k_loc) {
                 // Ghost индекс (i_pad = padded_Nx - 1)
                 u_padded[0][b.padded_index(b.padded_Nx - 1, j_loc + 1, k_loc + 1)] = 0.0; // u=0 на x=Lx
            }
        }
    }
    if (b.neighbors[4] == MPI_PROC_NULL) { // Глобальная граница Z=0
        for (int i_loc = 0; i_loc < b.local_Nx; ++i_loc) {
            for (int j_loc = 0; j_loc < b.local_Ny; ++j_loc) {
                 // Ghost индекс (k_pad = 0)
                 u_padded[0][b.padded_index(i_loc + 1, j_loc + 1, 0)] = 0.0; // u=0 на z=0
            }
        }
    }
    if (b.neighbors[5] == MPI_PROC_NULL) { // Глобальная граница Z=Lz
        for (int i_loc = 0; i_loc < b.local_Nx; ++i_loc) {
            for (int j_loc = 0; j_loc < b.local_Ny; ++j_loc) {
                 // Ghost индекс (k_pad = padded_Nz - 1)
                 u_padded[0][b.padded_index(i_loc + 1, j_loc + 1, b.padded_Nz - 1)] = 0.0; // u=0 на z=Lz
            }
        }
    }
    // Глобальные границы y (периодические) -> заполняются аналитически или будут обновлены через периодичность
    if (b.neighbors[2] == MPI_PROC_NULL) { // Глобальная граница Y=0
        for (int i_loc = 0; i_loc < b.local_Nx; ++i_loc) {
            for (int k_loc = 0; k_loc < b.local_Nz; ++k_loc) {
                 int global_i = b.x_start + i_loc;
                 int global_k = b.z_start + k_loc;
                 double x = global_i * g.h_x;
                 double z = global_k * g.h_z;
                 u_padded[0][b.padded_index(i_loc + 1, 0, k_loc + 1)] = u_analytical(g, x, 0.0, z, 0.0);
            }
        }
    }
    if (b.neighbors[3] == MPI_PROC_NULL) { // Глобальная граница Y=Ly
        for (int i_loc = 0; i_loc < b.local_Nx; ++i_loc) {
            for (int k_loc = 0; k_loc < b.local_Nz; ++k_loc) {
                 int global_i = b.x_start + i_loc;
                 int global_k = b.z_start + k_loc;
                 double x = global_i * g.h_x;
                 double z = global_k * g.h_z;
                 u_padded[0][b.padded_index(i_loc + 1, b.padded_Ny - 1, k_loc + 1)] = u_analytical(g, x, g.Ly, z, 0.0);
            }
        }
    }

    // Обмен ghost-слоями для u^0
    fill_send_buffers(b, u_padded[0]);
    exchange_ghost_layers(b, u_padded[0], comm_cart);

    // --- Шаг 2: Вычисляем u^1 по формуле (12) ТОЛЬКО для внутренних точек ---
    for (int i_loc = 0; i_loc < b.local_Nx; ++i_loc) {
        for (int j_loc = 0; j_loc < b.local_Ny; ++j_loc) {
            for (int k_loc = 0; k_loc < b.local_Nz; ++k_loc) {
                // Не вычисляем на глобальных границах x и z
                int global_i = b.x_start + i_loc;
                int global_k = b.z_start + k_loc;
                if ((global_i == 0 || global_i == g.N) || (global_k == 0 || global_k == g.N)) {
                    continue;
                }
                u_padded[1][b.local_index(i_loc, j_loc, k_loc)] = u_padded[0][b.local_index(i_loc, j_loc, k_loc)]
                    + 0.5 * g.a2 * g.tau * g.tau * laplace_operator(g, b, u_padded[0], i_loc, j_loc, k_loc);
            }
        }
    }

    // --- Шаг 3: Граничные точки для u^1 — копируем из аналитического решения ---
    for (int i_loc = 0; i_loc < b.local_Nx; ++i_loc) {
        for (int j_loc = 0; j_loc < b.local_Ny; ++j_loc) {
            for (int k_loc = 0; k_loc < b.local_Nz; ++k_loc) {
                int global_i = b.x_start + i_loc;
                int global_j = b.y_start + j_loc;
                int global_k = b.z_start + k_loc;
                // Проверяем, является ли точка глобальной границей
                if (global_i == 0 || global_i == g.N || global_j == 0 || global_j == g.N || global_k == 0 || global_k == g.N) {
                    double x = global_i * g.h_x;
                    double y = global_j * g.h_y;
                    double z = global_k * g.h_z;
                    u_padded[1][b.local_index(i_loc, j_loc, k_loc)] = u_analytical(g, x, y, z, g.tau);
                }
            }
        }
    }

    // --- Заполнение ghost-слоев u^1 по аналитическому решению или соседям ---
    if (b.neighbors[0] == MPI_PROC_NULL) { // Глобальная граница X=0
        for (int j_loc = 0; j_loc < b.local_Ny; ++j_loc) {
            for (int k_loc = 0; k_loc < b.local_Nz; ++k_loc) {
                 u_padded[1][b.padded_index(0, j_loc + 1, k_loc + 1)] = 0.0;
            }
        }
    }
    if (b.neighbors[1] == MPI_PROC_NULL) { // Глобальная граница X=Lx
        for (int j_loc = 0; j_loc < b.local_Ny; ++j_loc) {
            for (int k_loc = 0; k_loc < b.local_Nz; ++k_loc) {
                 u_padded[1][b.padded_index(b.padded_Nx - 1, j_loc + 1, k_loc + 1)] = 0.0;
            }
        }
    }
    if (b.neighbors[4] == MPI_PROC_NULL) { // Глобальная граница Z=0
        for (int i_loc = 0; i_loc < b.local_Nx; ++i_loc) {
            for (int j_loc = 0; j_loc < b.local_Ny; ++j_loc) {
                 u_padded[1][b.padded_index(i_loc + 1, j_loc + 1, 0)] = 0.0;
            }
        }
    }
    if (b.neighbors[5] == MPI_PROC_NULL) { // Глобальная граница Z=Lz
        for (int i_loc = 0; i_loc < b.local_Nx; ++i_loc) {
            for (int j_loc = 0; j_loc < b.local_Ny; ++j_loc) {
                 u_padded[1][b.padded_index(i_loc + 1, j_loc + 1, b.padded_Nz - 1)] = 0.0;
            }
        }
    }
    if (b.neighbors[2] == MPI_PROC_NULL) { // Глобальная граница Y=0
        for (int i_loc = 0; i_loc < b.local_Nx; ++i_loc) {
            for (int k_loc = 0; k_loc < b.local_Nz; ++k_loc) {
                 int global_i = b.x_start + i_loc;
                 int global_k = b.z_start + k_loc;
                 double x = global_i * g.h_x;
                 double z = global_k * g.h_z;
                 u_padded[1][b.padded_index(i_loc + 1, 0, k_loc + 1)] = u_analytical(g, x, 0.0, z, g.tau);
            }
        }
    }
    if (b.neighbors[3] == MPI_PROC_NULL) { // Глобальная граница Y=Ly
        for (int i_loc = 0; i_loc < b.local_Nx; ++i_loc) {
            for (int k_loc = 0; k_loc < b.local_Nz; ++k_loc) {
                 int global_i = b.x_start + i_loc;
                 int global_k = b.z_start + k_loc;
                 double x = global_i * g.h_x;
                 double z = global_k * g.h_z;
                 u_padded[1][b.padded_index(i_loc + 1, b.padded_Ny - 1, k_loc + 1)] = u_analytical(g, x, g.Ly, z, g.tau);
            }
        }
    }

    // Обмен ghost-слоями для u^1
    fill_send_buffers(b, u_padded[1]);
    exchange_ghost_layers(b, u_padded[1], comm_cart);

    // --- Шаг 4: Принудительно обеспечиваем периодичность по y для обоих слоёв ---
    if (b.neighbors[2] == MPI_PROC_NULL) { // Процесс на границе Y=0
        for (int i_loc = 0; i_loc < b.local_Nx; ++i_loc) {
            for (int k_loc = 0; k_loc < b.local_Nz; ++k_loc) {
                u_padded[0][b.padded_index(i_loc + 1, b.padded_Ny - 1, k_loc + 1)] = u_padded[0][b.padded_index(i_loc + 1, 1, k_loc + 1)]; // j=Ny+1 = j=1 (ghost)
                u_padded[1][b.padded_index(i_loc + 1, b.padded_Ny - 1, k_loc + 1)] = u_padded[1][b.padded_index(i_loc + 1, 1, k_loc + 1)]; // j=Ny+1 = j=1 (ghost)
            }
        }
    }
    if (b.neighbors[3] == MPI_PROC_NULL) { // Процесс на границе Y=Ly
        for (int i_loc = 0; i_loc < b.local_Nx; ++i_loc) {
            for (int k_loc = 0; k_loc < b.local_Nz; ++k_loc) {
                u_padded[0][b.padded_index(i_loc + 1, 0, k_loc + 1)] = u_padded[0][b.padded_index(i_loc + 1, b.padded_Ny - 2, k_loc + 1)]; // j=0 = j=Ny (ghost)
                u_padded[1][b.padded_index(i_loc + 1, 0, k_loc + 1)] = u_padded[1][b.padded_index(i_loc + 1, b.padded_Ny - 2, k_loc + 1)]; // j=0 = j=Ny (ghost)
            }
        }
    }

    // --- Шаг 5: Проверка погрешности на u^1 ---
    double local_error = 0.0;
    for (int i_loc = 0; i_loc < b.local_Nx; ++i_loc) {
        for (int j_loc = 0; j_loc < b.local_Ny; ++j_loc) {
            for (int k_loc = 0; k_loc < b.local_Nz; ++k_loc) {
                int global_i = b.x_start + i_loc;
                int global_j = b.y_start + j_loc;
                int global_k = b.z_start + k_loc;
                double x = global_i * g.h_x;
                double y = global_j * g.h_y;
                double z = global_k * g.h_z;
                double exact = u_analytical(g, x, y, z, g.tau);
                double computed = u_padded[1][b.local_index(i_loc, j_loc, k_loc)];
                double err = std::abs(computed - exact);
                local_error = std::max(local_error, err);
            }
        }
    }
    double step_max_error = -1.0;
    MPI_Reduce(&local_error, &step_max_error, 1, MPI_DOUBLE, MPI_MAX, 0, comm_cart);
    if (b.rank == 0) {
        max_inaccuracy = std::max(max_inaccuracy, step_max_error);
        first_step_inaccuracy = step_max_error;
        std::cout << "Steps inaccuracy:\n\tMax inaccuracy on step 1 = " << step_max_error << std::endl;
    }
}


// --- Основной цикл по времени ---
void run_algo(const Grid& g, Block& b, VVEC& u_padded, const int& dim0_n, const int& dim1_n, const int& dim2_n, MPI_Comm& comm_cart, double& max_inaccuracy, double& last_step_inaccuracy) {
    int next, curr, prev;
    for (int s = 2; s < TIME_STEPS; ++s) {
        next = s % 3;
        curr = (s - 1) % 3;
        prev = (s - 2) % 3;

        // --- Обмен ghost-слоев для текущего слоя ---
        fill_send_buffers(b, u_padded[curr]);
        exchange_ghost_layers(b, u_padded[curr], comm_cart);

        // --- 1. Внутренние точки: стандартная схема ---
        for (int i_loc = 0; i_loc < b.local_Nx; ++i_loc) {
            for (int j_loc = 0; j_loc < b.local_Ny; ++j_loc) {
                for (int k_loc = 0; k_loc < b.local_Nz; ++k_loc) {
                    // Пропускаем глобальные границы x и z
                    int global_i = b.x_start + i_loc;
                    int global_k = b.z_start + k_loc;
                    if ((global_i == 0 || global_i == g.N) || (global_k == 0 || global_k == g.N)) {
                        continue;
                    }
                    u_padded[next][b.local_index(i_loc, j_loc, k_loc)] = 2.0 * u_padded[curr][b.local_index(i_loc, j_loc, k_loc)]
                        - u_padded[prev][b.local_index(i_loc, j_loc, k_loc)]
                        + g.a2 * g.tau * g.tau * laplace_operator(g, b, u_padded[curr], i_loc, j_loc, k_loc);
                }
            }
        }

        // --- 2. Граничные точки x=0, x=N (1-го рода) ---
        for (int j_loc = 0; j_loc < b.local_Ny; ++j_loc) {
            for (int k_loc = 0; k_loc < b.local_Nz; ++k_loc) {
                if (b.x_start == 0) { // Процесс на глобальной границе x=0
                     u_padded[next][b.local_index(0, j_loc, k_loc)] = 0.0; // i_loc=0 соответствует global_i=x_start=0
                }
                if (b.x_end == g.N) { // Процесс на глобальной границе x=N
                     int local_i = b.local_Nx - 1; // Последний индекс в блоке
                     if (b.x_start + local_i == g.N) { // Проверяем, соответствует ли он глобальному N
                         u_padded[next][b.local_index(local_i, j_loc, k_loc)] = 0.0;
                     }
                }
            }
        }

        // --- 3. Граничные точки z=0, z=N (1-го рода) ---
        for (int i_loc = 0; i_loc < b.local_Nx; ++i_loc) {
            for (int j_loc = 0; j_loc < b.local_Ny; ++j_loc) {
                if (b.z_start == 0) { // Процесс на глобальной границе z=0
                     u_padded[next][b.local_index(i_loc, j_loc, 0)] = 0.0; // k_loc=0 соответствует global_k=z_start=0
                }
                if (b.z_end == g.N) { // Процесс на глобальной границе z=N
                     int local_k = b.local_Nz - 1; // Последний индекс в блоке
                     if (b.z_start + local_k == g.N) { // Проверяем, соответствует ли он глобальному N
                         u_padded[next][b.local_index(i_loc, j_loc, local_k)] = 0.0;
                     }
                }
            }
        }

        // --- 4. ВСЕ ОСТАЛЬНЫЕ ГРАНИЧНЫЕ ТОЧКИ (включая y-границы в углах и на ребрах) — из аналитики ---
        for (int i_loc = 0; i_loc < b.local_Nx; ++i_loc) {
            for (int j_loc = 0; j_loc < b.local_Ny; ++j_loc) {
                for (int k_loc = 0; k_loc < b.local_Nz; ++k_loc) {
                    int global_i = b.x_start + i_loc;
                    int global_j = b.y_start + j_loc;
                    int global_k = b.z_start + k_loc;
                    // Пропускаем уже обработанные точки (x=0, x=N, z=0, z=N)
                    if (global_i == 0 || global_i == g.N || global_k == 0 || global_k == g.N) continue;
                    // Обрабатываем оставшиеся — в частности, y=0 и y=N при 0<x<N, 0<z<N
                    if (global_j == 0 || global_j == g.N) {
                        double x = global_i * g.h_x;
                        double y = global_j * g.h_y;
                        double z = global_k * g.h_z;
                        u_padded[next][b.local_index(i_loc, j_loc, k_loc)] = u_analytical(g, x, y, z, s * g.tau);
                    }
                }
            }
        }

        // --- 5. Принудительно обеспечиваем ПЕРИОДИЧНОСТЬ по y (включая углы и ребра!) ---
        if (b.neighbors[2] == MPI_PROC_NULL) { // Процесс на границе Y=0
            for (int i_loc = 0; i_loc < b.local_Nx; ++i_loc) {
                for (int k_loc = 0; k_loc < b.local_Nz; ++k_loc) {
                    u_padded[next][b.padded_index(i_loc + 1, b.padded_Ny - 1, k_loc + 1)] = u_padded[next][b.padded_index(i_loc + 1, 1, k_loc + 1)]; // j=Ny+1 = j=1 (ghost)
                }
            }
        }
        if (b.neighbors[3] == MPI_PROC_NULL) { // Процесс на границе Y=Ly
            for (int i_loc = 0; i_loc < b.local_Nx; ++i_loc) {
                for (int k_loc = 0; k_loc < b.local_Nz; ++k_loc) {
                    u_padded[next][b.padded_index(i_loc + 1, 0, k_loc + 1)] = u_padded[next][b.padded_index(i_loc + 1, b.padded_Ny - 2, k_loc + 1)]; // j=0 = j=Ny (ghost)
                }
            }
        }

        // --- 6. Подсчёт погрешности ---
        double local_error = 0.0;
        for (int i_loc = 0; i_loc < b.local_Nx; ++i_loc) {
            for (int j_loc = 0; j_loc < b.local_Ny; ++j_loc) {
                for (int k_loc = 0; k_loc < b.local_Nz; ++k_loc) {
                    int global_i = b.x_start + i_loc;
                    int global_j = b.y_start + j_loc;
                    int global_k = b.z_start + k_loc;
                    double x = global_i * g.h_x;
                    double y = global_j * g.h_y;
                    double z = global_k * g.h_z;
                    double exact = u_analytical(g, x, y, z, s * g.tau);
                    double computed = u_padded[next][b.local_index(i_loc, j_loc, k_loc)];
                    double err = std::abs(computed - exact);
                    local_error = std::max(local_error, err);
                }
            }
        }
        double step_max_error = -1.0;
        MPI_Reduce(&local_error, &step_max_error, 1, MPI_DOUBLE, MPI_MAX, 0, comm_cart);
        if (b.rank == 0) {
            max_inaccuracy = std::max(max_inaccuracy, step_max_error);
            if (s == TIME_STEPS - 1) last_step_inaccuracy = step_max_error;
            std::cout << "\tMax inaccuracy on step " << s << " = " << step_max_error << std::endl;
        }
    }
}


// --- Основная функция решения ---
void solve_equation(const Grid& g, Block& b, const int& dim0_n, const int& dim1_n, const int& dim2_n, MPI_Comm& comm_cart, double& time, double& max_inaccuracy, double& first_step_inaccuracy, double& last_step_inaccuracy, VDOUB& result) {
    VDOUB u0_local(b.N), u1_local(b.N), u2_local(b.N);
    VVEC u_local{u0_local, u1_local, u2_local};

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    init(g, b, u_local, dim0_n, dim1_n, dim2_n, comm_cart, max_inaccuracy, first_step_inaccuracy);
    run_algo(g, b, u_local, dim0_n, dim1_n, dim2_n, comm_cart, max_inaccuracy, last_step_inaccuracy);
    double end_time = MPI_Wtime();
    double local_time = end_time - start_time;
    MPI_Reduce(&local_time, &time, 1, MPI_DOUBLE, MPI_MAX, 0, comm_cart);
    result = u_local[(TIME_STEPS - 1) % 3]; // Возвращаем результат из нужного слоя
}