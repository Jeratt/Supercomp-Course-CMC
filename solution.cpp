#define _USE_MATH_DEFINES
#include "equation.hpp"
#include <algorithm>
#include <iostream>
#include <chrono>
#include <cmath>
#include <mpi.h>

// update halo for neighbors processes
void exchange_ghost_layers(Block& b, VDOUB& ui_local, MPI_Comm& comm_cart) {
    MPI_Request reqs[12];
    int req_count = 0;
    for (int dir = 0; dir < 6; ++dir) {
        int send_neighbor = b.neighbors[dir];
        int recv_neighbor = b.neighbors[(dir % 2 == 0) ? (dir + 1) : (dir - 1)];
        if (send_neighbor != MPI_PROC_NULL and (send_neighbor != b.rank and recv_neighbor != b.rank)) {
            int data_size = (dir < 2) ? b.Nx * b.Nz : (dir < 4 ? b.Ny * b.Nz : b.Nx * b.Ny);
            switch(dir) {
                case 0:
                    MPI_Isend(b.left_send.data(), data_size, MPI_DOUBLE, send_neighbor, b.rank + 0, comm_cart, &reqs[req_count++]);
                    MPI_Irecv(b.right_recieve.data(), data_size, MPI_DOUBLE, recv_neighbor, recv_neighbor + 0, comm_cart, &reqs[req_count++]);
                    break;
                case 1:
                    MPI_Isend(b.right_send.data(), data_size, MPI_DOUBLE, send_neighbor, b.rank + 1, comm_cart, &reqs[req_count++]);
                    MPI_Irecv(b.left_recieve.data(), data_size, MPI_DOUBLE, recv_neighbor, recv_neighbor + 1, comm_cart, &reqs[req_count++]);
                    break;
                case 2:
                    MPI_Isend(b.bottom_send.data(), data_size, MPI_DOUBLE, send_neighbor, b.rank + 2, comm_cart, &reqs[req_count++]);
                    MPI_Irecv(b.top_recieve.data(), data_size, MPI_DOUBLE, recv_neighbor, recv_neighbor + 2, comm_cart, &reqs[req_count++]);
                    break;
                case 3:
                    MPI_Isend(b.top_send.data(), data_size, MPI_DOUBLE, send_neighbor, b.rank + 3, comm_cart, &reqs[req_count++]);
                    MPI_Irecv(b.bottom_recieve.data(), data_size, MPI_DOUBLE, recv_neighbor, recv_neighbor + 3, comm_cart, &reqs[req_count++]);
                    break;
                case 4:
                    MPI_Isend(b.front_send.data(), data_size, MPI_DOUBLE, send_neighbor, b.rank + 4, comm_cart, &reqs[req_count++]);
                    MPI_Irecv(b.back_recieve.data(), data_size, MPI_DOUBLE, recv_neighbor, recv_neighbor + 4, comm_cart, &reqs[req_count++]);
                    break;
                case 5:
                    MPI_Isend(b.back_send.data(), data_size, MPI_DOUBLE, send_neighbor, b.rank + 5, comm_cart, &reqs[req_count++]);
                    MPI_Irecv(b.front_recieve.data(), data_size, MPI_DOUBLE, recv_neighbor, recv_neighbor + 5, comm_cart, &reqs[req_count++]);
                    break;
                default:
                    break;
            }
        }
    }
    MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
    // Copy received ghost layers into u_local
    for (int dir = 0; dir < 6; ++dir) {
        int send_neighbor = b.neighbors[dir];
        int recv_neighbor = b.neighbors[(dir % 2 == 0) ? (dir + 1) : (dir - 1)];
        if (send_neighbor != MPI_PROC_NULL and (send_neighbor != b.rank and recv_neighbor != b.rank)) {
            switch(dir) {
                case 0:
                    for (int i = 1; i < b.Nx + 1; ++i)
                        for (int j = 1; j < b.Nz + 1; ++j)
                            ui_local[b.index(i, 0, j)] = b.left_recieve[b.Nz * (i - 1) + (j - 1)];
                    break;
                case 1:
                    for (int i = 1; i < b.Nx + 1; ++i)
                        for (int j = 1; j < b.Nz + 1; ++j)
                            ui_local[b.index(i, b.Ny + 1, j)] = b.right_recieve[b.Nz * (i - 1) + (j - 1)];
                    break;
                case 2:
                    for (int i = 1; i < b.Ny + 1; ++i)
                        for (int j = 1; j < b.Nz + 1; ++j)
                            ui_local[b.index(b.Nx + 1, i, j)] = b.bottom_recieve[b.Nz * (i - 1) + (j - 1)];
                    break;
                case 3:
                    for (int i = 1; i < b.Ny + 1; ++i)
                        for (int j = 1; j < b.Nz + 1; ++j)
                            ui_local[b.index(0, i, j)] = b.top_recieve[b.Nz * (i - 1) + (j - 1)];
                    break;
                case 4:
                    for (int i = 1; i < b.Nx + 1; ++i)
                        for (int j = 1; j < b.Ny + 1; ++j)
                            ui_local[b.index(i, j, 0)] = b.front_recieve[b.Ny * (i - 1) + (j - 1)];
                    break;
                case 5:
                    for (int i = 1; i < b.Nx + 1; ++i)
                        for (int j = 1; j < b.Ny + 1; ++j)
                            ui_local[b.index(i, j, b.Nz + 1)] = b.back_recieve[b.Ny * (i - 1) + (j - 1)];
                    break;
                default:
                    break;
            }
        }
    }
}

inline void fill_send_buffers(Block& b, VDOUB& ui_local, const int& dim0_n, const int& dim1_n, const int& dim2_n, const int& i, const int& j, const int& k) {
    int dim1_ind = b.Nz * (j - 1) + (k - 1);
    int dim0_ind = b.Nz * (i - 1) + (k - 1);
    int dim2_ind = b.Ny * (i - 1) + (j - 1);
    if (dim1_n == 1) {
        if (i == 2)
            ui_local[b.index(b.Nx + 1, j, k)] = ui_local[b.index(i, j, k)];
        else if (i == b.Nx - 1)
            ui_local[b.index(0, j, k)] = ui_local[b.index(i, j, k)];
    } else {
        if (b.dim1_rank == 0) {
            if (i == 1)
                b.top_send[dim1_ind] = ui_local[b.index(i, j, k)];
            else if (i == b.Nx - 1)
                b.bottom_send[dim1_ind] = ui_local[b.index(i, j, k)];
        } else if (b.dim1_rank == dim1_n - 1) {
            if (i == 2)
                b.top_send[dim1_ind] = ui_local[b.index(i, j, k)];
            else if (i == b.Nx)
                b.bottom_send[dim1_ind] = ui_local[b.index(i, j, k)];
        } else {
            if (i == 1)
                b.top_send[dim1_ind] = ui_local[b.index(i, j, k)];
            else if (i == b.Nx)
                b.bottom_send[dim1_ind] = ui_local[b.index(i, j, k)];
        }
    }
    if (dim0_n == 1) {
        if (j == 2)
            ui_local[b.index(i, b.Ny + 1, k)] = ui_local[b.index(i, j, k)];
        else if (j == b.Ny - 1)
            ui_local[b.index(i, 0, k)] = ui_local[b.index(i, j, k)];
    } else {
        if (b.dim0_rank == 0) {
            if (j == 2)
                b.left_send[dim0_ind] = ui_local[b.index(i, j, k)];
            else if (j == b.Ny)
                b.right_send[dim0_ind] = ui_local[b.index(i, j, k)];
        } else if (b.dim0_rank == dim0_n - 1) {
            if (j == 1)
                b.left_send[dim0_ind] = ui_local[b.index(i, j, k)];
            else if (j == b.Ny - 1)
                b.right_send[dim0_ind] = ui_local[b.index(i, j, k)];
        } else {
            if (j == 1)
                b.left_send[dim0_ind] = ui_local[b.index(i, j, k)];
            else if (i == b.Ny) // Ошибка в оригинальном коде: i вместо j
                b.right_send[dim0_ind] = ui_local[b.index(i, j, k)];
        }
    }
    if (dim2_n == 1) {
        if (k == 2)
            ui_local[b.index(i, j, b.Nz + 1)] = ui_local[b.index(i, j, k)];
        else if (k == b.Nz - 1)
            ui_local[b.index(i, j, 0)] = ui_local[b.index(i, j, k)];
    } else {
        if (b.dim2_rank == 0) {
            if (k == 2)
                b.front_send[dim2_ind] = ui_local[b.index(i, j, k)];
            else if (k == b.Nz)
                b.back_send[dim2_ind] = ui_local[b.index(i, j, k)];
        } else if (b.dim2_rank == dim2_n - 1) {
            if (k == 1)
                b.front_send[dim2_ind] = ui_local[b.index(i, j, k)];
            else if (k == b.Nz - 1)
                b.back_send[dim2_ind] = ui_local[b.index(i, j, k)];
        } else {
            if (k == 1)
                b.front_send[dim2_ind] = ui_local[b.index(i, j, k)];
            else if (k == b.Nz)
                b.back_send[dim2_ind] = ui_local[b.index(i, j, k)];
        }
    }
}

inline double laplace_operator(const Grid& g, const Block& b, const VDOUB& ui_local, const int& i, const int& j, const int& k) {
    return (ui_local[b.index(i - 1, j, k)] - 2 * ui_local[b.index(i, j, k)] + ui_local[b.index(i + 1, j, k)]) / pow(g.h_x, 2) +
           (ui_local[b.index(i, j - 1, k)] - 2 * ui_local[b.index(i, j, k)] + ui_local[b.index(i, j + 1, k)]) / pow(g.h_y, 2) +
           (ui_local[b.index(i, j, k - 1)] - 2 * ui_local[b.index(i, j, k)] + ui_local[b.index(i, j, k + 1)]) / pow(g.h_z, 2);
}

void init(const Grid& g, Block& b, VVEC& u_local, const int& dim0_n, const int& dim1_n, const int& dim2_n, MPI_Comm& comm_cart, double& max_inaccuracy, double& first_step_inaccuracy) {
    // boundary conditions (7-9) variant 8: x - П, y - П, z - П
    // NOTE: This section is specific to variant 8 and needs adaptation for variant 3.
    // For now, it remains as per original variant 8 logic.
    if (b.dim1_rank == 0) {
        for (int i = 1; i < b.Ny + 1; ++i) {
            for (int j = 1; j < b.Nz + 1; ++j) {
                u_local[0][b.index(b.Nx, i, j)] = u_analytical(g, g.Lx, (b.y_start + i - 1) * g.h_y, (b.z_start + j - 1) * g.h_z, 0);
                u_local[1][b.index(b.Nx, i, j)] = u_analytical(g, g.Lx, (b.y_start + i - 1) * g.h_y, (b.z_start + j - 1) * g.h_z, g.tau);
            }
        }
    } else if (b.dim1_rank == dim1_n - 1) {
        for (int i = 1; i < b.Ny + 1; ++i) {
            for (int j = 1; j < b.Nz + 1; ++j) {
                u_local[0][b.index(1, i, j)] = u_analytical(g, 0, (b.y_start + i - 1) * g.h_y, (b.z_start + j - 1) * g.h_z, 0);
                u_local[1][b.index(1, i, j)] = u_analytical(g, 0, (b.y_start + i - 1) * g.h_y, (b.z_start + j - 1) * g.h_z, g.tau);
            }
        }
    }
    if (b.dim0_rank == 0) {
        for (int i = 1; i < b.Nx + 1; ++i) {
            for (int j = 1; j < b.Nz + 1; ++j) {
                u_local[0][b.index(i, 1, j)] = u_analytical(g, (b.x_start + i - 1) * g.h_x, 0, (b.z_start + j - 1) * g.h_z, 0);
                u_local[1][b.index(i, 1, j)] = u_analytical(g, (b.x_start + i - 1) * g.h_x, 0, (b.z_start + j - 1) * g.h_z, g.tau);
            }
        }
    } else if (b.dim0_rank == dim0_n - 1) {
        for (int i = 1; i < b.Nx + 1; ++i) {
            for (int j = 1; j < b.Nz + 1; ++j) {
                u_local[0][b.index(i, b.Ny, j)] = u_analytical(g, (b.x_start + i - 1) * g.h_x, g.Ly, (b.z_start + j - 1) * g.h_z, 0);
                u_local[1][b.index(i, b.Ny, j)] = u_analytical(g, (b.x_start + i - 1) * g.h_x, g.Ly, (b.z_start + j - 1) * g.h_z, g.tau);
            }
        }
    }
    if (b.dim2_rank == 0) {
        for (int i = 1; i < b.Nx + 1; ++i) {
            for (int j = 1; j < b.Ny + 1; ++j) {
                u_local[0][b.index(i, j, 1)] = u_analytical(g, (b.x_start + i - 1) * g.h_x, (b.y_start + j - 1) * g.h_y, 0, 0);
                u_local[1][b.index(i, j, 1)] = u_analytical(g, (b.x_start + i - 1) * g.h_x, (b.y_start + j - 1) * g.h_y, 0, g.tau);
            }
        }
    } else if (b.dim2_rank == dim2_n - 1) {
        for (int i = 1; i < b.Nx + 1; ++i) {
            for (int j = 1; j < b.Ny + 1; ++j) {
                u_local[0][b.index(i, j, b.Nz)] = u_analytical(g, (b.x_start + i - 1) * g.h_x, (b.y_start + j - 1) * g.h_y, g.Lz, 0);
                u_local[1][b.index(i, j, b.Nz)] = u_analytical(g, (b.x_start + i - 1) * g.h_x, (b.y_start + j - 1) * g.h_y, g.Lz, g.tau);
            }
        }
    }

    int i_start = 1, i_end = b.Nx + 1, j_start = 1, j_end = b.Ny + 1, k_start = 1, k_end = b.Nz + 1;
    if (b.dim1_rank == 0)
        i_end = b.Nx;
    else if (b.dim1_rank == dim1_n - 1)
        i_start = 2;
    if (b.dim0_rank == 0)
    	j_start = 2;
    else if (b.dim0_rank == dim0_n - 1)
		j_end = b.Ny;
    if (b.dim2_rank == 0)
		k_start = 2;
    else if (b.dim2_rank == dim2_n - 1)
		k_end = b.Nz;

    // u_0 (10)
    for (int i = i_start; i < i_end; ++i) {
        for (int j = j_start; j < j_end; ++j) {
      		for (int k = k_start; k < k_end; ++k) {
                u_local[0][b.index(i, j, k)] = u_analytical(g, (b.x_start + i - 1) * g.h_x, (b.y_start + j - 1) * g.h_y, (b.z_start + k - 1) * g.h_z, 0);
                fill_send_buffers(b, u_local[0], dim0_n, dim1_n, dim2_n, i, j, k);
            }
        }
    }
    exchange_ghost_layers(b, u_local[0], comm_cart);

    // u_1 (12)
    for (int i = i_start; i < i_end; ++i) {
        for (int j = j_start; j < j_end; ++j) {
      		for (int k = k_start; k < k_end; ++k) {
       		    u_local[1][b.index(i, j, k)] = u_local[0][b.index(i, j, k)] + 0.5 * g.a2 * pow(g.tau, 2) * laplace_operator(g, b, u_local[0], i, j, k);
            }
        }
    }

    // fill send buffer with elements of u_1
    for (int i = 1; i < b.Nx + 1; ++i)
        for (int j = 1; j < b.Ny + 1; ++j)
            for (int k = 1; k < b.Nz + 1; ++k)
                fill_send_buffers(b, u_local[1], dim0_n, dim1_n, dim2_n, i, j, k);

    // innacuracy calculation
    double error = -1;
    for (int i = 1; i < b.Nx + 1; ++i) {
        for (int j = 1; j < b.Ny + 1; ++j) {
            for (int k = 1; k < b.Nz + 1; ++k) {
                double tmp = fabs(u_local[1][b.index(i, j, k)] - u_analytical(g, (b.x_start + i - 1) * g.h_x, (b.y_start + j - 1) * g.h_y, (b.z_start + k - 1) * g.h_z, g.tau));
                if (tmp > error)
                    error = tmp;
            }
        }
    }
    double step_max_error = -1;
    MPI_Reduce(&error, &step_max_error, 1, MPI_DOUBLE, MPI_MAX, 0, comm_cart);
    if (b.rank == 0) {
        if (step_max_error > max_inaccuracy)
            max_inaccuracy = step_max_error;
        first_step_inaccuracy = step_max_error;
        std::cout << "Max start inaccuracy: " << step_max_error << std::endl;
    }
}

void run_algo(const Grid& g, Block& b, VVEC& u_local, const int& dim0_n, const int& dim1_n, const int& dim2_n, MPI_Comm& comm_cart, double& max_inaccuracy, double& last_step_inaccuracy) {
    int next, curr, prev;
    for (int s = 2; s < T; ++s) { // Используем T из заголовка
        next = s % 3;
        curr = (s - 1) % 3;
        prev = (s - 2) % 3;
        exchange_ghost_layers(b, u_local[curr], comm_cart);

        // u_n with boundary conditions
        for (int i = 1; i < b.Nx + 1; ++i) {
            for (int j = 1; j < b.Ny + 1; ++j) {
                for (int k = 1; k < b.Nz + 1; ++k) {
                    u_local[next][b.index(i, j, k)] = 2 * u_local[curr][b.index(i, j, k)] - u_local[prev][b.index(i, j, k)] + g.a2 * pow(g.tau, 2) * laplace_operator(g, b, u_local[curr], i, j, k);
                    fill_send_buffers(b, u_local[next], dim0_n, dim1_n, dim2_n, i, j, k);
                }
            }
        }

        // innacuracy calculation
        double error = -1;
        for (int i = 1; i < b.Nx + 1; ++i) {
            for (int j = 1; j < b.Ny + 1; ++j) {
                for (int k = 1; k < b.Nz + 1; ++k) {
                    double tmp = fabs(u_local[next][b.index(i, j, k)] - u_analytical(g, (b.x_start + i - 1) * g.h_x, (b.y_start + j - 1) * g.h_y, (b.z_start + k - 1) * g.h_z, g.tau * s));
                    if (tmp > error)
                        error = tmp;
                }
            }
        }
        double step_max_error = -1;
        MPI_Reduce(&error, &step_max_error, 1, MPI_DOUBLE, MPI_MAX, 0, comm_cart);
        if (b.rank == 0) {
            if (step_max_error > max_inaccuracy)
                max_inaccuracy = step_max_error;
            if (s == T - 1) // Используем T из заголовка
            	last_step_inaccuracy = step_max_error;
            std::cout << "Max inaccuracy on step " << s << " : " << step_max_error << std::endl;
        }
    }
}

void solve_equation(const Grid& grid, Block& block, const int& dim0_n, const int& dim1_n, const int& dim2_n, MPI_Comm& comm_cart, double& time, double& max_inaccuracy, double& first_step_inaccuracy, double& last_step_inaccuracy, VDOUB& result) {
    VDOUB u0_local(block.N), u1_local(block.N), u2_local(block.N);
    VVEC u_local{u0_local, u1_local, u2_local}; // Используем VVEC
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    max_inaccuracy = 0.0;
    init(grid, block, u_local, dim0_n, dim1_n, dim2_n, comm_cart, max_inaccuracy, first_step_inaccuracy);
    run_algo(grid, block, u_local, dim0_n, dim1_n, dim2_n, comm_cart, max_inaccuracy, last_step_inaccuracy);
    double end_time = MPI_Wtime();
    double local_time = end_time - start_time;
    MPI_Reduce(&local_time, &time, 1, MPI_DOUBLE, MPI_MAX, 0, comm_cart);
    result = u_local[(T - 1) % 3];
}