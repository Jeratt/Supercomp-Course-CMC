#define _USE_MATH_DEFINES
#include "equation.hpp"
#include <mpi.h>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <vector>
#include <omp.h>   // OpenMP

using namespace std;

inline double laplace_operator(const Grid& g, const Block& b, const VDOUB& u, int i, int j, int k) {
    double d2x = (u[b.local_index(i - 1, j, k)] - 2.0 * u[b.local_index(i, j, k)] + u[b.local_index(i + 1, j, k)]) / (g.h_x * g.h_x);
    double d2y = (u[b.local_index(i, j - 1, k)] - 2.0 * u[b.local_index(i, j, k)] + u[b.local_index(i, j + 1, k)]) / (g.h_y * g.h_y);
    double d2z = (u[b.local_index(i, j, k - 1)] - 2.0 * u[b.local_index(i, j, k)] + u[b.local_index(i, j, k + 1)]) / (g.h_z * g.h_z);
    return d2x + d2y + d2z;
}

void exchange_halos(Block& b, VDOUB& u) {
    const int tag_left = 1, tag_right = 2,
              tag_bottom = 3, tag_top = 4,
              tag_front = 5, tag_back = 6;
    MPI_Request req[12];
    int nreq = 0;

    // X axis -> пакуем в send буферы (параллелим упаковку)
    if (b.neighbors[0] != -1) { // left neighbor exists
        #pragma omp parallel for collapse(2)
        for (int idx = 0, j = 1; j <= b.Ny; ++j)
            for (int k = 1; k <= b.Nz; ++k, ++idx)
                b.left_send[idx] = u[b.local_index(1, j, k)];
        MPI_Irecv(b.left_recv.data(), b.Ny * b.Nz, MPI_DOUBLE, b.neighbors[0], tag_right,  MPI_COMM_WORLD, &req[nreq++]);
        MPI_Isend(b.left_send.data(), b.Ny * b.Nz, MPI_DOUBLE, b.neighbors[0], tag_left,   MPI_COMM_WORLD, &req[nreq++]);
    }
    if (b.neighbors[1] != -1) { // right neighbor exists
        #pragma omp parallel for collapse(2)
        for (int idx = 0, j = 1; j <= b.Ny; ++j)
            for (int k = 1; k <= b.Nz; ++k, ++idx)
                b.right_send[idx] = u[b.local_index(b.Nx, j, k)];
        MPI_Irecv(b.right_recv.data(), b.Ny * b.Nz, MPI_DOUBLE, b.neighbors[1], tag_left,   MPI_COMM_WORLD, &req[nreq++]);
        MPI_Isend(b.right_send.data(), b.Ny * b.Nz, MPI_DOUBLE, b.neighbors[1], tag_right,  MPI_COMM_WORLD, &req[nreq++]);
    }

    // Y axis -> periodic (pack)
    if (b.neighbors[2] != -1) { // bottom (y-)
        #pragma omp parallel for collapse(2)
        for (int idx = 0, i = 1; i <= b.Nx; ++i)
            for (int k = 1; k <= b.Nz; ++k, ++idx)
                b.bottom_send[idx] = u[b.local_index(i, 1, k)];
        MPI_Irecv(b.bottom_recv.data(), b.Nx * b.Nz, MPI_DOUBLE, b.neighbors[2], tag_top,    MPI_COMM_WORLD, &req[nreq++]);
        MPI_Isend(b.bottom_send.data(), b.Nx * b.Nz, MPI_DOUBLE, b.neighbors[2], tag_bottom, MPI_COMM_WORLD, &req[nreq++]);
    }
    if (b.neighbors[3] != -1) { // top (y+)
        #pragma omp parallel for collapse(2)
        for (int idx = 0, i = 1; i <= b.Nx; ++i)
            for (int k = 1; k <= b.Nz; ++k, ++idx)
                b.top_send[idx] = u[b.local_index(i, b.Ny, k)];
        MPI_Irecv(b.top_recv.data(), b.Nx * b.Nz, MPI_DOUBLE, b.neighbors[3], tag_bottom, MPI_COMM_WORLD, &req[nreq++]);
        MPI_Isend(b.top_send.data(), b.Nx * b.Nz, MPI_DOUBLE, b.neighbors[3], tag_top,    MPI_COMM_WORLD, &req[nreq++]);
    }

    // Z axis -> pack
    if (b.neighbors[4] != -1) { // front (z-)
        #pragma omp parallel for collapse(2)
        for (int idx = 0, i = 1; i <= b.Nx; ++i)
            for (int j = 1; j <= b.Ny; ++j, ++idx)
                b.front_send[idx] = u[b.local_index(i, j, 1)];
        MPI_Irecv(b.front_recv.data(), b.Nx * b.Ny, MPI_DOUBLE, b.neighbors[4], tag_back,  MPI_COMM_WORLD, &req[nreq++]);
        MPI_Isend(b.front_send.data(), b.Nx * b.Ny, MPI_DOUBLE, b.neighbors[4], tag_front, MPI_COMM_WORLD, &req[nreq++]);
    }
    if (b.neighbors[5] != -1) { // back (z+)
        #pragma omp parallel for collapse(2)
        for (int idx = 0, i = 1; i <= b.Nx; ++i)
            for (int j = 1; j <= b.Ny; ++j, ++idx)
                b.back_send[idx] = u[b.local_index(i, j, b.Nz)];
        MPI_Irecv(b.back_recv.data(), b.Nx * b.Ny, MPI_DOUBLE, b.neighbors[5], tag_front, MPI_COMM_WORLD, &req[nreq++]);
        MPI_Isend(b.back_send.data(), b.Nx * b.Ny, MPI_DOUBLE, b.neighbors[5], tag_back,  MPI_COMM_WORLD, &req[nreq++]);
    }

    MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);

    // Распаковка recv в halo (параллельно)
    if (b.neighbors[0] != -1) {
        #pragma omp parallel for collapse(2)
        for (int idx = 0, j = 1; j <= b.Ny; ++j)
            for (int k = 1; k <= b.Nz; ++k, ++idx)
                u[b.local_index(0, j, k)] = b.left_recv[idx];
    }
    if (b.neighbors[1] != -1) {
        #pragma omp parallel for collapse(2)
        for (int idx = 0, j = 1; j <= b.Ny; ++j)
            for (int k = 1; k <= b.Nz; ++k, ++idx)
                u[b.local_index(b.Nx + 1, j, k)] = b.right_recv[idx];
    }
    if (b.neighbors[2] != -1) {
        #pragma omp parallel for collapse(2)
        for (int idx = 0, i = 1; i <= b.Nx; ++i)
            for (int k = 1; k <= b.Nz; ++k, ++idx)
                u[b.local_index(i, 0, k)] = b.bottom_recv[idx];
    }
    if (b.neighbors[3] != -1) {
        #pragma omp parallel for collapse(2)
        for (int idx = 0, i = 1; i <= b.Nx; ++i)
            for (int k = 1; k <= b.Nz; ++k, ++idx)
                u[b.local_index(i, b.Ny + 1, k)] = b.top_recv[idx];
    }
    if (b.neighbors[4] != -1) {
        #pragma omp parallel for collapse(2)
        for (int idx = 0, i = 1; i <= b.Nx; ++i)
            for (int j = 1; j <= b.Ny; ++j, ++idx)
                u[b.local_index(i, j, 0)] = b.front_recv[idx];
    }
    if (b.neighbors[5] != -1) {
        #pragma omp parallel for collapse(2)
        for (int idx = 0, i = 1; i <= b.Nx; ++i)
            for (int j = 1; j <= b.Ny; ++j, ++idx)
                u[b.local_index(i, j, b.Nz + 1)] = b.back_recv[idx];
    }
}

void enforce_periodic_y(const Grid& g, Block& b, VDOUB& u) {
    bool is_y_min_block = (b.y_start == 0);
    bool is_y_max_block = (b.y_end == g.N);

    if (is_y_min_block && b.neighbors[3] != -1) {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i <= b.Nx + 1; ++i)
            for (int k = 0; k <= b.Nz + 1; ++k)
                u[b.local_index(i, 0, k)] = u[b.local_index(i, b.Ny + 1, k)];
    }

    if (is_y_max_block && b.neighbors[2] != -1) {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i <= b.Nx + 1; ++i)
            for (int k = 0; k <= b.Nz + 1; ++k)
                u[b.local_index(i, b.Ny + 1, k)] = u[b.local_index(i, 0, k)];
    }
}

void apply_boundary_conditions(const Grid& g, Block& b, VDOUB& u, double t) {
    if (b.x_start == 0) {
        #pragma omp parallel for collapse(2)
        for (int j = 0; j <= b.Ny + 1; ++j)
            for (int k = 0; k <= b.Nz + 1; ++k)
                u[b.local_index(0, j, k)] = 0.0;
    }
    if (b.x_end == g.N) {
        #pragma omp parallel for collapse(2)
        for (int j = 0; j <= b.Ny + 1; ++j)
            for (int k = 0; k <= b.Nz + 1; ++k)
                u[b.local_index(b.Nx + 1, j, k)] = 0.0;
    }

    if (b.z_start == 0) {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i <= b.Nx + 1; ++i)
            for (int j = 0; j <= b.Ny + 1; ++j)
                u[b.local_index(i, j, 0)] = 0.0;
    }
    if (b.z_end == g.N) {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i <= b.Nx + 1; ++i)
            for (int j = 0; j <= b.Ny + 1; ++j)
                u[b.local_index(i, j, b.Nz + 1)] = 0.0;
    }

    if (b.y_start == 0) {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i <= b.Nx + 1; ++i) {
            double x = (b.x_start + i - 1) * g.h_x;
            for (int k = 0; k <= b.Nz + 1; ++k) {
                double z = (b.z_start + k - 1) * g.h_z;
                u[b.local_index(i, 0, k)] = u_analytical(g, x, 0.0, z, t);
            }
        }
    }
    if (b.y_end == g.N) {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i <= b.Nx + 1; ++i) {
            double x = (b.x_start + i - 1) * g.h_x;
            for (int k = 0; k <= b.Nz + 1; ++k) {
                double z = (b.z_start + k - 1) * g.h_z;
                u[b.local_index(i, b.Ny + 1, k)] = u_analytical(g, x, g.Ly, z, t);
            }
        }
    }
}

void init(const Grid& g, Block& b, VVEC& u, double& max_inacc, double& inacc_first) {
    int total_size = b.padded_Nx * b.padded_Ny * b.padded_Nz;

    // --- u0 analytic for all points ---
    #pragma omp parallel for collapse(3)
    for (int i = 0; i <= b.Nx + 1; ++i) {
        for (int j = 0; j <= b.Ny + 1; ++j) {
            for (int k = 0; k <= b.Nz + 1; ++k) {
                double x = (b.x_start + i - 1) * g.h_x;
                double y = (b.y_start + j - 1) * g.h_y;
                double z = (b.z_start + k - 1) * g.h_z;
                u[0][b.local_index(i, j, k)] = u_analytical(g, x, y, z, 0.0);
            }
        }
    }

    // --- u1 for internal points ---
    #pragma omp parallel for collapse(3)
    for (int i = 1; i <= b.Nx; ++i) {
        for (int j = 1; j <= b.Ny; ++j) {
            for (int k = 1; k <= b.Nz; ++k) {
                u[1][b.local_index(i, j, k)] = u[0][b.local_index(i, j, k)]
                    + 0.5 * g.a2 * g.tau * g.tau * laplace_operator(g, b, u[0], i, j, k);
            }
        }
    }

    // --- boundary points from analytical ---
    #pragma omp parallel for collapse(3)
    for (int i = 0; i <= b.Nx + 1; ++i) {
        for (int j = 0; j <= b.Ny + 1; ++j) {
            for (int k = 0; k <= b.Nz + 1; ++k) {
                if (i >= 1 && i <= b.Nx && j >= 1 && j <= b.Ny && k >= 1 && k <= b.Nz)
                    continue;
                double x = (b.x_start + i - 1) * g.h_x;
                double y = (b.y_start + j - 1) * g.h_y;
                double z = (b.z_start + k - 1) * g.h_z;
                u[1][b.local_index(i, j, k)] = u_analytical(g, x, y, z, g.tau);
            }
        }
    }

    // exchange halos
    exchange_halos(b, u[0]);
    exchange_halos(b, u[1]);

    // periodic y
    enforce_periodic_y(g, b, u[0]);
    enforce_periodic_y(g, b, u[1]);

    // boundary conditions
    apply_boundary_conditions(g, b, u[0], 0.0);
    apply_boundary_conditions(g, b, u[1], g.tau);

    // compute error on u1
    double local_max_error = 0.0;
    #pragma omp parallel for collapse(3) reduction(max : local_max_error)
    for (int i = 1; i <= b.Nx; ++i) {
        for (int j = 1; j <= b.Ny; ++j) {
            for (int k = 1; k <= b.Nz; ++k) {
                double x = (b.x_start + i - 1) * g.h_x;
                double y = (b.y_start + j - 1) * g.h_y;
                double z = (b.z_start + k - 1) * g.h_z;
                double exact = u_analytical(g, x, y, z, g.tau);
                double err = fabs(u[1][b.local_index(i, j, k)] - exact);
                if (err > local_max_error) local_max_error = err;
            }
        }
    }

    double global_max_error;
    MPI_Allreduce(&local_max_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    max_inacc = max(max_inacc, global_max_error);
    inacc_first = global_max_error;

    if (b.rank == 0)
        cout << "Max start inaccuracy: " << global_max_error << endl;
}

void run_algo(const Grid& g, Block& b, VVEC& u,
              double& max_inacc, double& last_step_inaccuracy) {
    for (int step = 2; step < TIME_STEPS; ++step) {
        int prev = (step - 2) % 3;
        int curr = (step - 1) % 3;
        int next = step % 3;
        double t = step * g.tau;

        // 1. internal points (parallel)
        #pragma omp parallel for collapse(3)
        for (int i = 1; i <= b.Nx; ++i) {
            for (int j = 1; j <= b.Ny; ++j) {
                for (int k = 1; k <= b.Nz; ++k) {
                    u[next][b.local_index(i, j, k)] = 2.0 * u[curr][b.local_index(i, j, k)]
                        - u[prev][b.local_index(i, j, k)]
                        + g.a2 * g.tau * g.tau * laplace_operator(g, b, u[curr], i, j, k);
                }
            }
        }

        // 2. exchange halos for next
        exchange_halos(b, u[next]);

        // 3. boundary points from analytical
        #pragma omp parallel for collapse(3)
        for (int i = 0; i <= b.Nx + 1; ++i) {
            for (int j = 0; j <= b.Ny + 1; ++j) {
                for (int k = 0; k <= b.Nz + 1; ++k) {
                    if (i >= 1 && i <= b.Nx && j >= 1 && j <= b.Ny && k >= 1 && k <= b.Nz)
                        continue;
                    double x = (b.x_start + i - 1) * g.h_x;
                    double y = (b.y_start + j - 1) * g.h_y;
                    double z = (b.z_start + k - 1) * g.h_z;
                    u[next][b.local_index(i, j, k)] = u_analytical(g, x, y, z, t);
                }
            }
        }

        // 4. periodic y
        enforce_periodic_y(g, b, u[next]);

        // 5. boundary conditions
        apply_boundary_conditions(g, b, u[next], t);

        // 6. compute inaccuracy
        double local_max_error = 0.0;
        #pragma omp parallel for collapse(3) reduction(max : local_max_error)
        for (int i = 1; i <= b.Nx; ++i) {
            for (int j = 1; j <= b.Ny; ++j) {
                for (int k = 1; k <= b.Nz; ++k) {
                    double x = (b.x_start + i - 1) * g.h_x;
                    double y = (b.y_start + j - 1) * g.h_y;
                    double z = (b.z_start + k - 1) * g.h_z;
                    double exact = u_analytical(g, x, y, z, t);
                    double err = fabs(u[next][b.local_index(i, j, k)] - exact);
                    if (err > local_max_error) local_max_error = err;
                }
            }
        }

        double global_max_error;
        MPI_Allreduce(&local_max_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (global_max_error > max_inacc)
            max_inacc = global_max_error;

        if (step == TIME_STEPS - 1)
            last_step_inaccuracy = global_max_error;

        if (b.rank == 0)
            cout << "Max inaccuracy on step " << step << " : " << global_max_error << endl;
    }
}

void solve_mpi(const Grid& g, Block& b,
               int dimx, int dimy, int dimz,
               MPI_Comm comm_cart,
               double& time,
               double& max_inaccuracy,
               double& first_step_inaccuracy,
               double& last_step_inaccuracy,
               VDOUB& result) {
    int total_size = b.padded_Nx * b.padded_Ny * b.padded_Nz;
    VDOUB u0(total_size), u1(total_size), u2(total_size);
    VVEC u = {u0, u1, u2};

    double start_time;
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    max_inaccuracy = 0.0;
    init(g, b, u, max_inaccuracy, first_step_inaccuracy);
    run_algo(g, b, u, max_inaccuracy, last_step_inaccuracy);

    double end_time;
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();

    time = end_time - start_time;

    // Копируем результат для внутренних точек
    result.resize(b.Nx * b.Ny * b.Nz);
    #pragma omp parallel for collapse(3)
    for (int i = 1; i <= b.Nx; ++i)
        for (int j = 1; j <= b.Ny; ++j)
            for (int k = 1; k <= b.Nz; ++k) {
                int idx = (i - 1) * b.Ny * b.Nz + (j - 1) * b.Nz + (k - 1);
                result[idx] = u[(TIME_STEPS - 1) % 3][b.local_index(i, j, k)];
            }
}
