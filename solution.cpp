#define _USE_MATH_DEFINES
#include "equation.hpp"
#include <mpi.h>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cstring>
using namespace std;

inline double laplace_operator(const Grid& g, Block& b, const VDOUB& u, int i, int j, int k) {
    double d2x = (u[b.local_index(i-1, j, k)] - 2.0*u[b.local_index(i, j, k)] + u[b.local_index(i+1, j, k)]) 
                / (g.h_x * g.h_x);
    
    double d2y = (u[b.local_index(i, j-1, k)] - 2.0*u[b.local_index(i, j, k)] + u[b.local_index(i, j+1, k)]) 
                / (g.h_y * g.h_y);
    
    double d2z = (u[b.local_index(i, j, k-1)] - 2.0*u[b.local_index(i, j, k)] + u[b.local_index(i, j, k+1)]) 
                / (g.h_z * g.h_z);
    
    return d2x + d2y + d2z;
}

// Безопасный обмен гало-ячейками с уникальными тегами для КАЖДОГО направления
void exchange_halos_safe(Block& b, VDOUB& u) {
    // Уникальные теги для КАЖДОГО направления и КАЖДОГО процесса
    const int base_tag = 1000;
    const int tags[6] = {
        base_tag + b.rank * 6,     // left (x-)
        base_tag + b.rank * 6 + 1, // right (x+)
        base_tag + b.rank * 6 + 2, // bottom (y-)
        base_tag + b.rank * 6 + 3, // top (y+)
        base_tag + b.rank * 6 + 4, // front (z-)
        base_tag + b.rank * 6 + 5  // back (z+)
    };
    
    MPI_Request req[12];
    int nreq = 0;
    
    // X direction - Dirichlet (1st order)
    if (b.neighbors[0] != MPI_PROC_NULL) {
        for (int idx = 0, j = 1; j <= b.Ny; ++j)
            for (int k = 1; k <= b.Nz; ++k, ++idx)
                b.left_send[idx] = u[b.local_index(1, j, k)];
                
        MPI_Irecv(b.left_recv.data(), b.Ny*b.Nz, MPI_DOUBLE, b.neighbors[0], tags[1], MPI_COMM_WORLD, &req[nreq++]);
        MPI_Isend(b.left_send.data(), b.Ny*b.Nz, MPI_DOUBLE, b.neighbors[0], tags[0], MPI_COMM_WORLD, &req[nreq++]);
    }
    
    if (b.neighbors[1] != MPI_PROC_NULL) {
        for (int idx = 0, j = 1; j <= b.Ny; ++j)
            for (int k = 1; k <= b.Nz; ++k, ++idx)
                b.right_send[idx] = u[b.local_index(b.Nx, j, k)];
                
        MPI_Irecv(b.right_recv.data(), b.Ny*b.Nz, MPI_DOUBLE, b.neighbors[1], tags[0], MPI_COMM_WORLD, &req[nreq++]);
        MPI_Isend(b.right_send.data(), b.Ny*b.Nz, MPI_DOUBLE, b.neighbors[1], tags[1], MPI_COMM_WORLD, &req[nreq++]);
    }
    
    // Y direction - Periodic (НОВЫЙ БЕЗОПАСНЫЙ ПОДХОД)
    if (b.neighbors[2] != MPI_PROC_NULL) {
        for (int idx = 0, i = 1; i <= b.Nx; ++i)
            for (int k = 1; k <= b.Nz; ++k, ++idx)
                b.bottom_send[idx] = u[b.local_index(i, 1, k)];
                
        MPI_Irecv(b.bottom_recv.data(), b.Nx*b.Nz, MPI_DOUBLE, b.neighbors[2], tags[3], MPI_COMM_WORLD, &req[nreq++]);
        MPI_Isend(b.bottom_send.data(), b.Nx*b.Nz, MPI_DOUBLE, b.neighbors[2], tags[2], MPI_COMM_WORLD, &req[nreq++]);
    }
    
    if (b.neighbors[3] != MPI_PROC_NULL) {
        for (int idx = 0, i = 1; i <= b.Nx; ++i)
            for (int k = 1; k <= b.Nz; ++k, ++idx)
                b.top_send[idx] = u[b.local_index(i, b.Ny, k)];
                
        MPI_Irecv(b.top_recv.data(), b.Nx*b.Nz, MPI_DOUBLE, b.neighbors[3], tags[2], MPI_COMM_WORLD, &req[nreq++]);
        MPI_Isend(b.top_send.data(), b.Nx*b.Nz, MPI_DOUBLE, b.neighbors[3], tags[3], MPI_COMM_WORLD, &req[nreq++]);
    }
    
    // Z direction - Dirichlet (1st order)
    if (b.neighbors[4] != MPI_PROC_NULL) {
        for (int idx = 0, i = 1; i <= b.Nx; ++i)
            for (int j = 1; j <= b.Ny; ++j, ++idx)
                b.front_send[idx] = u[b.local_index(i, j, 1)];
                
        MPI_Irecv(b.front_recv.data(), b.Nx*b.Ny, MPI_DOUBLE, b.neighbors[4], tags[5], MPI_COMM_WORLD, &req[nreq++]);
        MPI_Isend(b.front_send.data(), b.Nx*b.Ny, MPI_DOUBLE, b.neighbors[4], tags[4], MPI_COMM_WORLD, &req[nreq++]);
    }
    
    if (b.neighbors[5] != MPI_PROC_NULL) {
        for (int idx = 0, i = 1; i <= b.Nx; ++i)
            for (int j = 1; j <= b.Ny; ++j, ++idx)
                b.back_send[idx] = u[b.local_index(i, j, b.Nz)];
                
        MPI_Irecv(b.back_recv.data(), b.Nx*b.Ny, MPI_DOUBLE, b.neighbors[5], tags[4], MPI_COMM_WORLD, &req[nreq++]);
        MPI_Isend(b.back_send.data(), b.Nx*b.Ny, MPI_DOUBLE, b.neighbors[5], tags[5], MPI_COMM_WORLD, &req[nreq++]);
    }
    
    if (nreq > 0)
        MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);
    
    // Update halo cells
    if (b.neighbors[0] != MPI_PROC_NULL) {
        for (int idx = 0, j = 1; j <= b.Ny; ++j)
            for (int k = 1; k <= b.Nz; ++k, ++idx)
                u[b.local_index(0, j, k)] = b.left_recv[idx];
    }
    
    if (b.neighbors[1] != MPI_PROC_NULL) {
        for (int idx = 0, j = 1; j <= b.Ny; ++j)
            for (int k = 1; k <= b.Nz; ++k, ++idx)
                u[b.local_index(b.Nx+1, j, k)] = b.right_recv[idx];
    }
    
    if (b.neighbors[2] != MPI_PROC_NULL) {
        for (int idx = 0, i = 1; i <= b.Nx; ++i)
            for (int k = 1; k <= b.Nz; ++k, ++idx)
                u[b.local_index(i, 0, k)] = b.bottom_recv[idx];
    }
    
    if (b.neighbors[3] != MPI_PROC_NULL) {
        for (int idx = 0, i = 1; i <= b.Nx; ++i)
            for (int k = 1; k <= b.Nz; ++k, ++idx)
                u[b.local_index(i, b.Ny+1, k)] = b.top_recv[idx];
    }
    
    if (b.neighbors[4] != MPI_PROC_NULL) {
        for (int idx = 0, i = 1; i <= b.Nx; ++i)
            for (int j = 1; j <= b.Ny; ++j, ++idx)
                u[b.local_index(i, j, 0)] = b.front_recv[idx];
    }
    
    if (b.neighbors[5] != MPI_PROC_NULL) {
        for (int idx = 0, i = 1; i <= b.Nx; ++i)
            for (int j = 1; j <= b.Ny; ++j, ++idx)
                u[b.local_index(i, j, b.Nz+1)] = b.back_recv[idx];
    }
}

void apply_boundary_conditions(const Grid& g, Block& b, VDOUB& u) {
    // X boundaries - Dirichlet (u=0)
    if (b.x_start == 0) {
        for (int j = 0; j < b.padded_Ny; ++j)
            for (int k = 0; k < b.padded_Nz; ++k)
                u[b.local_index(0, j, k)] = 0.0;
    }
    
    if (b.x_end == g.N) {
        for (int j = 0; j < b.padded_Ny; ++j)
            for (int k = 0; k < b.padded_Nz; ++k)
                u[b.local_index(b.Nx+1, j, k)] = 0.0;
    }
    
    // Z boundaries - Dirichlet (u=0)
    if (b.z_start == 0) {
        for (int i = 0; i < b.padded_Nx; ++i)
            for (int j = 0; j < b.padded_Ny; ++j)
                u[b.local_index(i, j, 0)] = 0.0;
    }
    
    if (b.z_end == g.N) {
        for (int i = 0; i < b.padded_Nx; ++i)
            for (int j = 0; j < b.padded_Ny; ++j)
                u[b.local_index(i, j, b.Nz+1)] = 0.0;
    }
}

void init(const Grid& g, Block& b, VVEC& u, double& max_inaccuracy, double& first_step_inaccuracy) {
    // Initialize u^0 with analytical solution for ALL points including boundaries
    for (int i = 0; i <= b.Nx+1; ++i) {
        int i_global = b.x_start + i - 1;
        if (i_global < 0 || i_global > g.N) continue;
        double x = i_global * g.h_x;
        
        for (int j = 0; j <= b.Ny+1; ++j) {
            int j_global = b.y_start + j - 1;
            if (j_global < 0 || j_global > g.N) continue;
            double y = j_global * g.h_y;
            
            for (int k = 0; k <= b.Nz+1; ++k) {
                int k_global = b.z_start + k - 1;
                if (k_global < 0 || k_global > g.N) continue;
                double z = k_global * g.h_z;
                
                u[0][b.local_index(i, j, k)] = u_analytical(g, x, y, z, 0.0);
            }
        }
    }
    
    // Exchange halos for u^0
    exchange_halos_safe(b, u[0]);
    
    // Apply boundary conditions for u^0
    apply_boundary_conditions(g, b, u[0]);
    
    // Initialize u^1
    for (int i = 1; i <= b.Nx; ++i) {
        int i_global = b.x_start + i - 1;
        double x = i_global * g.h_x;
        
        for (int j = 1; j <= b.Ny; ++j) {
            int j_global = b.y_start + j - 1;
            double y = j_global * g.h_y;
            
            for (int k = 1; k <= b.Nz; ++k) {
                int k_global = b.z_start + k - 1;
                double z = k_global * g.h_z;
                
                if (i_global > 0 && i_global < g.N && 
                    j_global > 0 && j_global < g.N && 
                    k_global > 0 && k_global < g.N) {
                    // Internal points: numerical scheme
                    u[1][b.local_index(i, j, k)] = u[0][b.local_index(i, j, k)]
                        + 0.5 * g.a2 * g.tau * g.tau * laplace_operator(g, b, u[0], i, j, k);
                } else {
                    // Boundary points: analytical solution
                    u[1][b.local_index(i, j, k)] = u_analytical(g, x, y, z, g.tau);
                }
            }
        }
    }
    
    // Exchange halos for u^1
    exchange_halos_safe(b, u[1]);
    
    // Apply boundary conditions for u^1
    apply_boundary_conditions(g, b, u[1]);
    
    // Compute error for u^1
    double local_max_err = 0.0;
    double t = g.tau;
    
    for (int i = 1; i <= b.Nx; ++i) {
        int i_global = b.x_start + i - 1;
        double x = i_global * g.h_x;
        
        for (int j = 1; j <= b.Ny; ++j) {
            int j_global = b.y_start + j - 1;
            double y = j_global * g.h_y;
            
            for (int k = 1; k <= b.Nz; ++k) {
                int k_global = b.z_start + k - 1;
                double z = k_global * g.h_z;
                
                double exact = u_analytical(g, x, y, z, t);
                double err = fabs(u[1][b.local_index(i, j, k)] - exact);
                local_max_err = max(local_max_err, err);
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

void run_algo(const Grid& g, Block& b, VVEC& u, double& max_inaccuracy, double& last_step_inaccuracy) {
    for (int step = 2; step < TIME_STEPS; ++step) {
        int prev = (step-2) % 3;
        int curr = (step-1) % 3;
        int next = step % 3;
        double t = step * g.tau;
        
        // Compute internal points
        for (int i = 1; i <= b.Nx; ++i) {
            int i_global = b.x_start + i - 1;
            double x = i_global * g.h_x;
            
            for (int j = 1; j <= b.Ny; ++j) {
                int j_global = b.y_start + j - 1;
                double y = j_global * g.h_y;
                
                for (int k = 1; k <= b.Nz; ++k) {
                    int k_global = b.z_start + k - 1;
                    double z = k_global * g.h_z;
                    
                    if (i_global > 0 && i_global < g.N && 
                        j_global > 0 && j_global < g.N && 
                        k_global > 0 && k_global < g.N) {
                        // Internal points: explicit scheme
                        u[next][b.local_index(i, j, k)] = 2.0 * u[curr][b.local_index(i, j, k)]
                            - u[prev][b.local_index(i, j, k)]
                            + g.a2 * g.tau * g.tau * laplace_operator(g, b, u[curr], i, j, k);
                    } else {
                        // Boundary points: analytical solution
                        u[next][b.local_index(i, j, k)] = u_analytical(g, x, y, z, t);
                    }
                }
            }
        }
        
        // Exchange halos - ИСПОЛЬЗУЕМ БЕЗОПАСНУЮ ВЕРСИЮ
        exchange_halos_safe(b, u[next]);
        
        // Apply boundary conditions
        apply_boundary_conditions(g, b, u[next]);
        
        // Compute error
        double local_max_err = 0.0;
        
        for (int i = 1; i <= b.Nx; ++i) {
            int i_global = b.x_start + i - 1;
            double x = i_global * g.h_x;
            
            for (int j = 1; j <= b.Ny; ++j) {
                int j_global = b.y_start + j - 1;
                double y = j_global * g.h_y;
                
                for (int k = 1; k <= b.Nz; ++k) {
                    int k_global = b.z_start + k - 1;
                    double z = k_global * g.h_z;
                    
                    double exact = u_analytical(g, x, y, z, t);
                    double err = fabs(u[next][b.local_index(i, j, k)] - exact);
                    local_max_err = max(local_max_err, err);
                }
            }
        }
        
        double global_max_err;
        MPI_Allreduce(&local_max_err, &global_max_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        
        if (global_max_err > max_inaccuracy)
            max_inaccuracy = global_max_err;
            
        if (step == TIME_STEPS-1)
            last_step_inaccuracy = global_max_err;
            
        if (b.rank == 0)
            cout << "Step " << step << ": max inaccuracy = " << global_max_err << endl;
    }
}

void solve_mpi(const Grid& g, Block& b, int dimx, int dimy, int dimz, MPI_Comm comm_cart,
              double& time, double& max_inaccuracy, double& first_step_inaccuracy,
              double& last_step_inaccuracy, VDOUB& result) {
    int total_size = b.padded_Nx * b.padded_Ny * b.padded_Nz;
    VDOUB u0(total_size), u1(total_size), u2(total_size);
    VVEC u = {u0, u1, u2};
    
    double start = MPI_Wtime();
    
    // Initialize and first step
    init(g, b, u, max_inaccuracy, first_step_inaccuracy);
    
    // Main time loop
    run_algo(g, b, u, max_inaccuracy, last_step_inaccuracy);
    
    double end = MPI_Wtime();
    time = end - start;
    
    // Collect result from last time step
    result.resize(b.Nx * b.Ny * b.Nz);
    int next = (TIME_STEPS-1) % 3;
    
    for (int i = 1; i <= b.Nx; ++i)
        for (int j = 1; j <= b.Ny; ++j)
            for (int k = 1; k <= b.Nz; ++k) {
                int idx = (i-1)*b.Ny*b.Nz + (j-1)*b.Nz + (k-1);
                result[idx] = u[next][b.local_index(i, j, k)];
            }
}