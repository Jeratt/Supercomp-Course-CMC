#define _USE_MATH_DEFINES
#include "equation.hpp"
#include <mpi.h>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;

inline double laplace_operator(const Grid& g, Block& b, const VDOUB& u, int i, int j, int k) {
    double d2x = (u[b.index(i-1, j, k)] - 2.0*u[b.index(i, j, k)] + u[b.index(i+1, j, k)]) 
                / (g.h_x * g.h_x);
    
    // Периодические условия по Y
    double d2y = (u[b.index(i, j-1, k)] - 2.0*u[b.index(i, j, k)] + u[b.index(i, j+1, k)]) 
                / (g.h_y * g.h_y);
    
    double d2z = (u[b.index(i, j, k-1)] - 2.0*u[b.index(i, j, k)] + u[b.index(i, j, k+1)]) 
                / (g.h_z * g.h_z);
    
    return d2x + d2y + d2z;
}

void exchange_halos(Block& b, VDOUB& u, MPI_Comm comm) {
    // Уникальные теги для каждого направления
    const int base_tag = 1000;
    int tags[6];
    for (int i = 0; i < 6; i++) {
        tags[i] = base_tag + b.rank * 6 + i;
    }
    
    MPI_Request req[12];
    int nreq = 0;
    
    // X direction neighbors (Dirichlet)
    if (b.neighbors[0] != MPI_PROC_NULL) { // left (x-)
        // Fill send buffer
        for (int idx = 0, j = 1; j <= b.Ny; ++j)
            for (int k = 1; k <= b.Nz; ++k, ++idx)
                b.left_send[idx] = u[b.index(1, j, k)];
                
        // Post receives first
        MPI_Irecv(b.left_recieve.data(), b.Ny*b.Nz, MPI_DOUBLE, b.neighbors[0], tags[1], comm, &req[nreq++]);
        // Then sends
        MPI_Isend(b.left_send.data(), b.Ny*b.Nz, MPI_DOUBLE, b.neighbors[0], tags[0], comm, &req[nreq++]);
    }
    
    if (b.neighbors[1] != MPI_PROC_NULL) { // right (x+)
        for (int idx = 0, j = 1; j <= b.Ny; ++j)
            for (int k = 1; k <= b.Nz; ++k, ++idx)
                b.right_send[idx] = u[b.index(b.Nx, j, k)];
                
        MPI_Irecv(b.right_recieve.data(), b.Ny*b.Nz, MPI_DOUBLE, b.neighbors[1], tags[0], comm, &req[nreq++]);
        MPI_Isend(b.right_send.data(), b.Ny*b.Nz, MPI_DOUBLE, b.neighbors[1], tags[1], comm, &req[nreq++]);
    }
    
    // Y direction neighbors (Periodic)
    if (b.neighbors[2] != MPI_PROC_NULL) { // bottom (y-)
        for (int idx = 0, i = 1; i <= b.Nx; ++i)
            for (int k = 1; k <= b.Nz; ++k, ++idx)
                b.bottom_send[idx] = u[b.index(i, 1, k)];
                
        MPI_Irecv(b.bottom_recieve.data(), b.Nx*b.Nz, MPI_DOUBLE, b.neighbors[2], tags[3], comm, &req[nreq++]);
        MPI_Isend(b.bottom_send.data(), b.Nx*b.Nz, MPI_DOUBLE, b.neighbors[2], tags[2], comm, &req[nreq++]);
    }
    
    if (b.neighbors[3] != MPI_PROC_NULL) { // top (y+)
        for (int idx = 0, i = 1; i <= b.Nx; ++i)
            for (int k = 1; k <= b.Nz; ++k, ++idx)
                b.top_send[idx] = u[b.index(i, b.Ny, k)];
                
        MPI_Irecv(b.top_recieve.data(), b.Nx*b.Nz, MPI_DOUBLE, b.neighbors[3], tags[2], comm, &req[nreq++]);
        MPI_Isend(b.top_send.data(), b.Nx*b.Nz, MPI_DOUBLE, b.neighbors[3], tags[3], comm, &req[nreq++]);
    }
    
    // Z direction neighbors (Dirichlet)
    if (b.neighbors[4] != MPI_PROC_NULL) { // front (z-)
        for (int idx = 0, i = 1; i <= b.Nx; ++i)
            for (int j = 1; j <= b.Ny; ++j, ++idx)
                b.front_send[idx] = u[b.index(i, j, 1)];
                
        MPI_Irecv(b.front_recieve.data(), b.Nx*b.Ny, MPI_DOUBLE, b.neighbors[4], tags[5], comm, &req[nreq++]);
        MPI_Isend(b.front_send.data(), b.Nx*b.Ny, MPI_DOUBLE, b.neighbors[4], tags[4], comm, &req[nreq++]);
    }
    
    if (b.neighbors[5] != MPI_PROC_NULL) { // back (z+)
        for (int idx = 0, i = 1; i <= b.Nx; ++i)
            for (int j = 1; j <= b.Ny; ++j, ++idx)
                b.back_send[idx] = u[b.index(i, j, b.Nz)];
                
        MPI_Irecv(b.back_recieve.data(), b.Nx*b.Ny, MPI_DOUBLE, b.neighbors[5], tags[4], comm, &req[nreq++]);
        MPI_Isend(b.back_send.data(), b.Nx*b.Ny, MPI_DOUBLE, b.neighbors[5], tags[5], comm, &req[nreq++]);
    }
    
    // Wait for all communications to complete
    if (nreq > 0)
        MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);
    
    // Update halo cells
    if (b.neighbors[0] != MPI_PROC_NULL) {
        for (int idx = 0, j = 1; j <= b.Ny; ++j)
            for (int k = 1; k <= b.Nz; ++k, ++idx)
                u[b.index(0, j, k)] = b.left_recieve[idx];
    }
    
    if (b.neighbors[1] != MPI_PROC_NULL) {
        for (int idx = 0, j = 1; j <= b.Ny; ++j)
            for (int k = 1; k <= b.Nz; ++k, ++idx)
                u[b.index(b.Nx+1, j, k)] = b.right_recieve[idx];
    }
    
    if (b.neighbors[2] != MPI_PROC_NULL) {
        for (int idx = 0, i = 1; i <= b.Nx; ++i)
            for (int k = 1; k <= b.Nz; ++k, ++idx)
                u[b.index(i, 0, k)] = b.bottom_recieve[idx];
    }
    
    if (b.neighbors[3] != MPI_PROC_NULL) {
        for (int idx = 0, i = 1; i <= b.Nx; ++i)
            for (int k = 1; k <= b.Nz; ++k, ++idx)
                u[b.index(i, b.Ny+1, k)] = b.top_recieve[idx];
    }
    
    if (b.neighbors[4] != MPI_PROC_NULL) {
        for (int idx = 0, i = 1; i <= b.Nx; ++i)
            for (int j = 1; j <= b.Ny; ++j, ++idx)
                u[b.index(i, j, 0)] = b.front_recieve[idx];
    }
    
    if (b.neighbors[5] != MPI_PROC_NULL) {
        for (int idx = 0, i = 1; i <= b.Nx; ++i)
            for (int j = 1; j <= b.Ny; ++j, ++idx)
                u[b.index(i, j, b.Nz+1)] = b.back_recieve[idx];
    }
    
    // SPECIAL HANDLING FOR PERIODIC BOUNDARIES in Y direction
    if (b.neighbors[2] == MPI_PROC_NULL && b.neighbors[3] == MPI_PROC_NULL) {
        // Single process in Y dimension - just copy values to maintain periodicity
        for (int i = 0; i <= b.Nx+1; ++i) {
            for (int k = 0; k <= b.Nz+1; ++k) {
                u[b.index(i, 0, k)] = u[b.index(i, b.Ny, k)];
                u[b.index(i, b.Ny+1, k)] = u[b.index(i, 1, k)];
            }
        }
    }
}

void apply_boundary_conditions(const Grid& g, Block& b, VDOUB& u) {
    // X boundaries - Dirichlet (u=0)
    if (b.x_start == 0) {
        for (int j = 1; j <= b.Ny; ++j)
            for (int k = 1; k <= b.Nz; ++k)
                u[b.index(0, j, k)] = 0.0;
    }
    
    if (b.x_end == g.N) {
        for (int j = 1; j <= b.Ny; ++j)
            for (int k = 1; k <= b.Nz; ++k)
                u[b.index(b.Nx+1, j, k)] = 0.0;
    }
    
    // Z boundaries - Dirichlet (u=0)
    if (b.z_start == 0) {
        for (int i = 1; i <= b.Nx; ++i)
            for (int j = 1; j <= b.Ny; ++j)
                u[b.index(i, j, 0)] = 0.0;
    }
    
    if (b.z_end == g.N) {
        for (int i = 1; i <= b.Nx; ++i)
            for (int j = 1; j <= b.Ny; ++j)
                u[b.index(i, j, b.Nz+1)] = 0.0;
    }
}

void init(const Grid& g, Block& b, VVEC& u, MPI_Comm comm, 
          double& max_inaccuracy, double& first_step_inaccuracy) {
    // Initialize u^0 with analytical solution for all points
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
                
                u[0][b.index(i, j, k)] = u_analytical(g, x, y, z, 0.0);
            }
        }
    }
    
    // Exchange halos for u^0
    exchange_halos(b, u[0], comm);
    
    // Apply boundary conditions for u^0
    apply_boundary_conditions(g, b, u[0]);
    
    // Initialize u^1 for internal points
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
                    u[1][b.index(i, j, k)] = u[0][b.index(i, j, k)]
                        + 0.5 * g.a2 * g.tau * g.tau * laplace_operator(g, b, u[0], i, j, k);
                } else {
                    // Boundary points - use analytical solution for accuracy
                    u[1][b.index(i, j, k)] = u_analytical(g, x, y, z, g.tau);
                }
            }
        }
    }
    
    // Exchange halos for u^1
    exchange_halos(b, u[1], comm);
    
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
                double err = fabs(u[1][b.index(i, j, k)] - exact);
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

void run_algo(const Grid& g, Block& b, VVEC& u, MPI_Comm comm,
              double& max_inaccuracy, double& last_step_inaccuracy) {
    for (int step = 2; step < TIME_STEPS; ++step) {
        int prev = (step-2) % 3;
        int curr = (step-1) % 3;
        int next = step % 3;
        double t = step * g.tau;
        
        // Compute values for next step
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
                        u[next][b.index(i, j, k)] = 2.0 * u[curr][b.index(i, j, k)]
                            - u[prev][b.index(i, j, k)]
                            + g.a2 * g.tau * g.tau * laplace_operator(g, b, u[curr], i, j, k);
                    } else {
                        // Boundary points: analytical solution
                        u[next][b.index(i, j, k)] = u_analytical(g, x, y, z, t);
                    }
                }
            }
        }
        
        // Exchange halos
        exchange_halos(b, u[next], comm);
        
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
                    double err = fabs(u[next][b.index(i, j, k)] - exact);
                    if (err > local_max_err) local_max_err = err;
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
    
    double start = MPI_Wtime();
    
    // Initialization and first time step
    init(g, b, u, comm_cart, max_inaccuracy, first_step_inaccuracy);
    
    // Main time loop
    run_algo(g, b, u, comm_cart, max_inaccuracy, last_step_inaccuracy);
    
    double end = MPI_Wtime();
    time = end - start;
    
    // Collect result
    result.resize(b.Nx * b.Ny * b.Nz);
    int next = (TIME_STEPS-1) % 3;
    
    for (int i = 1; i <= b.Nx; ++i)
        for (int j = 1; j <= b.Ny; ++j)
            for (int k = 1; k <= b.Nz; ++k) {
                int idx = (i-1)*b.Ny*b.Nz + (j-1)*b.Nz + (k-1);
                result[idx] = u[next][b.index(i, j, k)];
            }
}