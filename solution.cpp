#define _USE_MATH_DEFINES
#include "equation.hpp"
#include <algorithm>
#include <iostream>
#include <cmath>
#include <mpi.h>

// Exchange ghost layers between neighboring processes
void exchange_ghost_layers(Block& b, VDOUB& ui_local, MPI_Comm& comm_cart) {
    MPI_Request reqs[12];
    int req_count = 0;

    // X direction (Dirichlet boundaries - no exchange at physical boundaries)
    if (b.neighbors[0] != MPI_PROC_NULL) {  // left neighbor exists
        // Fill send buffer: leftmost interior layer (send to left neighbor)
        for (int idx = 0, j = 1; j <= b.Ny; ++j)
            for (int k = 1; k <= b.Nz; ++k, ++idx)
                b.left_send[idx] = ui_local[b.index(1, j, k)];
        
        MPI_Irecv(b.left_recieve.data(), b.Ny * b.Nz, MPI_DOUBLE, b.neighbors[0], 
                  b.rank * 6 + 0, comm_cart, &reqs[req_count++]);
        MPI_Isend(b.left_send.data(), b.Ny * b.Nz, MPI_DOUBLE, b.neighbors[0], 
                  b.neighbors[0] * 6 + 1, comm_cart, &reqs[req_count++]);
    }
    
    if (b.neighbors[1] != MPI_PROC_NULL) {  // right neighbor exists
        // Fill send buffer: rightmost interior layer (send to right neighbor)
        for (int idx = 0, j = 1; j <= b.Ny; ++j)
            for (int k = 1; k <= b.Nz; ++k, ++idx)
                b.right_send[idx] = ui_local[b.index(b.Nx, j, k)];
        
        MPI_Irecv(b.right_recieve.data(), b.Ny * b.Nz, MPI_DOUBLE, b.neighbors[1], 
                  b.rank * 6 + 1, comm_cart, &reqs[req_count++]);
        MPI_Isend(b.right_send.data(), b.Ny * b.Nz, MPI_DOUBLE, b.neighbors[1], 
                  b.neighbors[1] * 6 + 0, comm_cart, &reqs[req_count++]);
    }

    // Y direction (Periodic boundaries)
    if (b.neighbors[2] != MPI_PROC_NULL) {  // bottom neighbor (y-)
        for (int idx = 0, i = 1; i <= b.Nx; ++i)
            for (int k = 1; k <= b.Nz; ++k, ++idx)
                b.bottom_send[idx] = ui_local[b.index(i, 1, k)];
        
        MPI_Irecv(b.bottom_recieve.data(), b.Nx * b.Nz, MPI_DOUBLE, b.neighbors[2], 
                  b.rank * 6 + 2, comm_cart, &reqs[req_count++]);
        MPI_Isend(b.bottom_send.data(), b.Nx * b.Nz, MPI_DOUBLE, b.neighbors[2], 
                  b.neighbors[2] * 6 + 3, comm_cart, &reqs[req_count++]);
    }
    
    if (b.neighbors[3] != MPI_PROC_NULL) {  // top neighbor (y+)
        for (int idx = 0, i = 1; i <= b.Nx; ++i)
            for (int k = 1; k <= b.Nz; ++k, ++idx)
                b.top_send[idx] = ui_local[b.index(i, b.Ny, k)];
        
        MPI_Irecv(b.top_recieve.data(), b.Nx * b.Nz, MPI_DOUBLE, b.neighbors[3], 
                  b.rank * 6 + 3, comm_cart, &reqs[req_count++]);
        MPI_Isend(b.top_send.data(), b.Nx * b.Nz, MPI_DOUBLE, b.neighbors[3], 
                  b.neighbors[3] * 6 + 2, comm_cart, &reqs[req_count++]);
    }

    // Z direction (Dirichlet boundaries - no exchange at physical boundaries)
    if (b.neighbors[4] != MPI_PROC_NULL) {  // front neighbor (z-)
        for (int idx = 0, i = 1; i <= b.Nx; ++i)
            for (int j = 1; j <= b.Ny; ++j, ++idx)
                b.front_send[idx] = ui_local[b.index(i, j, 1)];
        
        MPI_Irecv(b.front_recieve.data(), b.Nx * b.Ny, MPI_DOUBLE, b.neighbors[4], 
                  b.rank * 6 + 4, comm_cart, &reqs[req_count++]);
        MPI_Isend(b.front_send.data(), b.Nx * b.Ny, MPI_DOUBLE, b.neighbors[4], 
                  b.neighbors[4] * 6 + 5, comm_cart, &reqs[req_count++]);
    }
    
    if (b.neighbors[5] != MPI_PROC_NULL) {  // back neighbor (z+)
        for (int idx = 0, i = 1; i <= b.Nx; ++i)
            for (int j = 1; j <= b.Ny; ++j, ++idx)
                b.back_send[idx] = ui_local[b.index(i, j, b.Nz)];
        
        MPI_Irecv(b.back_recieve.data(), b.Nx * b.Ny, MPI_DOUBLE, b.neighbors[5], 
                  b.rank * 6 + 5, comm_cart, &reqs[req_count++]);
        MPI_Isend(b.back_send.data(), b.Nx * b.Ny, MPI_DOUBLE, b.neighbors[5], 
                  b.neighbors[5] * 6 + 4, comm_cart, &reqs[req_count++]);
    }

    MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);

    // Copy received data into ghost layers
    if (b.neighbors[0] != MPI_PROC_NULL) {
        for (int idx = 0, j = 1; j <= b.Ny; ++j)
            for (int k = 1; k <= b.Nz; ++k, ++idx)
                ui_local[b.index(0, j, k)] = b.left_recieve[idx];
    }
    
    if (b.neighbors[1] != MPI_PROC_NULL) {
        for (int idx = 0, j = 1; j <= b.Ny; ++j)
            for (int k = 1; k <= b.Nz; ++k, ++idx)
                ui_local[b.index(b.Nx + 1, j, k)] = b.right_recieve[idx];
    }
    
    if (b.neighbors[2] != MPI_PROC_NULL) {
        for (int idx = 0, i = 1; i <= b.Nx; ++i)
            for (int k = 1; k <= b.Nz; ++k, ++idx)
                ui_local[b.index(i, 0, k)] = b.bottom_recieve[idx];
    }
    
    if (b.neighbors[3] != MPI_PROC_NULL) {
        for (int idx = 0, i = 1; i <= b.Nx; ++i)
            for (int k = 1; k <= b.Nz; ++k, ++idx)
                ui_local[b.index(i, b.Ny + 1, k)] = b.top_recieve[idx];
    }
    
    if (b.neighbors[4] != MPI_PROC_NULL) {
        for (int idx = 0, i = 1; i <= b.Nx; ++i)
            for (int j = 1; j <= b.Ny; ++j, ++idx)
                ui_local[b.index(i, j, 0)] = b.front_recieve[idx];
    }
    
    if (b.neighbors[5] != MPI_PROC_NULL) {
        for (int idx = 0, i = 1; i <= b.Nx; ++i)
            for (int j = 1; j <= b.Ny; ++j, ++idx)
                ui_local[b.index(i, j, b.Nz + 1)] = b.back_recieve[idx];
    }
}

// Laplace operator for 7-point stencil
inline double laplace_operator(const Grid& g, const Block& b, const VDOUB& ui_local, 
                               const int& i, const int& j, const int& k) {
    double d2x = (ui_local[b.index(i - 1, j, k)] - 2.0 * ui_local[b.index(i, j, k)] + 
                  ui_local[b.index(i + 1, j, k)]) / (g.h_x * g.h_x);
    
    // Y direction: periodic (handled via ghost layers)
    double d2y = (ui_local[b.index(i, j - 1, k)] - 2.0 * ui_local[b.index(i, j, k)] + 
                  ui_local[b.index(i, j + 1, k)]) / (g.h_y * g.h_y);
    
    double d2z = (ui_local[b.index(i, j, k - 1)] - 2.0 * ui_local[b.index(i, j, k)] + 
                  ui_local[b.index(i, j, k + 1)]) / (g.h_z * g.h_z);
    
    return d2x + d2y + d2z;
}

// Apply Dirichlet boundary conditions (u=0) at physical boundaries
void apply_boundary_conditions(const Grid& g, Block& b, VDOUB& u) {
    // X boundaries: Dirichlet (u=0) at x=0 and x=Lx
    if (b.x_start == 0) {
        for (int j = 0; j <= b.Ny + 1; ++j)
            for (int k = 0; k <= b.Nz + 1; ++k)
                u[b.index(0, j, k)] = 0.0;
    }
    
    if (b.x_end == g.N) {
        for (int j = 0; j <= b.Ny + 1; ++j)
            for (int k = 0; k <= b.Nz + 1; ++k)
                u[b.index(b.Nx + 1, j, k)] = 0.0;
    }
    
    // Z boundaries: Dirichlet (u=0) at z=0 and z=Lz
    if (b.z_start == 0) {
        for (int i = 0; i <= b.Nx + 1; ++i)
            for (int j = 0; j <= b.Ny + 1; ++j)
                u[b.index(i, j, 0)] = 0.0;
    }
    
    if (b.z_end == g.N) {
        for (int i = 0; i <= b.Nx + 1; ++i)
            for (int j = 0; j <= b.Ny + 1; ++j)
                u[b.index(i, j, b.Nz + 1)] = 0.0;
    }
    
    // Y boundaries: periodic (handled by MPI_Cart_shift, but ensure consistency)
    // If single process in Y dimension, copy values
    if (b.neighbors[2] == MPI_PROC_NULL && b.neighbors[3] == MPI_PROC_NULL) {
        for (int i = 0; i <= b.Nx + 1; ++i)
            for (int k = 0; k <= b.Nz + 1; ++k) {
                u[b.index(i, 0, k)] = u[b.index(i, b.Ny, k)];
                u[b.index(i, b.Ny + 1, k)] = u[b.index(i, 1, k)];
            }
    }
}

void init(const Grid& g, Block& b, VVEC& u_local, const int& dim0_n, const int& dim1_n, 
          const int& dim2_n, MPI_Comm& comm_cart, double& max_inaccuracy, 
          double& first_step_inaccuracy) {
    // Initialize u^0: fill all interior points with analytical solution
    for (int i = 1; i <= b.Nx; ++i) {
        int i_global = b.x_start + i - 1;
        double x = i_global * g.h_x;
        
        for (int j = 1; j <= b.Ny; ++j) {
            int j_global = b.y_start + j - 1;
            double y = j_global * g.h_y;
            
            for (int k = 1; k <= b.Nz; ++k) {
                int k_global = b.z_start + k - 1;
                double z = k_global * g.h_z;
                
                u_local[0][b.index(i, j, k)] = u_analytical(g, x, y, z, 0.0);
            }
        }
    }
    
    // Exchange ghost layers for u^0
    exchange_ghost_layers(b, u_local[0], comm_cart);
    
    // Apply boundary conditions
    apply_boundary_conditions(g, b, u_local[0]);
    
    // Initialize u^1: use numerical scheme for interior points
    for (int i = 1; i <= b.Nx; ++i) {
        int i_global = b.x_start + i - 1;
        double x = i_global * g.h_x;
        
        for (int j = 1; j <= b.Ny; ++j) {
            int j_global = b.y_start + j - 1;
            double y = j_global * g.h_y;
            
            for (int k = 1; k <= b.Nz; ++k) {
                int k_global = b.z_start + k - 1;
                double z = k_global * g.h_z;
                
                // Check if point is on physical boundary (Dirichlet)
                bool on_x_boundary = (i_global == 0 || i_global == g.N);
                bool on_z_boundary = (k_global == 0 || k_global == g.N);
                
                if (on_x_boundary || on_z_boundary) {
                    // Use analytical solution at boundaries
                    u_local[1][b.index(i, j, k)] = u_analytical(g, x, y, z, g.tau);
                } else {
                    // Interior point: use numerical scheme
                    u_local[1][b.index(i, j, k)] = u_local[0][b.index(i, j, k)]
                        + 0.5 * g.a2 * g.tau * g.tau * laplace_operator(g, b, u_local[0], i, j, k);
                }
            }
        }
    }
    
    // Exchange ghost layers for u^1
    exchange_ghost_layers(b, u_local[1], comm_cart);
    
    // Apply boundary conditions
    apply_boundary_conditions(g, b, u_local[1]);
    
    // Calculate error for u^1
    double local_max_error = 0.0;
    for (int i = 1; i <= b.Nx; ++i) {
        int i_global = b.x_start + i - 1;
        double x = i_global * g.h_x;
        
        for (int j = 1; j <= b.Ny; ++j) {
            int j_global = b.y_start + j - 1;
            double y = j_global * g.h_y;
            
            for (int k = 1; k <= b.Nz; ++k) {
                int k_global = b.z_start + k - 1;
                double z = k_global * g.h_z;
                
                double exact = u_analytical(g, x, y, z, g.tau);
                double err = std::abs(u_local[1][b.index(i, j, k)] - exact);
                if (err > local_max_error)
                    local_max_error = err;
            }
        }
    }
    
    double global_max_error;
    MPI_Reduce(&local_max_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX, 0, comm_cart);
    
    if (b.rank == 0) {
        max_inaccuracy = global_max_error;
        first_step_inaccuracy = global_max_error;
        std::cout << "Max start inaccuracy: " << global_max_error << std::endl;
    }
}

void run_algo(const Grid& g, Block& b, VVEC& u_local, const int& dim0_n, const int& dim1_n, 
              const int& dim2_n, MPI_Comm& comm_cart, double& max_inaccuracy, 
              double& last_step_inaccuracy) {
    int next, curr, prev;
    
    for (int s = 2; s < TIME_STEPS; ++s) {
        next = s % 3;
        curr = (s - 1) % 3;
        prev = (s - 2) % 3;
        double t = s * g.tau;
        
        // Exchange ghost layers before computation
        exchange_ghost_layers(b, u_local[curr], comm_cart);
        
        // Compute next time step
        for (int i = 1; i <= b.Nx; ++i) {
            int i_global = b.x_start + i - 1;
            double x = i_global * g.h_x;
            
            for (int j = 1; j <= b.Ny; ++j) {
                int j_global = b.y_start + j - 1;
                double y = j_global * g.h_y;
                
                for (int k = 1; k <= b.Nz; ++k) {
                    int k_global = b.z_start + k - 1;
                    double z = k_global * g.h_z;
                    
                    // Check if point is on physical boundary (Dirichlet)
                    bool on_x_boundary = (i_global == 0 || i_global == g.N);
                    bool on_z_boundary = (k_global == 0 || k_global == g.N);
                    
                    if (on_x_boundary || on_z_boundary) {
                        // Use analytical solution at boundaries
                        u_local[next][b.index(i, j, k)] = u_analytical(g, x, y, z, t);
                    } else {
                        // Interior point: explicit scheme
                        u_local[next][b.index(i, j, k)] = 2.0 * u_local[curr][b.index(i, j, k)]
                            - u_local[prev][b.index(i, j, k)]
                            + g.a2 * g.tau * g.tau * laplace_operator(g, b, u_local[curr], i, j, k);
                    }
                }
            }
        }
        
        // Exchange ghost layers after computation
        exchange_ghost_layers(b, u_local[next], comm_cart);
        
        // Apply boundary conditions
        apply_boundary_conditions(g, b, u_local[next]);
        
        // Calculate error
        double local_max_error = 0.0;
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
                    double err = std::abs(u_local[next][b.index(i, j, k)] - exact);
                    if (err > local_max_error)
                        local_max_error = err;
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

void solve_equation(const Grid& grid, Block& block, const int& dim0_n, const int& dim1_n, 
                    const int& dim2_n, MPI_Comm& comm_cart, double& time, 
                    double& max_inaccuracy, double& first_step_inaccuracy, 
                    double& last_step_inaccuracy, VDOUB& result) {
    VDOUB u0_local(block.N), u1_local(block.N), u2_local(block.N);
    VVEC u_local{u0_local, u1_local, u2_local};

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    init(grid, block, u_local, dim0_n, dim1_n, dim2_n, comm_cart, 
         max_inaccuracy, first_step_inaccuracy);
    run_algo(grid, block, u_local, dim0_n, dim1_n, dim2_n, comm_cart, 
             max_inaccuracy, last_step_inaccuracy);

    double end_time = MPI_Wtime();
    double local_time = end_time - start_time;

    MPI_Reduce(&local_time, &time, 1, MPI_DOUBLE, MPI_MAX, 0, comm_cart);

    result = u_local[(TIME_STEPS - 1) % 3];
}
