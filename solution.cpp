#define _USE_MATH_DEFINES
#include "equation.h"
#include <algorithm>
#include <iostream>
#include <chrono>
#include <cmath>
#include <mpi.h>

// update halo for neighbors processes
void exchange_ghost_layers(Block& b, VDOUB& ui_local, MPI_Comm& comm_cart, const Grid& g, const int& dim0_n, const int& dim1_n, const int& dim2_n) {
    MPI_Request reqs[12];
    int req_count = 0;
    
    // Send and receive for Y direction (periodic)
    if (dim0_n > 1) {
        // Left (Y-)
        if (b.neighbors[0] != MPI_PROC_NULL && b.neighbors[0] != b.rank) {
            MPI_Isend(b.left_send.data(), b.Ny * b.Nz, MPI_DOUBLE, b.neighbors[0], b.rank * 6 + 0, comm_cart, &reqs[req_count++]);
            MPI_Irecv(b.left_recieve.data(), b.Ny * b.Nz, MPI_DOUBLE, b.neighbors[1], b.neighbors[1] * 6 + 0, comm_cart, &reqs[req_count++]);
        }
        // Right (Y+)
        if (b.neighbors[1] != MPI_PROC_NULL && b.neighbors[1] != b.rank) {
            MPI_Isend(b.right_send.data(), b.Ny * b.Nz, MPI_DOUBLE, b.neighbors[1], b.rank * 6 + 1, comm_cart, &reqs[req_count++]);
            MPI_Irecv(b.right_recieve.data(), b.Ny * b.Nz, MPI_DOUBLE, b.neighbors[0], b.neighbors[0] * 6 + 1, comm_cart, &reqs[req_count++]);
        }
    }
    
    // Send and receive for X direction (Dirichlet on boundaries)
    if (dim1_n > 1) {
        // Bottom (X-)
        if (b.neighbors[2] != MPI_PROC_NULL && b.neighbors[2] != b.rank) {
            MPI_Isend(b.bottom_send.data(), b.Nx * b.Nz, MPI_DOUBLE, b.neighbors[2], b.rank * 6 + 2, comm_cart, &reqs[req_count++]);
            MPI_Irecv(b.bottom_recieve.data(), b.Nx * b.Nz, MPI_DOUBLE, b.neighbors[3], b.neighbors[3] * 6 + 2, comm_cart, &reqs[req_count++]);
        }
        // Top (X+)
        if (b.neighbors[3] != MPI_PROC_NULL && b.neighbors[3] != b.rank) {
            MPI_Isend(b.top_send.data(), b.Nx * b.Nz, MPI_DOUBLE, b.neighbors[3], b.rank * 6 + 3, comm_cart, &reqs[req_count++]);
            MPI_Irecv(b.top_recieve.data(), b.Nx * b.Nz, MPI_DOUBLE, b.neighbors[2], b.neighbors[2] * 6 + 3, comm_cart, &reqs[req_count++]);
        }
    }
    
    // Send and receive for Z direction (Dirichlet on boundaries)
    if (dim2_n > 1) {
        // Front (Z-)
        if (b.neighbors[4] != MPI_PROC_NULL && b.neighbors[4] != b.rank) {
            MPI_Isend(b.front_send.data(), b.Nx * b.Ny, MPI_DOUBLE, b.neighbors[4], b.rank * 6 + 4, comm_cart, &reqs[req_count++]);
            MPI_Irecv(b.front_recieve.data(), b.Nx * b.Ny, MPI_DOUBLE, b.neighbors[5], b.neighbors[5] * 6 + 4, comm_cart, &reqs[req_count++]);
        }
        // Back (Z+)
        if (b.neighbors[5] != MPI_PROC_NULL && b.neighbors[5] != b.rank) {
            MPI_Isend(b.back_send.data(), b.Nx * b.Ny, MPI_DOUBLE, b.neighbors[5], b.rank * 6 + 5, comm_cart, &reqs[req_count++]);
            MPI_Irecv(b.back_recieve.data(), b.Nx * b.Ny, MPI_DOUBLE, b.neighbors[4], b.neighbors[4] * 6 + 5, comm_cart, &reqs[req_count++]);
        }
    }
    
    if (req_count > 0) {
        MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
    }
    
    // Copy received ghost layers into ui_local
    // For Y direction (periodic)
    if (dim0_n > 1) {
        if (b.neighbors[0] != MPI_PROC_NULL && b.neighbors[0] != b.rank) {
            for (int i = 1; i < b.Nx + 1; ++i) {
                for (int k = 1; k < b.Nz + 1; ++k) {
                    ui_local[b.index(i, 0, k)] = b.left_recieve[(i-1) * b.Nz + (k-1)];
                }
            }
        }
        if (b.neighbors[1] != MPI_PROC_NULL && b.neighbors[1] != b.rank) {
            for (int i = 1; i < b.Nx + 1; ++i) {
                for (int k = 1; k < b.Nz + 1; ++k) {
                    ui_local[b.index(i, b.Ny + 1, k)] = b.right_recieve[(i-1) * b.Nz + (k-1)];
                }
            }
        }
    }
    
    // For X direction (Dirichlet)
    if (dim1_n > 1) {
        if (b.neighbors[2] != MPI_PROC_NULL && b.neighbors[2] != b.rank) {
            for (int j = 1; j < b.Ny + 1; ++j) {
                for (int k = 1; k < b.Nz + 1; ++k) {
                    ui_local[b.index(0, j, k)] = b.bottom_recieve[(j-1) * b.Nz + (k-1)];
                }
            }
        }
        if (b.neighbors[3] != MPI_PROC_NULL && b.neighbors[3] != b.rank) {
            for (int j = 1; j < b.Ny + 1; ++j) {
                for (int k = 1; k < b.Nz + 1; ++k) {
                    ui_local[b.index(b.Nx + 1, j, k)] = b.top_recieve[(j-1) * b.Nz + (k-1)];
                }
            }
        }
    }
    
    // For Z direction (Dirichlet)
    if (dim2_n > 1) {
        if (b.neighbors[4] != MPI_PROC_NULL && b.neighbors[4] != b.rank) {
            for (int i = 1; i < b.Nx + 1; ++i) {
                for (int j = 1; j < b.Ny + 1; ++j) {
                    ui_local[b.index(i, j, 0)] = b.front_recieve[(i-1) * b.Ny + (j-1)];
                }
            }
        }
        if (b.neighbors[5] != MPI_PROC_NULL && b.neighbors[5] != b.rank) {
            for (int i = 1; i < b.Nx + 1; ++i) {
                for (int j = 1; j < b.Ny + 1; ++j) {
                    ui_local[b.index(i, j, b.Nz + 1)] = b.back_recieve[(i-1) * b.Ny + (j-1)];
                }
            }
        }
    }
    
    // Apply boundary conditions
    // X direction (Dirichlet - zeros on boundaries)
    if (b.is_x_boundary_start) {
        for (int j = 0; j < b.padded_Ny; ++j) {
            for (int k = 0; k < b.padded_Nz; ++k) {
                ui_local[b.index(0, j, k)] = 0.0;
            }
        }
    }
    if (b.is_x_boundary_end) {
        for (int j = 0; j < b.padded_Ny; ++j) {
            for (int k = 0; k < b.padded_Nz; ++k) {
                ui_local[b.index(b.Nx + 1, j, k)] = 0.0;
            }
        }
    }
    
    // Y direction (periodic)
    if (dim0_n == 1) {
        // Periodic conditions within a single block
        for (int i = 0; i < b.padded_Nx; ++i) {
            for (int k = 0; k < b.padded_Nz; ++k) {
                ui_local[b.index(i, 0, k)] = ui_local[b.index(i, b.Ny, k)];
                ui_local[b.index(i, b.Ny + 1, k)] = ui_local[b.index(i, 1, k)];
            }
        }
    }
    
    // Z direction (Dirichlet - zeros on boundaries)
    if (b.is_z_boundary_start) {
        for (int i = 0; i < b.padded_Nx; ++i) {
            for (int j = 0; j < b.padded_Ny; ++j) {
                ui_local[b.index(i, j, 0)] = 0.0;
            }
        }
    }
    if (b.is_z_boundary_end) {
        for (int i = 0; i < b.padded_Nx; ++i) {
            for (int j = 0; j < b.padded_Ny; ++j) {
                ui_local[b.index(i, j, b.Nz + 1)] = 0.0;
            }
        }
    }
}

inline void fill_send_buffers(Block& b, VDOUB& ui_local, const int& dim0_n, const int& dim1_n, const int& dim2_n, const int& i, const int& j, const int& k) {
    // Y direction (periodic)
    if (dim0_n > 1) {
        if (b.is_y_boundary_start && j == 1) {
            int idx = (i-1) * b.Nz + (k-1);
            if (idx >= 0 && idx < b.left_send.size()) {
                b.left_send[idx] = ui_local[b.index(i, j, k)];
            }
        }
        if (b.is_y_boundary_end && j == b.Ny) {
            int idx = (i-1) * b.Nz + (k-1);
            if (idx >= 0 && idx < b.right_send.size()) {
                b.right_send[idx] = ui_local[b.index(i, j, k)];
            }
        }
    }
    
    // X direction (Dirichlet - no need to send boundary values as they are zero)
    if (dim1_n > 1) {
        if (!b.is_x_boundary_start && i == 1) {
            int idx = (j-1) * b.Nz + (k-1);
            if (idx >= 0 && idx < b.bottom_send.size()) {
                b.bottom_send[idx] = ui_local[b.index(i, j, k)];
            }
        }
        if (!b.is_x_boundary_end && i == b.Nx) {
            int idx = (j-1) * b.Nz + (k-1);
            if (idx >= 0 && idx < b.top_send.size()) {
                b.top_send[idx] = ui_local[b.index(i, j, k)];
            }
        }
    }
    
    // Z direction (Dirichlet - no need to send boundary values as they are zero)
    if (dim2_n > 1) {
        if (!b.is_z_boundary_start && k == 1) {
            int idx = (i-1) * b.Ny + (j-1);
            if (idx >= 0 && idx < b.front_send.size()) {
                b.front_send[idx] = ui_local[b.index(i, j, k)];
            }
        }
        if (!b.is_z_boundary_end && k == b.Nz) {
            int idx = (i-1) * b.Ny + (j-1);
            if (idx >= 0 && idx < b.back_send.size()) {
                b.back_send[idx] = ui_local[b.index(i, j, k)];
            }
        }
    }
}

inline double laplace_operator(const Grid& g, const Block& b, const VDOUB& ui_local, const int& i, const int& j, const int& k) {
    return (ui_local[b.index(i - 1, j, k)] - 2 * ui_local[b.index(i, j, k)] + ui_local[b.index(i + 1, j, k)]) / (g.h_x * g.h_x) +
           (ui_local[b.index(i, j - 1, k)] - 2 * ui_local[b.index(i, j, k)] + ui_local[b.index(i, j + 1, k)]) / (g.h_y * g.h_y) +
           (ui_local[b.index(i, j, k - 1)] - 2 * ui_local[b.index(i, j, k)] + ui_local[b.index(i, j, k + 1)]) / (g.h_z * g.h_z);
}

void init(const Grid& g, Block& b, VVEC& u_local, const int& dim0_n, const int& dim1_n, const int& dim2_n, MPI_Comm& comm_cart, double& max_inaccuracy, double& first_step_inaccuracy) {
    // Initialize all values to zero
    for (int i = 0; i < b.N; ++i) {
        u_local[0][i] = 0.0;
        u_local[1][i] = 0.0;
    }
    
    // Set boundary conditions and initial values
    for (int i = 1; i < b.Nx + 1; ++i) {
        double global_x = (b.x_start + i - 1) * g.h_x;
        for (int j = 1; j < b.Ny + 1; ++j) {
            double global_y = (b.y_start + j - 1) * g.h_y;
            for (int k = 1; k < b.Nz + 1; ++k) {
                double global_z = (b.z_start + k - 1) * g.h_z;
                
                bool on_x_boundary = (b.is_x_boundary_start && i == 1) || (b.is_x_boundary_end && i == b.Nx);
                bool on_z_boundary = (b.is_z_boundary_start && k == 1) || (b.is_z_boundary_end && k == b.Nz);
                
                if (on_x_boundary || on_z_boundary) {
                    // Dirichlet boundary conditions (zero)
                    u_local[0][b.index(i, j, k)] = 0.0;
                } else {
                    // Internal points - use analytical solution for t=0
                    u_local[0][b.index(i, j, k)] = u_analytical(g, global_x, global_y, global_z, 0.0);
                }
                
                // Fill send buffers
                fill_send_buffers(b, u_local[0], dim0_n, dim1_n, dim2_n, i, j, k);
            }
        }
    }
    
    // Exchange ghost layers
    exchange_ghost_layers(b, u_local[0], comm_cart, g, dim0_n, dim1_n, dim2_n);
    
    // Calculate u_1 (12)
    for (int i = 1; i < b.Nx + 1; ++i) {
        double global_x = (b.x_start + i - 1) * g.h_x;
        for (int j = 1; j < b.Ny + 1; ++j) {
            double global_y = (b.y_start + j - 1) * g.h_y;
            for (int k = 1; k < b.Nz + 1; ++k) {
                double global_z = (b.z_start + k - 1) * g.h_z;
                
                bool on_x_boundary = (b.is_x_boundary_start && i == 1) || (b.is_x_boundary_end && i == b.Nx);
                bool on_z_boundary = (b.is_z_boundary_start && k == 1) || (b.is_z_boundary_end && k == b.Nz);
                
                if (on_x_boundary || on_z_boundary) {
                    // Dirichlet boundary conditions (zero)
                    u_local[1][b.index(i, j, k)] = 0.0;
                } else {
                    // Internal points
                    u_local[1][b.index(i, j, k)] = u_local[0][b.index(i, j, k)] + 0.5 * (g.tau * g.tau) * laplace_operator(g, b, u_local[0], i, j, k);
                }
                
                // Fill send buffers
                fill_send_buffers(b, u_local[1], dim0_n, dim1_n, dim2_n, i, j, k);
            }
        }
    }
    
    // Exchange ghost layers for u_1
    exchange_ghost_layers(b, u_local[1], comm_cart, g, dim0_n, dim1_n, dim2_n);
    
    // Calculate error for step 1
    double error = -1.0;
    for (int i = 1; i < b.Nx + 1; ++i) {
        double global_x = (b.x_start + i - 1) * g.h_x;
        for (int j = 1; j < b.Ny + 1; ++j) {
            double global_y = (b.y_start + j - 1) * g.h_y;
            for (int k = 1; k < b.Nz + 1; ++k) {
                double global_z = (b.z_start + k - 1) * g.h_z;
                
                double analytical_value = u_analytical(g, global_x, global_y, global_z, g.tau);
                double tmp = fabs(u_local[1][b.index(i, j, k)] - analytical_value);
                if (tmp > error)
                    error = tmp;
            }
        }
    }
    
    double step_max_error = -1.0;
    MPI_Reduce(&error, &step_max_error, 1, MPI_DOUBLE, MPI_MAX, 0, comm_cart);
    
    if (b.rank == 0) {
        if (step_max_error > max_inaccuracy)
            max_inaccuracy = step_max_error;
        first_step_inaccuracy = step_max_error;
        std::cout << "Steps inaccuracy:\n\tMax inaccuracy on step 1 = " << step_max_error << std::endl;
    }
}

void run_algo(const Grid& g, Block& b, VVEC& u_local, const int& dim0_n, const int& dim1_n, const int& dim2_n, MPI_Comm& comm_cart, double& max_inaccuracy, double& last_step_inaccuracy) {
    int next, curr, prev;
    for (int s = 2; s < TIME_STEPS; ++s) {
        next = s % 3;
        curr = (s - 1) % 3;
        prev = (s - 2) % 3;
        
        // Exchange ghost layers for current time step
        exchange_ghost_layers(b, u_local[curr], comm_cart, g, dim0_n, dim1_n, dim2_n);
        
        // Calculate next time step
        for (int i = 1; i < b.Nx + 1; ++i) {
            double global_x = (b.x_start + i - 1) * g.h_x;
            for (int j = 1; j < b.Ny + 1; ++j) {
                double global_y = (b.y_start + j - 1) * g.h_y;
                for (int k = 1; k < b.Nz + 1; ++k) {
                    double global_z = (b.z_start + k - 1) * g.h_z;
                    
                    bool on_x_boundary = (b.is_x_boundary_start && i == 1) || (b.is_x_boundary_end && i == b.Nx);
                    bool on_z_boundary = (b.is_z_boundary_start && k == 1) || (b.is_z_boundary_end && k == b.Nz);
                    
                    if (on_x_boundary || on_z_boundary) {
                        // Dirichlet boundary conditions (zero)
                        u_local[next][b.index(i, j, k)] = 0.0;
                    } else {
                        // Internal points
                        u_local[next][b.index(i, j, k)] = 2 * u_local[curr][b.index(i, j, k)] - u_local[prev][b.index(i, j, k)] + 
                                                        (g.tau * g.tau) * laplace_operator(g, b, u_local[curr], i, j, k);
                    }
                    
                    // Fill send buffers
                    fill_send_buffers(b, u_local[next], dim0_n, dim1_n, dim2_n, i, j, k);
                }
            }
        }
        
        // Calculate error for current step
        double error = -1.0;
        for (int i = 1; i < b.Nx + 1; ++i) {
            double global_x = (b.x_start + i - 1) * g.h_x;
            for (int j = 1; j < b.Ny + 1; ++j) {
                double global_y = (b.y_start + j - 1) * g.h_y;
                for (int k = 1; k < b.Nz + 1; ++k) {
                    double global_z = (b.z_start + k - 1) * g.h_z;
                    
                    double analytical_value = u_analytical(g, global_x, global_y, global_z, g.tau * s);
                    double tmp = fabs(u_local[next][b.index(i, j, k)] - analytical_value);
                    if (tmp > error)
                        error = tmp;
                }
            }
        }
        
        double step_max_error = -1.0;
        MPI_Reduce(&error, &step_max_error, 1, MPI_DOUBLE, MPI_MAX, 0, comm_cart);
        
        if (b.rank == 0) {
            if (step_max_error > max_inaccuracy)
                max_inaccuracy = step_max_error;
            
            if (s == TIME_STEPS - 1)
                last_step_inaccuracy = step_max_error;
                
            std::cout << "\tMax inaccuracy on step " << s << " = " << step_max_error << std::endl;
        }
    }
}

void solve_equation(const Grid& grid, Block& block, const int& dim0_n, const int& dim1_n, const int& dim2_n, MPI_Comm& comm_cart, double& time, double& max_inaccuracy, double& first_step_inaccuracy, double& last_step_inaccuracy, VDOUB& result) {
    VDOUB u0_local(block.N, 0.0), u1_local(block.N, 0.0), u2_local(block.N, 0.0);
    VVEC u_local{u0_local, u1_local, u2_local};
    
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    
    init(grid, block, u_local, dim0_n, dim1_n, dim2_n, comm_cart, max_inaccuracy, first_step_inaccuracy);
    run_algo(grid, block, u_local, dim0_n, dim1_n, dim2_n, comm_cart, max_inaccuracy, last_step_inaccuracy);
    
    double end_time = MPI_Wtime();
    double local_time = end_time - start_time;
    MPI_Reduce(&local_time, &time, 1, MPI_DOUBLE, MPI_MAX, 0, comm_cart);
    
    result = u_local[(TIME_STEPS - 1) % 3];
}