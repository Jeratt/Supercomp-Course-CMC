#ifndef EQUATION_HPP
#define EQUATION_HPP
#include <cmath>
#include <vector>
#include <iostream>
#include <mpi.h>
#define TIME_STEPS 20
typedef std::vector<std::vector<double>> VVEC;
typedef std::vector<double> VDOUB;
typedef std::vector<int> VINT;

class Grid {
public:
    int N;
    double Lx, Ly, Lz, h_x, h_y, h_z, a2, tau;
    std::string domain_label;
    
    Grid(int N, double Lx, double Ly, double Lz, const std::string& label)
        : N(N), Lx(Lx), Ly(Ly), Lz(Lz), domain_label(label) {
        h_x = Lx / N;
        h_y = Ly / N;
        h_z = Lz / N;
        a2 = 0.25;
        // Для устойчивости при больших N уменьшаем шаг по времени
        tau = 0.00005;
    }
    
    inline int global_index(int i, int j, int k) const {
        return (i * (N + 1) + j) * (N + 1) + k;
    }
    
    inline double x_coord(int i_global) const {
        return i_global * h_x;
    }
    
    inline double y_coord(int j_global) const {
        return j_global * h_y;
    }
    
    inline double z_coord(int k_global) const {
        return k_global * h_z;
    }
};

class Block {
public:
    int Nx, Ny, Nz;
    int padded_Nx, padded_Ny, padded_Nz;
    int x_start, x_end, y_start, y_end, z_start, z_end;
    int y_start_global, y_end_global; // глобальные координаты по y для обработки периодичности
    int rank, coord_x, coord_y, coord_z;
    int dimx, dimy, dimz; // размеры декомпозиции
    VINT neighbors;
    
    VDOUB left_send,  right_send,
           bottom_send, top_send,
           front_send,  back_send;
           
    VDOUB left_recv,  right_recv,
           bottom_recv, top_recv,
           front_recv,  back_recv;
    
    Block(const Grid& g, const VINT& neighbors, const int coords[3], 
          int dimx, int dimy, int dimz, int rank) : 
        rank(rank), 
        coord_x(coords[0]),
        coord_y(coords[1]), 
        coord_z(coords[2]),
        dimx(dimx), dimy(dimy), dimz(dimz),
        neighbors(neighbors)
    {
        this->dimx = dimx;
        this->dimy = dimy;
        this->dimz = dimz;
        
        // Расчет размеров блока по оси x
        int base_x = (g.N + 1) / dimx;
        int rem_x  = (g.N + 1) % dimx;
        x_start = 0;
        for (int i = 0; i < coord_x; ++i)
            x_start += (i < dimx - rem_x) ? base_x : base_x + 1;
        Nx = (coord_x < dimx - rem_x) ? base_x : base_x + 1;
        x_end = x_start + Nx - 1;
        
        // Расчет размеров блока по оси y
        int base_y = (g.N + 1) / dimy;
        int rem_y  = (g.N + 1) % dimy;
        y_start = 0;
        y_start_global = 0;
        for (int i = 0; i < coord_y; ++i) {
            y_start += (i < dimy - rem_y) ? base_y : base_y + 1;
            y_start_global += (i < dimy - rem_y) ? base_y : base_y + 1;
        }
        Ny = (coord_y < dimy - rem_y) ? base_y : base_y + 1;
        y_end = y_start + Ny - 1;
        y_end_global = y_start_global + Ny - 1;
        
        // Расчет размеров блока по оси z
        int base_z = (g.N + 1) / dimz;
        int rem_z  = (g.N + 1) % dimz;
        z_start = 0;
        for (int i = 0; i < coord_z; ++i)
            z_start += (i < dimz - rem_z) ? base_z : base_z + 1;
        Nz = (coord_z < dimz - rem_z) ? base_z : base_z + 1;
        z_end = z_start + Nz - 1;
        
        // Padded размеры для гало-ячеек
        padded_Nx = Nx + 2;
        padded_Ny = Ny + 2;
        padded_Nz = Nz + 2;
        
        // Инициализация буферов обмена
        left_send.resize(Ny * Nz);   right_send.resize(Ny * Nz);
        left_recv.resize(Ny * Nz);   right_recv.resize(Ny * Nz);
        bottom_send.resize(Nx * Nz); top_send.resize(Nx * Nz);
        bottom_recv.resize(Nx * Nz); top_recv.resize(Nx * Nz);
        front_send.resize(Nx * Ny);  back_send.resize(Nx * Ny);
        front_recv.resize(Nx * Ny);  back_recv.resize(Nx * Ny);
    }
    
    inline int local_index(int i, int j, int k) const {
        return i * (padded_Ny * padded_Nz) + j * padded_Nz + k;
    }
};

inline double u_analytical(const Grid& g, double x, double y, double z, double t) {
    double at = (M_PI / 2.0) * std::sqrt(1.0/(g.Lx*g.Lx) + 4.0/(g.Ly*g.Ly) + 9.0/(g.Lz*g.Lz));
    return std::sin(M_PI * x / g.Lx) *
           std::sin(2.0 * M_PI * y / g.Ly) *
           std::sin(3.0 * M_PI * z / g.Lz) *
           std::cos(at * t);
}

void solve_mpi(const Grid& grid, Block& block,
               int dimx, int dimy, int dimz,
               MPI_Comm comm_cart,
               double& time,
               double& max_inaccuracy,
               double& first_step_inaccuracy,
               double& last_step_inaccuracy,
               VDOUB& result);
#endif