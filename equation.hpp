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
    char* L_type;
    
    Grid() {}
    
    Grid(int N, char* L_type, double Lx, double Ly, double Lz) {
        this->N = N;
        this->L_type = L_type;
        this->Lx = Lx;
        this->Ly = Ly;
        this->Lz = Lz;
        this->h_x = Lx / N;
        this->h_y = Ly / N;
        this->h_z = Lz / N;
        this->a2 = 0.25;  // Для варианта 3
        this->tau = 0.00005;  // Уменьшен для устойчивости
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
    int dim0_rank, dim1_rank, dim2_rank;
    int rank;
    VINT neighbors;
    
    // Буферы для обмена (периодические граничные условия по Y)
    VDOUB left_send, right_send;      // Обмен по X (условия 1-го рода)
    VDOUB bottom_send, top_send;      // Обмен по Y (периодические условия)
    VDOUB front_send, back_send;      // Обмен по Z (условия 1-го рода)
    
    VDOUB left_recieve, right_recieve;
    VDOUB bottom_recieve, top_recieve;
    VDOUB front_recieve, back_recieve;
    
    Block(const Grid& g, const VINT& neighbors, const int coords[3], 
          int dimx, int dimy, int dimz, int rank) : 
        rank(rank), 
        dim0_rank(coords[0]),  // X
        dim1_rank(coords[1]),  // Y (периодическое направление)
        dim2_rank(coords[2]),  // Z
        neighbors(neighbors)
    {
        // Расчет размеров блока по оси X (условия 1-го рода)
        int base_x = (g.N + 1) / dimx;
        int rem_x  = (g.N + 1) % dimx;
        x_start = 0;
        for (int i = 0; i < dim0_rank; ++i)
            x_start += (i < dimx - rem_x) ? base_x : base_x + 1;
        Nx = (dim0_rank < dimx - rem_x) ? base_x : base_x + 1;
        x_end = x_start + Nx - 1;
        
        // Расчет размеров блока по оси Y (периодические условия)
        int base_y = (g.N + 1) / dimy;
        int rem_y  = (g.N + 1) % dimy;
        y_start = 0;
        for (int i = 0; i < dim1_rank; ++i)
            y_start += (i < dimy - rem_y) ? base_y : base_y + 1;
        Ny = (dim1_rank < dimy - rem_y) ? base_y : base_y + 1;
        y_end = y_start + Ny - 1;
        
        // Расчет размеров блока по оси Z (условия 1-го рода)
        int base_z = (g.N + 1) / dimz;
        int rem_z  = (g.N + 1) % dimz;
        z_start = 0;
        for (int i = 0; i < dim2_rank; ++i)
            z_start += (i < dimz - rem_z) ? base_z : base_z + 1;
        Nz = (dim2_rank < dimz - rem_z) ? base_z : base_z + 1;
        z_end = z_start + Nz - 1;
        
        // Padded размеры для гало-ячеек
        padded_Nx = Nx + 2;
        padded_Ny = Ny + 2;
        padded_Nz = Nz + 2;
        
        // Инициализация буферов обмена
        left_send.resize(Ny * Nz);   right_send.resize(Ny * Nz);
        left_recieve.resize(Ny * Nz); right_recieve.resize(Ny * Nz);
        bottom_send.resize(Nx * Nz); top_send.resize(Nx * Nz);
        bottom_recieve.resize(Nx * Nz); top_recieve.resize(Nx * Nz);
        front_send.resize(Nx * Ny);  back_send.resize(Nx * Ny);
        front_recieve.resize(Nx * Ny); back_recieve.resize(Nx * Ny);
    }
    
    inline int index(int i, int j, int k) const {
        return i * (padded_Ny * padded_Nz) + j * padded_Nz + k;
    }
};

// Аналитическое решение для варианта 3: 1Р П 1Р
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