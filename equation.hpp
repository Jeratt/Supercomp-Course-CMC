#ifndef EQUATION_HPP
#define EQUATION_HPP
#include <cmath>
#include <vector>
#include <iostream>
#include <mpi.h>
#include <string>
typedef std::vector<std::vector<double>> VDOUB2D;
typedef std::vector<double> VDOUB;
typedef std::vector<VDOUB> VVEC;
#define TIME_STEPS 20  // 20 временных шагов

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
        a2 = 0.25;  // задано в варианте 3
        // Для устойчивости (условие Куранта): τ ≤ h_min / (a * sqrt(3))
        tau = 0.00005;  // безопасное значение для N от 128 до 512 (как было)
    }
    inline int index(int i, int j, int k) const {
        return (i * (N + 1) + j) * (N + 1) + k;
    }
};

class Block {
public:
    int rank;
    std::vector<int> neighbors;
    int x_start, x_end, y_start, y_end, z_start, z_end;
    int Nx, Ny, Nz, padded_Nx, padded_Ny, padded_Nz;
    std::vector<double> left_send, left_recv, right_send, right_recv;
    std::vector<double> bottom_send, bottom_recv, top_send, top_recv;
    std::vector<double> front_send, front_recv, back_send, back_recv;
    
    Block(const Grid& g, const std::vector<int>& nb, const int coords[3],
          int dimx, int dimy, int dimz, int r) : rank(r), neighbors(nb) {
        // Расчёт локальных границ блока (как в исходном)
        x_start = coords[0] * (g.N / dimx);
        x_end = (coords[0] + 1) * (g.N / dimx);
        if (coords[0] == dimx - 1) x_end = g.N;
        y_start = coords[1] * (g.N / dimy);
        y_end = (coords[1] + 1) * (g.N / dimy);
        if (coords[1] == dimy - 1) y_end = g.N;
        z_start = coords[2] * (g.N / dimz);
        z_end = (coords[2] + 1) * (g.N / dimz);
        if (coords[2] == dimz - 1) z_end = g.N;
        Nx = x_end - x_start;
        Ny = y_end - y_start;
        Nz = z_end - z_start;
        // Размеры с учётом гало-зон (1 слой с каждой стороны)
        padded_Nx = Nx + 2;
        padded_Ny = Ny + 2;
        padded_Nz = Nz + 2;
        // Инициализация буферов для обмена гало-зонами
        int yz_size = Ny * Nz;
        int xz_size = Nx * Nz;
        int xy_size = Nx * Ny;
        left_send.resize(yz_size); left_recv.resize(yz_size);
        right_send.resize(yz_size); right_recv.resize(yz_size);
        bottom_send.resize(xz_size); bottom_recv.resize(xz_size);
        top_send.resize(xz_size); top_recv.resize(xz_size);
        front_send.resize(xy_size); front_recv.resize(xy_size);
        back_send.resize(xy_size); back_recv.resize(xy_size);
    }
    
    inline int local_index(int i, int j, int k) const {
        return ((i) * padded_Ny + (j)) * padded_Nz + (k);
    }
};

inline double u_analytical(const Grid& g, double x, double y, double z, double t) {
    double at = (M_PI / 2.0) * std::sqrt(1.0 / (g.Lx * g.Lx) + 4.0 / (g.Ly * g.Ly) + 9.0 / (g.Lz * g.Lz));
    return std::sin(M_PI * x / g.Lx)
         * std::sin(2.0 * M_PI * y / g.Ly)
         * std::sin(3.0 * M_PI * z / g.Lz)
         * std::cos(at * t);
}

void solve_mpi(const Grid& g, Block& b,
               int dimx, int dimy, int dimz,
               MPI_Comm comm_cart,
               double& time,
               double& max_inaccuracy,
               double& first_step_inaccuracy,
               double& last_step_inaccuracy,
               VDOUB& result);
#endif