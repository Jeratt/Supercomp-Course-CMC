#ifndef EQUATION_HPP
#define EQUATION_HPP
#include <cmath>
#include <vector>
#include <iostream>
typedef std::vector<std::vector<double>> VDOUB2D;
typedef std::vector<double> VDOUB;
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
        tau = 0.00005;  // безопасное значение для N от 128 до 512
    }
    inline int index(int i, int j, int k) const {
        return (i * (N + 1) + j) * (N + 1) + k;
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
