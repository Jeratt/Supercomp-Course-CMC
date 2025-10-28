#ifndef EQUATION_HPP
#define EQUATION_HPP

#include <cmath>
#include <vector>
#include <iostream>

typedef std::vector< std::vector<double>> VDOUB2D;
typedef std::vector<double> VDOUB;

#define T 20

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
        tau = 0.0001;
    }

    inline int index(const int& i, const int& j, const int& k) const {
        return (i * (N + 1) + j) * (N + 1) + k;
    }
};

inline double u_analytical(const Grid& g, const double& x, const double& y, const double& z, const double& t) {
    double at = (M_PI / 2.0) * sqrt(1.0 / (g.Lx * g.Lx) + 4.0 / (g.Ly * g.Ly) + 9.0 / (g.Lz * g.Lz));
    return sin(M_PI * x / g.Lx) * sin(2.0 * M_PI * y / g.Ly) * sin(3.0 * M_PI * z / g.Lz) * cos(at * t);
}

void solve(Grid& grid, double& time, double& max_inaccuracy, double& first_step_inaccuracy, double& last_step_inaccuracy, VDOUB& result, int& threads_num);

#endif