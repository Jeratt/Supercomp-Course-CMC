#define _USE_MATH_DEFINES
#include "equation.hpp"
#include <cmath>
#include <chrono>
#include <omp.h>
#include <iostream>

using namespace std;

inline double laplace_operator(const Grid& g, const VDOUB& ui, const int& i, const int& j, const int& k) {
    // x- first order
    double d2x = (ui[g.index(i - 1, j, k)] - 2.0 * ui[g.index(i, j, k)] + ui[g.index(i + 1, j, k)]) / (g.h_x * g.h_x);

    // y - periodic
    int j_prev = (j - 1 + g.N + 1) % (g.N + 1);
    int j_next = (j + 1) % (g.N + 1);
    double d2y = (ui[g.index(i, j_prev, k)] - 2.0 * ui[g.index(i, j, k)] + ui[g.index(i, j_next, k)]) / (g.h_y * g.h_y);

    // z - first order
    double d2z = (ui[g.index(i, j, k - 1)] - 2.0 * ui[g.index(i, j, k)] + ui[g.index(i, j, k + 1)]) / (g.h_z * g.h_z);

    return d2x + d2y + d2z;
}

void init(const Grid& g, VDOUB2D& u, double& max_inacc, double& inacc_first) {
    // x - first order
    #pragma omp parallel for collapse(2)
    for (int j = 0; j < g.N + 1; ++j)
    {
        for (int k = 0; k < g.N + 1; ++k) {
            u[0][g.index(0, j, k)] = 0.0;
            u[0][g.index(g.N, j, k)] = 0.0;
            u[1][g.index(0, j, k)] = 0.0;
            u[1][g.index(g.N, j, k)] = 0.0;
        }
    }

    // z - first order
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < g.N + 1; ++i)
    {
        for (int j = 0; j < g.N + 1; ++j) {
            u[0][g.index(i, j, 0)] = 0.0;
            u[0][g.index(i, j, g.N)] = 0.0;
            u[1][g.index(i, j, 0)] = 0.0;
            u[1][g.index(i, j, g.N)] = 0.0;
        }
    }

    // y - periodic
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < g.N + 1; ++i)
    {
        for (int k = 0; k < g.N + 1; ++k) {
            double y0_val = u_analytical(g, i * g.h_x, 0.0, k * g.h_z, 0.0);
            double yN_val = u_analytical(g, i * g.h_x, g.Ly, k * g.h_z, 0.0);
            u[0][g.index(i, 0, k)] = y0_val;
            u[0][g.index(i, g.N, k)] = yN_val;

            y0_val = u_analytical(g, i * g.h_x, 0.0, k * g.h_z, g.tau);
            yN_val = u_analytical(g, i * g.h_x, g.Ly, k * g.h_z, g.tau);
            u[1][g.index(i, 0, k)] = y0_val;
            u[1][g.index(i, g.N, k)] = yN_val;
        }
    }

    // u_0 
    #pragma omp parallel for collapse(3)
    for (int i = 1; i < g.N; ++i)
        for (int j = 1; j < g.N; ++j)
            for (int k = 1; k < g.N; ++k)
                u[0][g.index(i, j, k)] = u_analytical(g, i * g.h_x, j * g.h_y, k * g.h_z, 0.0);

    // u_1 
    #pragma omp parallel for collapse(3)
    for (int i = 1; i < g.N; ++i)
        for (int j = 1; j < g.N; ++j)
            for (int k = 1; k < g.N; ++k)
                u[1][g.index(i, j, k)] = u[0][g.index(i, j, k)] + 0.5 * g.a2 * pow(g.tau, 2) * laplace_operator(g, u[0], i, j, k);

    // innacuracy 
    double step_max_error = -1;
    #pragma omp parallel for collapse(3) reduction(max : step_max_error)
    for (int i = 0; i < g.N + 1; ++i)
    {
        for (int j = 0; j < g.N + 1; ++j)
        {
            for (int k = 0; k < g.N + 1; ++k)
            {
                double tmp = fabs(u[1][g.index(i, j, k)] - u_analytical(g, i * g.h_x, j * g.h_y, k * g.h_z, g.tau));
                if (tmp > step_max_error)
                {
                    step_max_error = tmp;
                }
            }
        }
    }

    max_inacc = max(max_inacc, step_max_error);
    inacc_first = step_max_error;

    cout << "Max start inaccuracy:" << " " << step_max_error << endl;
}

void run_algo(Grid& g, VDOUB2D& u, double& max_inacc, double& last_step_inaccuracy) {
    for (int s = 2; s < T; ++s) {
        int prev = (s - 2) % 3;
        int curr = (s - 1) % 3;
        int next = s % 3;

        #pragma omp parallel for collapse(3)
        for (int i = 1; i < g.N; ++i)
        {
            for (int j = 1; j < g.N; ++j)
            {
                for (int k = 1; k < g.N; ++k)
                {
                    u[next][g.index(i, j, k)] = 2 * u[curr][g.index(i, j, k)] - u[prev][g.index(i, j, k)]
                     + g.a2 * pow(g.tau, 2) * laplace_operator(g, u[curr], i, j, k);
                }
            }
        }


        // x - first order
        #pragma omp parallel for collapse(2)
        for (int j = 0; j < g.N + 1; ++j)
        {
            for (int k = 0; k < g.N + 1; ++k)
            {
                u[next][g.index(0, j, k)] = 0.0;
                u[next][g.index(g.N, j, k)] = 0.0;
            }
        }

        // z - first order
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < g.N + 1; ++i)
        {
            for (int j = 0; j < g.N + 1; ++j)
            {
                u[next][g.index(i, j, 0)] = 0.0;
                u[next][g.index(i, j, g.N)] = 0.0;
            }
        }

        // y - periodic
        #pragma omp parallel for collapse(2)
        for (int i = 1; i < g.N; ++i)
        {
            for (int k = 1; k < g.N; ++k)
            {
                u[next][g.index(i, 0, k)] = 2 * u[curr][g.index(i, 0, k)] - u[prev][g.index(i, 0, k)]
                                            + g.a2 * pow(g.tau, 2) * laplace_operator(g, u[curr], i, 0, k);
                u[next][g.index(i, g.N, k)] = 2 * u[curr][g.index(i, g.N, k)] - u[prev][g.index(i, g.N, k)]
                                              + g.a2 * pow(g.tau, 2) * laplace_operator(g, u[curr], i, g.N, k);
            }
        }

        // inaccuracy
        double step_max_error = -1;
        #pragma omp parallel for collapse(3) reduction(max : step_max_error)
        for (int i = 0; i < g.N + 1; ++i)
        {
            for (int j = 0; j < g.N + 1; ++j)
            {
                for (int k = 0; k < g.N + 1; ++k)
                {
                    double tmp = fabs(u[next][g.index(i, j, k)] - u_analytical(g, i * g.h_x, j * g.h_y, k * g.h_z, g.tau * s));
                    if (tmp > step_max_error) step_max_error = tmp;
                }
            }
        }

        if (step_max_error > max_inacc)
            max_inacc = step_max_error;
        if (s == T - 1)
            last_step_inaccuracy = step_max_error;

        cout << "Max inaccuracy on step " << s << " : " << step_max_error << endl;
    }
}

void solve(Grid& g, double& time, double& max_inacc, double& inacc_first, double& last_step_inaccuracy, VDOUB& result, int& threads_num) {
	omp_set_dynamic(0);
	omp_set_num_threads(threads_num);

	int n = pow(g.N + 1, 3);
	VDOUB u0(n), u1(n), u2(n);
	VDOUB2D u{u0, u1, u2};

	auto start = omp_get_wtime();

	init(g, u, max_inacc, inacc_first);
	run_algo(g, u, max_inacc, last_step_inaccuracy);

	auto end = omp_get_wtime();
	time = end - start;

	result = u[(T - 1) % 3];
}