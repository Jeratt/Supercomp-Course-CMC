#define _USE_MATH_DEFINES
#include "equation.hpp"
#include <cmath>
#include <omp.h>
#include <iostream>
#include <algorithm>

using namespace std;

inline double laplace_operator(const Grid& g, const VDOUB& ui, int i, int j, int k) {
    // x-направление: граничные точки не передаются сюда (i ∈ [1, N−1])
    double d2x = (ui[g.index(i - 1, j, k)] - 2.0 * ui[g.index(i, j, k)] + ui[g.index(i + 1, j, k)]) / (g.h_x * g.h_x);

    // y-направление: периодическое
    int j_prev = (j - 1 + g.N + 1) % (g.N + 1);
    int j_next = (j + 1) % (g.N + 1);
    double d2y = (ui[g.index(i, j_prev, k)] - 2.0 * ui[g.index(i, j, k)] + ui[g.index(i, j_next, k)]) / (g.h_y * g.h_y);

    // z-направление
    double d2z = (ui[g.index(i, j, k - 1)] - 2.0 * ui[g.index(i, j, k)] + ui[g.index(i, j, k + 1)]) / (g.h_z * g.h_z);

    return d2x + d2y + d2z;
}

void init(const Grid& g, VDOUB2D& u, double& max_inacc, double& inacc_first) {
    int n_total = (g.N + 1) * (g.N + 1) * (g.N + 1);

    // --- Шаг 1: Заполняем ВСЕ точки u^0 аналитически ---
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < g.N + 1; ++i)
        for (int j = 0; j < g.N + 1; ++j)
            for (int k = 0; k < g.N + 1; ++k) {
                double x = i * g.h_x;
                double y = j * g.h_y;
                double z = k * g.h_z;
                u[0][g.index(i, j, k)] = u_analytical(g, x, y, z, 0.0);
            }

    // --- Шаг 2: Вычисляем u^1 по формуле (12) ТОЛЬКО для внутренних точек ---
    #pragma omp parallel for collapse(3)
    for (int i = 1; i < g.N; ++i)
        for (int j = 1; j < g.N; ++j)
            for (int k = 1; k < g.N; ++k) {
                u[1][g.index(i, j, k)] = u[0][g.index(i, j, k)]
                    + 0.5 * g.a2 * g.tau * g.tau * laplace_operator(g, u[0], i, j, k);
            }

    // --- Шаг 3: Граничные точки для u^1 — копируем из аналитического решения ---
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < g.N + 1; ++i)
        for (int j = 0; j < g.N + 1; ++j)
            for (int k = 0; k < g.N + 1; ++k) {
                // Пропускаем внутренние точки, они уже посчитаны
                if (i == 0 || i == g.N || j == 0 || j == g.N || k == 0 || k == g.N) {
                    double x = i * g.h_x;
                    double y = j * g.h_y;
                    double z = k * g.h_z;
                    u[1][g.index(i, j, k)] = u_analytical(g, x, y, z, g.tau);
                }
            }

    // --- Шаг 4: Принудительно обеспечиваем периодичность по y для обоих слоёв ---
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < g.N + 1; ++i)
        for (int k = 0; k < g.N + 1; ++k) {
            u[0][g.index(i, g.N, k)] = u[0][g.index(i, 0, k)];
            u[1][g.index(i, g.N, k)] = u[1][g.index(i, 0, k)];
        }

    // --- Шаг 5: Проверка погрешности на u^1 ---
    double step_max_error = 0.0;
    #pragma omp parallel for collapse(3) reduction(max : step_max_error)
    for (int i = 0; i < g.N + 1; ++i)
        for (int j = 0; j < g.N + 1; ++j)
            for (int k = 0; k < g.N + 1; ++k) {
                double x = i * g.h_x;
                double y = j * g.h_y;
                double z = k * g.h_z;
                double exact = u_analytical(g, x, y, z, g.tau);
                double err = std::abs(u[1][g.index(i, j, k)] - exact);
                step_max_error = std::max(step_max_error, err);
            }

    max_inacc = std::max(max_inacc, step_max_error);
    inacc_first = step_max_error;
    std::cout << "Max start inaccuracy: " << step_max_error << std::endl;
}

void run_algo(Grid& g, VDOUB2D& u, double& max_inacc, double& last_step_inaccuracy) {
    for (int s = 2; s < T; ++s) {
        int prev = (s - 2) % 3;
        int curr = (s - 1) % 3;
        int next = s % 3;

        // --- 1. Внутренние точки: стандартная схема ---
        #pragma omp parallel for collapse(3)
        for (int i = 1; i < g.N; ++i)
            for (int j = 1; j < g.N; ++j)
                for (int k = 1; k < g.N; ++k) {
                    u[next][g.index(i, j, k)] = 2.0 * u[curr][g.index(i, j, k)]
                        - u[prev][g.index(i, j, k)]
                        + g.a2 * g.tau * g.tau * laplace_operator(g, u[curr], i, j, k);
                }

        // --- 2. Граничные точки x=0, x=Lx (1-го рода) ---
        #pragma omp parallel for collapse(2)
        for (int j = 0; j < g.N + 1; ++j)
            for (int k = 0; k < g.N + 1; ++k) {
                u[next][g.index(0, j, k)] = 0.0;
                u[next][g.index(g.N, j, k)] = 0.0;
            }

        // --- 3. Граничные точки z=0, z=Lz (1-го рода) ---
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < g.N + 1; ++i)
            for (int j = 0; j < g.N + 1; ++j) {
                u[next][g.index(i, j, 0)] = 0.0;
                u[next][g.index(i, j, g.N)] = 0.0;
            }

        // --- 4. ВСЕ ОСТАЛЬНЫЕ ГРАНИЧНЫЕ ТОЧКИ (включая y-границы в углах) — из аналитики ---
        // Это важно: не вычисляем через схему, чтобы избежать дрейфа периодичности
        #pragma omp parallel for collapse(3)
        for (int i = 0; i < g.N + 1; ++i)
            for (int j = 0; j < g.N + 1; ++j)
                for (int k = 0; k < g.N + 1; ++k) {
                    // Пропускаем уже обработанные точки (x=0, x=N, z=0, z=N)
                    if (i == 0 || i == g.N || k == 0 || k == g.N) continue;
                    // Обрабатываем оставшиеся — в частности, y=0 и y=N при 0<i<N, 0<k<N
                    double x = i * g.h_x;
                    double y = j * g.h_y;
                    double z = k * g.h_z;
                    u[next][g.index(i, j, k)] = u_analytical(g, x, y, z, s * g.tau);
                }

        // --- 5. Принудительно обеспечиваем ПЕРИОДИЧНОСТЬ по y (включая углы!) ---
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < g.N + 1; ++i)
            for (int k = 0; k < g.N + 1; ++k) {
                u[next][g.index(i, g.N, k)] = u[next][g.index(i, 0, k)];
            }

        // --- 6. Подсчёт погрешности ---
        double step_max_error = 0.0;
        #pragma omp parallel for collapse(3) reduction(max : step_max_error)
        for (int i = 0; i < g.N + 1; ++i)
            for (int j = 0; j < g.N + 1; ++j)
                for (int k = 0; k < g.N + 1; ++k) {
                    double x = i * g.h_x;
                    double y = j * g.h_y;
                    double z = k * g.h_z;
                    double exact = u_analytical(g, x, y, z, s * g.tau);
                    double err = std::abs(u[next][g.index(i, j, k)] - exact);
                    step_max_error = std::max(step_max_error, err);
                }

        max_inacc = std::max(max_inacc, step_max_error);
        if (s == T - 1) last_step_inaccuracy = step_max_error;

        std::cout << "Max inaccuracy on step " << s << " : " << step_max_error << std::endl;
    }
}

void solve(Grid& g, double& time, double& max_inacc, double& inacc_first,
           double& last_step_inaccuracy, VDOUB& result, int threads_num) {
    omp_set_dynamic(0);
    omp_set_num_threads(threads_num);

    int n = (g.N + 1) * (g.N + 1) * (g.N + 1);
    VDOUB u0(n), u1(n), u2(n);
    VDOUB2D u{u0, u1, u2};

    double t_start = omp_get_wtime();
    max_inacc = 0.0;
    init(g, u, max_inacc, inacc_first);
    run_algo(g, u, max_inacc, last_step_inaccuracy);
    double t_end = omp_get_wtime();

    time = t_end - t_start;
    result = u[(T - 1) % 3];
}