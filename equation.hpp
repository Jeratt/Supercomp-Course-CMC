#ifndef EQUATION_H
#define EQUATION_H

#include <iostream>
#include <vector>
#include <cmath>
#include <mpi.h>
#include <algorithm> // Для std::min/max

#define TIME_STEPS 20

typedef std::vector< std::vector<double>> VVEC;
typedef std::vector<double> VDOUB;
typedef std::vector<int> VINT;

class Grid {
  public:
    int N;
    double Lx, Ly, Lz, h_x, h_y, h_z, a2, tau;
    std::string domain_label;
    std::string label_x, label_y, label_z;

    Grid(int N, double Lx, double Ly, double Lz, const std::string& label_x, const std::string& label_y, const std::string& label_z)
        : N(N), Lx(Lx), Ly(Ly), Lz(Lz), domain_label(label_x + "_" + label_y + "_" + label_z), label_x(label_x), label_y(label_y), label_z(label_z) {
        h_x = Lx / N;
        h_y = Ly / N;
        h_z = Lz / N;
        a2 = 0.25;  // a^2 = 1/4 for variant 3
        // Примерная настройка tau для устойчивости
        // double min_h = std::min({h_x, h_y, h_z}); // Не поддерживается в g++ 4.8.2 без C++11
        // Используем совместимый способ для нахождения минимума из трех
        double temp_min = std::min(h_x, h_y);
        double min_h = std::min(temp_min, h_z);
        tau = 0.5 * min_h / (std::sqrt(a2 * 3)); // Примерный расчет, может потребоваться корректировка
    }

    inline int index(const int& i, const int& j, const int& k) const {
        // Глобальный индекс для сетки (i, j, k от 0 до N включительно)
        return (i * (N + 1) + j) * (N + 1) + k;
    }
};

class Block {
  public:
    int N; // Общий размер padded сетки
    int local_Nx, local_Ny, local_Nz; // Размеры внутренней (локальной) сетки (без ghost)
    int padded_Nx, padded_Ny, padded_Nz; // Размеры сетки с ghost-слоями
    int x_start, y_start, z_start, x_end, y_end, z_end; // Индексы начала и конца локальной области во ВСЕЙ (глобальной) сетке (включая границы)
    int rank, dim0_rank, dim1_rank, dim2_rank;
    VINT neighbors; // 6 соседей + 2 для хранения ghost-размеров, если нужно (не используется напрямую)

    // Ghost слои для обмена
    VDOUB send_x_low, send_x_high, recv_x_low, recv_x_high;
    VDOUB send_y_low, send_y_high, recv_y_low, recv_y_high;
    VDOUB send_z_low, send_z_high, recv_z_low, recv_z_high;

    Block(const Grid& g, VINT& neighbors, int coords[], const int& dim0_n, const int& dim1_n, const int& dim2_n, int& rank) {
        this->rank = rank;
        this->dim0_rank = coords[0];
        this->dim1_rank = coords[1];
        this->dim2_rank = coords[2];
        this->neighbors = neighbors; // Копируем 6 соседей

        // --- Расчет локального размера и глобальных индексов ---
        // Используем тот же алгоритм разбиения, что и в оригинальном решении
        int base_size_x = (g.N + 1) / dim1_n;
        int remainder_x = (g.N + 1) % dim1_n;
        int base_size_y = (g.N + 1) / dim0_n;
        int remainder_y = (g.N + 1) % dim0_n;
        int base_size_z = (g.N + 1) / dim2_n;
        int remainder_z = (g.N + 1) % dim2_n;

        this->local_Nx = (this->dim1_rank < dim1_n - remainder_x) ? base_size_x : (base_size_x + 1);
        this->local_Ny = (this->dim0_rank < remainder_y) ? (base_size_y + 1) : base_size_y;
        this->local_Nz = (this->dim2_rank < remainder_z) ? (base_size_z + 1) : base_size_z;

        // Глобальные индексы начала и конца локальной области
        this->x_end = g.N; // начинаем с конца
        for (int i = dim1_rank; i > 0; --i) {
            this->x_end -= (i <= dim1_n - remainder_x) ? base_size_x : (base_size_x + 1);
        }
        this->x_start = this->x_end - this->local_Nx + 1;

        this->y_start = 0; // начинаем с начала
        for (int i = 0; i < dim0_rank; ++i) {
            this->y_start += (i < remainder_y) ? (base_size_y + 1) : base_size_y;
        }
        this->y_end = this->y_start + this->local_Ny - 1;

        this->z_start = 0; // начинаем с начала
        for (int i = 0; i < dim2_rank; ++i) {
            this->z_start += (i < remainder_z) ? (base_size_z + 1) : base_size_z;
        }
        this->z_end = this->z_start + this->local_Nz - 1;

        // --- Ghost слои ---
        this->padded_Nx = this->local_Nx + 2;
        this->padded_Ny = this->local_Ny + 2;
        this->padded_Nz = this->local_Nz + 2;
        this->N = this->padded_Nx * this->padded_Ny * this->padded_Nz; // Размер вектора с ghost-ячейками

        // --- Буферы для обмена ---
        // X-направление
        this->send_x_low.resize(this->local_Ny * this->local_Nz);
        this->send_x_high.resize(this->local_Ny * this->local_Nz);
        this->recv_x_low.resize(this->local_Ny * this->local_Nz);
        this->recv_x_high.resize(this->local_Ny * this->local_Nz);

        // Y-направление
        this->send_y_low.resize(this->local_Nx * this->local_Nz);
        this->send_y_high.resize(this->local_Nx * this->local_Nz);
        this->recv_y_low.resize(this->local_Nx * this->local_Nz);
        this->recv_y_high.resize(this->local_Nx * this->local_Nz);

        // Z-направление
        this->send_z_low.resize(this->local_Nx * this->local_Ny);
        this->send_z_high.resize(this->local_Nx * this->local_Ny);
        this->recv_z_low.resize(this->local_Nx * this->local_Ny);
        this->recv_z_high.resize(this->local_Nx * this->local_Ny);
    }

    inline int padded_index(const int& i_padded, const int& j_padded, const int& k_padded) const {
        // Индекс в локальном векторе с ghost-слоями
        // i_padded, j_padded, k_padded от 0 до padded_Nx/y/z-1
        // Соответствует локальным индексам от -1 до local_Nx/y/z
        if (i_padded < 0 || i_padded >= padded_Nx || j_padded < 0 || j_padded >= padded_Ny || k_padded < 0 || k_padded >= padded_Nz) {
            // std::cerr << "Warning: Accessing out-of-bounds padded index (" << i_padded << ", " << j_padded << ", " << k_padded << ") on rank " << rank << std::endl;
            // Возвращаем индекс на границе, например, 0,0,0 ghost
            return 0;
        }
        return i_padded * (padded_Ny * padded_Nz) + j_padded * padded_Nz + k_padded;
    }

    inline int local_index(const int& i_local, const int& j_local, const int& k_local) const {
        // Преобразует локальный индекс (от 0 до local_Nx/y/z-1) в индекс ghost-сетки (от 1 до local_Nx/y/z)
        // Используется для доступа к внутренним точкам в padded сетке
        return padded_index(i_local + 1, j_local + 1, k_local + 1);
    }

    inline int global_index(const int& i_global, const int& j_global, const int& k_global) const {
        // Используется для вычисления аналитического решения в глобальных координатах
        // Проверяем, принадлежит ли точка локальному блоку
        if (i_global >= x_start && i_global <= x_end && j_global >= y_start && j_global <= y_end && k_global >= z_start && k_global <= z_end) {
             // Преобразуем глобальный индекс в локальный индекс (от 0 до local_N-1)
             int local_i = i_global - x_start;
             int local_j = j_global - y_start;
             int local_k = k_global - z_start;
             // Преобразуем локальный индекс в индекс padded сетки
             return padded_index(local_i + 1, local_j + 1, local_k + 1);
        }
        // Если точка не принадлежит блоку, возвращаем -1 или бросаем исключение
        return -1;
    }

    void print_block_info() const {
        std::cout << "Block " << this->rank << " coord ("<< this->dim0_rank << "," << this->dim1_rank << "," << this->dim2_rank << ") info:\n\t" <<
            "local_Nx = " << this->local_Nx << "\n\t\t" << "x_start = " << this->x_start << "\n\t\t" << "x_end = " << this->x_end << "\n\t" <<
                "local_Ny = " << this->local_Ny << "\n\t\t" << "y_start = " << this->y_start << "\n\t\t" << "y_end = " << this->y_end << "\n\t" <<
                    "local_Nz = " << this->local_Nz << "\n\t\t" << "z_start = " << this->z_start << "\n\t\t" << "z_end = " << this->z_end << "\n\t" <<
                    "padded_Nx/y/z = " << padded_Nx << "/" << padded_Ny << "/" << padded_Nz << "\n\t" <<  std::endl;
    }
};

// u_analytical variant 3
inline double u_analytical(const Grid& g, const double& x, const double& y, const double& z, const double& t) {
    double at = (M_PI / 2.0) * std::sqrt(1.0 / (g.Lx * g.Lx) + 4.0 / (g.Ly * g.Ly) + 9.0 / (g.Lz * g.Lz));
    return std::sin(M_PI * x / g.Lx)
         * std::sin(2.0 * M_PI * y / g.Ly)
         * std::sin(3.0 * M_PI * z / g.Lz)
         * std::cos(at * t);
}

void solve_equation(const Grid& grid, Block& block, const int& dim0_n, const int& dim1_n, const int& dim2_n, MPI_Comm& comm_cart, double& time, double& max_inaccuracy, double& first_step_inaccuracy, double& last_step_inaccuracy, VDOUB& result);

#endif //EQUATION_H