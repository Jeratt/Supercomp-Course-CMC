#define _USE_MATH_DEFINES
#include "equation.hpp"
#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <cmath>

using namespace std;

double parse_length(const char* arg, string& label_part) {
    if (strcmp(arg, "pi") == 0)
    {
        label_part = "pi";
        return M_PI;
    }
    else
    {
        label_part = string(arg);
        return atof(arg);
    }
}

void save_results(const Grid& g, const VDOUB& u, const string& filename, const string& data_type) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error with open file for " <<  data_type << " save"  << endl;
        return;
    }
    for (int i = 0; i < g.N + 1; ++i) {
        for (int j = 0; j < g.N + 1; ++j) {
            for (int k = 0; k < g.N + 1; ++k) {
                if (data_type == "nums")
                {
                    file << u[g.index(i, j, k)];
                } else if (data_type == "anal")
                {
                    file << u_analytical(g, i * g.h_x, j * g.h_y, k * g.h_z, (T - 1) * g.tau);
                } else if (data_type == "inacc")
                {
                    file << fabs(u[g.index(i, j, k)] - u_analytical(g, i * g.h_x, j * g.h_y, k * g.h_z, (T - 1) * g.tau));
                } else {
                    cerr << "Unknown matrix type: " << data_type << endl;
                }
                if (k < g.N) file << ",";
            }
            file << "\n";
        }
        file << "\n";
    }
    file.close();
}

void save_stats(const Grid& g, double time, double max_inaccuracy,
                     double first_step_inaccuracy, double last_step_inaccuracy, int threads_num) {
    string filename = "results/statistics/" + to_string(g.N) + "_" +
                           to_string(threads_num) + "_" + g.domain_label + "_statistics.txt";
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Failed to open file '" << filename << "' for writing." << endl;
        return;
    }
    file << "Time = " << time << "\n"
         << "Max inaccuracy = " << max_inaccuracy << "\n"
         << "First step inaccuracy = " << first_step_inaccuracy << "\n"
         << "Last step inaccuracy = " << last_step_inaccuracy << endl;
}

int main(int argc, char* argv[]) {
    if (argc != 6) {
        cerr << "Usage: " << argv[0] << " N THREADS Lx Ly Lz\n"
                  << "  Lx, Ly, Lz: numbers (e.g. 1.0) or 'pi'\n";
        return 1;
    }

    int N = atoi(argv[1]);
    int threads_num = atoi(argv[2]);
    if (N <= 0 || threads_num <= 0) {
        cerr << "N and THREADS must be positive integers.\n";
        return 1;
    }

    string lx_label, ly_label, lz_label;
    double Lx = parse_length(argv[3], lx_label);
    double Ly = parse_length(argv[4], ly_label);
    double Lz = parse_length(argv[5], lz_label);
    string domain_label = lx_label + "_" + ly_label + "_" + lz_label;

    Grid grid(N, Lx, Ly, Lz, domain_label);

    cout << "Input values:\n"
              << "\tN = " << grid.N << "\n"
              << "\tThreads = " << threads_num << "\n"
              << "\tLx = " << grid.Lx << "\n"
              << "\tLy = " << grid.Ly << "\n"
              << "\tLz = " << grid.Lz << "\n"
              << "\tDomain label = " << grid.domain_label << endl;

    VDOUB result_vec;
    double time = 0, max_inaccuracy = -1, first_step_inaccuracy = 0, last_step_inaccuracy = 0;

    solve(grid, time, max_inaccuracy, first_step_inaccuracy, last_step_inaccuracy, result_vec, threads_num);

    cout << "Result:\n"
              << "\tTime = " << time << "\n"
              << "\tMax inaccuracy = " << max_inaccuracy << "\n"
              << "\tFirst step inaccuracy = " << first_step_inaccuracy << "\n"
              << "\tLast step inaccuracy = " << last_step_inaccuracy << endl;

    save_stats(grid, time, max_inaccuracy, first_step_inaccuracy, last_step_inaccuracy, threads_num);
    save_results(grid, result_vec, "results/grid/" + to_string(N) + "_" +
                     to_string(threads_num) + "_" + grid.domain_label + "_nums.csv", "nums");
    save_results(grid, result_vec, "results/grid/" + to_string(N) + "_" +
                     to_string(threads_num) + "_" + grid.domain_label + "_anal.csv", "anal");
    save_results(grid, result_vec, "results/grid/" + to_string(N) + "_" +
                     to_string(threads_num) + "_" + grid.domain_label + "_inacc.csv", "inacc");

    return 0;
}