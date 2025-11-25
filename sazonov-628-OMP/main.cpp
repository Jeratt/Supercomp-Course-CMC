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

    return 0;
}