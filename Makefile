CFLAGS = -O3 -Wall -Wextra -std=c++11
G++ = OMPI_CXX=g++ mpicxx -I/opt/ibm/spectrum_mpi/include
LIBS = -L/opt/ibm/spectrum_mpi/lib -lmpiprofilesupport -lmpi_ibm

TARGET = wave3d_mpi
SOURCES = main_mpi.cpp solution_mpi.cpp

compile_polus: $(G++) $(CFLAGS) $(SOURCES) -o $(TARGET) $(LIBS)
