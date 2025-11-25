CFLAGS = -O3 -Wall -Wextra -std=c++11
MPICXX = OMPI_CXX=g++ mpicxx
INCLUDES = -I/opt/ibm/spectrum_mpi/include
LIBS = -L/opt/ibm/spectrum_mpi/lib -lmpiprofilesupport -lmpi_ibm

TARGET = wave3d_mpi
SOURCES = main_mpi.cpp solution_mpi.cpp

compile_polus:
	$(MPICXX) $(CFLAGS) $(INCLUDES) $(SOURCES) -o $(TARGET) $(LIBS)