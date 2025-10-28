CFLAGS = -O3 -Wall -Wextra -std=c++11 -fopenmp
CC_POL = g++

TARGET = wave3d
SOURCES = main.cpp solution.cpp

compile_polus: create_result_dir
	$(CC_POL) $(CFLAGS) $(SOURCES) -o $(TARGET)

create_result_dir:
	mkdir -p results/statistics
	mkdir -p results/grid

clean_results:
	rm -rf results/*

clean:
	rm -f $(TARGET)

clean_all: clean
	rm -rf results/