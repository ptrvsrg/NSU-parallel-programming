np=1

all: clean run clean

build: mpi_multiply.c
	mpicc -o mpi_multiply mpi_multiply.c -lm -Wall

run: build mpi_multiply
	mpirun -np $(np) ./mpi_multiply

clean:
	rm -rf mpi_multiply