np=1

build: mpi_multiply.c
	mpicc -o mpi_multiply mpi_multiply.c -lm -Wall

run_default_grid: build mpi_multiply
	mpirun -np $(np) ./mpi_multiply

run_custom_grid: build mpi_multiply
	mpirun -np $(np) ./mpi_multiply $(dims_x) $(dims_y)

clean:
	rm -rf mpi_multiply