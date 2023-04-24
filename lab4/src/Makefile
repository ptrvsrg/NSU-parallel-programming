np=1

build: mpi_jacobi.c
	mpicc -o mpi_jacobi mpi_jacobi.c -lm -Wall

run: build mpi_jacobi
	mpirun -np $(np) ./mpi_jacobi

clean:
	rm -rf mpi_jacobi