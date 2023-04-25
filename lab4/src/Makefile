np=1
version=1

build:
	mpicc -o mpi_jacobi_$(version) mpi_jacobi_$(version).c -lm -Wall

run: build
	mpirun -np $(np) ./mpi_jacobi_$(version)

clean:
	rm -rf mpi_jacobi_$(version)