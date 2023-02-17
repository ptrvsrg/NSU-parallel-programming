n=1

all: run clean

build: mpi_slae_$(version).c
	mpicc -o mpi_slae_$(version) mpi_slae_$(version).c -lm -Wall

run: build mpi_slae_$(version)
	mpiexec -n $(n) ./mpi_slae_$(version)

clean:
	rm -rf mpi_slae_$(version)