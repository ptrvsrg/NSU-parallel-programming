n=1

all: run clean

build: openmp_slae_$(version).c
	gcc -fopenmp -o openmp_slae_$(version) openmp_slae_$(version).c -lm -Wall

run: build openmp_slae_$(version)
	OMP_NUM_THREADS=$(n) ./openmp_slae_$(version)

clean:
	rm -rf openmp_slae_$(version)