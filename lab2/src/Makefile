np=1
version=1
schedule=static

build: openmp_slae_$(version).c
	gcc -fopenmp -o openmp_slae_$(version) openmp_slae_$(version).c -lm -Wall

run: build openmp_slae_$(version)
	OMP_NUM_THREADS=$(np) OMP_SCHEDULE=$(schedule) ./openmp_slae_$(version)

clean:
	rm -rf openmp_slae_$(version)