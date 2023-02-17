n=1

all: run

build: $(name).c
	mpicc -o $(name) $(name).c -lm -Wall

run: build $(name)
	mpiexec -n $(n) ./$(name)
	rm -rf $(name)