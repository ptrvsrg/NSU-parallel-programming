n=1

all: run clean

build: $(name).c
	gcc -o $(name) $(name).c -lm -lrt -Wall

run: build $(name)
	./$(name)

clean:
	rm -rf $(name)