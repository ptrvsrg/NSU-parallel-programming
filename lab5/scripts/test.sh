#!/bin/bash

echo "Building..."
./build.sh

for (( i = 1; i <= 16; ++i ))
do
	echo "Running for $i MPI processes..."
	./run.sh "$i" > ../logs/cluster_"$i".log
done
