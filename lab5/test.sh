#!/bin/bash

./build.sh

for (( i = 1; i <= 12; ++i ))
do
	echo "MPI process count: $i"
	./run.sh "$i" > logs/cluster_"$i".log
	echo "Create logs/cluster_$i.log"
done
