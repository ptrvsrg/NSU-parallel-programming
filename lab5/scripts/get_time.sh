#!/bin/bash

for (( i = 1; i <= 12; ++i ))
do
	echo "MPI process count: $i"
	cat cluster_$i.log | grep "Time"
	cat cluster_$i.log | grep "Summary weight"
	echo
done
