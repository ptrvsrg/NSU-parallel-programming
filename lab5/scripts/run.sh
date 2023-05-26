#!/bin/bash

if [ $# -ne 1 ]; then
	echo "USAGE: run.sh <MPI process count>"
	exit 1
fi

mpiexec -np="$1" ../build/cluster