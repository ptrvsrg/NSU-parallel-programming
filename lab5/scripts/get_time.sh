#!/bin/bash

if [ $# -ne 1 ]; then
	echo "USAGE: run.sh <MPI process count>"
	exit 1
fi

echo "MPI process count: $1"
cat cluster_$1.log | grep "Time"
cat cluster_$1.log | grep "Summary weight"
echo
