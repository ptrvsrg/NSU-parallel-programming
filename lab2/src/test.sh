#!/bin/bash

echo "Testing version 1 for 1 process:"
make version=1 np=1

echo "Testing version 1 for 16 process:"
make version=1 np=16

echo "Testing version 2 for 1 process:"
make version=2 np=1

echo "Testing version 2 for 16 process:"
make version=2 np=16

echo "Testing version 3 for 16 process with default static schedule:"
make version=2 np=16 schedule=static

echo "Testing version 3 for 16 process with custom static schedule:"
make version=2 np=16 schedule=static,200

echo "Testing version 3 for 16 process with default dynamic schedule:"
make version=2 np=16 schedule=dynamic

echo "Testing version 3 for 16 process with custom dynamic schedule:"
make version=2 np=16 schedule=dynamic,200

echo "Testing version 3 for 16 process with default guided schedule:"
make version=2 np=16 schedule=guided

echo "Testing version 3 for 16 process with custom guided schedule:"
make version=2 np=16 schedule=guided,200