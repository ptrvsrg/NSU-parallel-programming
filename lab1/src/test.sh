#!/bin/bash

echo "Testing version 1 for 1 process:"
make version=1 np=1

echo "Testing version 1 for 16 process:"
make version=1 np=16

echo "Testing version 2 for 1 process:"
make version=2 np=1

echo "Testing version 2 for 16 process:"
make version=2 np=16