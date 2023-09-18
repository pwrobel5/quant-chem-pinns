#!/bin/bash

for i in 20 40 80
do
  python3 function.py -n 2 -l 0 -nnodes $i
  python3 function.py -n 4 -l 1 -nnodes $i
  python3 function.py -n 5 -l 4 -nnodes $i
  python3 pde.py -n 2 -l 0 -nnodes $i
  python3 pde.py -n 4 -l 1 -nnodes $i
  python3 pde.py -n 5 -l 4 -nnodes $i
done
