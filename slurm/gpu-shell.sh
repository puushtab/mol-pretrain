#!/bin/bash
# Interactive GPU shell for smoke tests (1 L40S, 4 cores, 16 GB, 2 h).
exec salloc -p mit_normal_gpu -G 1 -c 4 --mem=16G --time=02:00:00
