#!/bin/bash
# Interactive GPU shell for smoke tests. One L40S for two hours.
exec salloc -p mit_normal_gpu -G 1 -c 8 --mem=32G --time=02:00:00
