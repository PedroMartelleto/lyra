#!/bin/bash

# Check if there are any active jobs in the debug partition
debug_jobs=$(squeue -p debug -h | wc -l)

if [ "$debug_jobs" -eq 0 ]; then
    echo "Debug partition is empty. Using debug..."
    srun --partition=debug --time=01:00:00 --environment=lyra -A a144 --pty bash
else
    echo "Debug partition is busy. Using normal..."
    srun --partition=normal --time=00:30:00 --environment=lyra -A a144 --pty bash
fi