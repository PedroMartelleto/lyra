#!/bin/bash

# Check if there are any active jobs in the debug partition
debug_jobs=$(squeue -p debug -h | wc -l)

# lesser than 13 means there are available slots in debug partition
if [ "$debug_jobs" -lt 13 ]; then
    echo "Debug partition has enough nodes. Using debug..."
    srun --partition=debug --time=01:00:00 --environment=lyra -A a144 --pty bash
else
    echo "Debug partition is busy. Using normal..."
    srun --partition=normal --time=01:00:00 --environment=lyra -A a144 --pty bash
fi