#!/bin/sh

if [[ "$1" == "report" ]]; then
    python3 -m Performance.perf_testing_model
fi