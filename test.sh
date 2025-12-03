#!/bin/sh

python3 -m unittest test.Test

if [ -f "test_embeddings.pkl" ]; then
    rm "test_embeddings.pkl"
fi