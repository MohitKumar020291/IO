#!/bin/sh

source .env

if [ "$environment" == "development" ]; 
then
    uvicorn app:app --host localhost --port 8000 --reload
fi