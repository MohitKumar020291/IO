#!/bin/sh


if [[ "$1" == "ptm" ]]; then
    source ./Performance/run.sh ${@:2}
fi