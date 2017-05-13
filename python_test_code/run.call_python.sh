#!/bin/bash

source test.inc

echo | python run_submitted_code.py ${1} > $logdir/run.call_python.${1}.out 2>&1
