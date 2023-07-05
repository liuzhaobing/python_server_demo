#!/bin/bash

export PYTHONPATH=`pwd`:$PYTHONPATH

mkdir -p runtime/logs/

CONSOLE_LOG="runtime/logs/console_`date +%Y%m%d`.log"

echo $CONSOLE_LOG

echo `python3 --version`

nohup python3 server.py >> $CONSOLE_LOG 2>&1 &