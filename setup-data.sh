#!/bin/bash

mkdir data
python ./src/get-data.py
unzip ./data/data.zip -d ./data