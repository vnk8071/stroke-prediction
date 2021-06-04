#!/bin/bash

#mkdir data
python ./src/get-data.py
unzip ./data/stroke-data.zip -d ./data