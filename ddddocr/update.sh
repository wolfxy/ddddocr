#!/bin/sh
wget "$1" -O ddddocr.zip
rm -fr ddddocr
unzip ddddocr.zip
ps aux | grep python | grep 9782 | awk '{print $2}' | xargs kill -9