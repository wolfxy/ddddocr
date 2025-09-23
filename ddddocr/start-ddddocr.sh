#!/bin/sh
ps aux|grep python | grep 9782 | awk '{print $2}' | xargs kill -9
while (true)
do
       python -m ddddocr api --host 0.0.0.0 --port 9782 > ddddocr.log
       sleep 1
done