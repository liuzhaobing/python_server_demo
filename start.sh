#!/bin/bash

docker stop python_server_demo
docker rm python_server_demo
docker rmi python_server_demo:1.0
docker build -t python_server_demo:1.0 .
docker run -d -p 8091:8091 -p 50051:50051 python_server_demo:1.0 --name python_server_demo

