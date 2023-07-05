#!/bin/bash

docker stop python_server_demo
docker rm python_server_demo
docker rmi python_server_demo:1.0
docker build -t python_server_demo:1.0 .
docker volume create huggingface_cache
docker run -d -p 50251:50251 -p 50252:50252 -v huggingface_cache:/opt/huggingface_cache --name python_server_demo python_server_demo:1.0
