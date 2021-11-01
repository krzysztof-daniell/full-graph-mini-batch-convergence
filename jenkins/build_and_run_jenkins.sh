#!/bin/bash

docker build -t dgl-sigopt-jenkins-image .
docker run \
  -d \
  -it \
  --volume /home/jenkins/volumes/dgl_sigopt_jenkins_certs:/certs/client \
  --volume /home/jenkins/volumes/dgl_sigopt_jenkins_home:/var/jenkins_home \
  --publish 8080:8080 \
  --publish 50000:50000 \
  --name dgl-sigopt-jenkins \
  dgl-sigopt-jenkins-image:latest
