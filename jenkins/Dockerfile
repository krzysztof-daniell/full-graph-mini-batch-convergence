FROM jenkins/jenkins:latest

USER root

RUN apt-get update && apt-get install -y \
    ca-certificates curl gnupg lsb-release \
  && curl -fsSL https://download.docker.com/linux/debian/gpg | \
    gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg \
  && echo "deb [arch=$(dpkg --print-architecture) \ 
    signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] \
    https://download.docker.com/linux/debian \
    $(lsb_release -cs) stable" | \
    tee /etc/apt/sources.list.d/docker.list > /dev/null \
  && apt-get update && apt-get install -y docker-ce-cli \
  && groupadd docker && usermod -aG docker jenkins

ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}
ENV no_proxy=${no_proxy}
