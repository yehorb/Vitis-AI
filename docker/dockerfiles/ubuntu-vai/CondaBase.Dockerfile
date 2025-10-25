ARG VAI_BASE=ubuntu:18.04
FROM $VAI_BASE

ARG DOCKER_TYPE
ARG TARGET_FRAMEWORK

SHELL ["/bin/bash", "-c"]

ENV TZ=America/Denver
ENV VAI_ROOT=/opt/vitis_ai
ENV VAI_HOME=/vitis_ai_home
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace
ADD ./common/ .
RUN bash ./install_base.sh ${DOCKER_TYPE} ${TARGET_FRAMEWORK}

USER vitis-ai-user
RUN bash ./install_conda.sh
