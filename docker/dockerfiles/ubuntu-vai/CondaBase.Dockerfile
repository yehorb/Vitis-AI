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

RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  bash ./install_base.sh ${DOCKER_TYPE} ${TARGET_FRAMEWORK}

USER vitis-ai-user
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  bash ./install_conda.sh
