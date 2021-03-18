###############################################################################
# TARGET: build
###############################################################################
FROM python:3.7 as build
WORKDIR app

ARG PENNSIEVE_NEXUS_USER
ARG PENNSIEVE_NEXUS_PW

COPY bin/wait-for-it.sh ./wait-for-it.sh
COPY Makefile requirements.txt requirements-dev.txt setup.cfg ./

RUN make install

COPY conftest.py ./conftest.py
COPY openapi     ./openapi
COPY publish     ./publish
COPY generator   ./generator
COPY migrate     ./migrate
COPY utils       ./utils
COPY loader      ./loader
COPY core        ./core
COPY server      ./server
COPY tests       ./tests
