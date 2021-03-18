###############################################################################
# TARGET: build
###############################################################################

FROM python:3.7-alpine as build

ARG PENNSIEVE_NEXUS_USER
ARG PENNSIEVE_NEXUS_PW

RUN apk update && apk add build-base

WORKDIR app

RUN echo 'manylinux1_compatible = True' > /usr/local/lib/python3.7/site-packages/_manylinux.py
COPY requirements.txt ./
RUN pip install -r requirements.txt --user --extra-index-url "https://$PENNSIEVE_NEXUS_USER:$PENNSIEVE_NEXUS_PW@nexus.pennsieve.cc/repository/pypi-prod/simple"

COPY main.py  ./main.py
COPY core/    ./core/
COPY server/  ./server/
COPY openapi/ ./openapi/

###############################################################################
# TARGET: service
###############################################################################

FROM pennsieve/python-cloudwrap:3.7-alpine-0.5.9 as service

COPY --chown=pennsieve:pennsieve --from=build app/ app/
COPY --chown=pennsieve:pennsieve --from=build /root/.local /home/pennsieve/.local

WORKDIR app

ENV PATH "/home/pennsieve/.local/bin:$PATH"

CMD ["--service", "model-service", "exec", "newrelic-admin", "run-python", "main.py", "--host", "0.0.0.0", "--port", "8080", "--threads", "4"]

###############################################################################
# TARGET: publish
###############################################################################

FROM pennsieve/python-cloudwrap:3.7-alpine-0.5.9 as publish

COPY --chown=pennsieve:pennsieve --from=build app/ app/
COPY --chown=pennsieve:pennsieve --from=build /root/.local /home/pennsieve/.local

WORKDIR app

COPY --chown=pennsieve:pennsieve publish/ ./publish/

ENV PATH "/home/pennsieve/.local/bin:$PATH"

CMD ["--service", "model-publish", "exec", "python", "-m", "publish"]
