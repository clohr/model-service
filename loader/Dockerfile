FROM pennsieve/python-cloudwrap:3.7-alpine-0.5.9

USER root

RUN apk add bash

USER pennsieve

WORKDIR /app

COPY --chown=pennsieve:pennsieve bin/wait-for-it.sh        ./wait-for-it.sh
COPY --chown=pennsieve:pennsieve loader/requirements.txt   ./requirements.txt
COPY --chown=pennsieve:pennsieve loader/import_to_neo4j.py ./import_to_neo4j.py

ENV PATH "/home/pennsieve/.local/bin:$PATH"

RUN pip install --user --upgrade pip && \
  pip install --user --no-cache-dir -r /app/requirements.txt

ENTRYPOINT ["python", "./import_to_neo4j.py", "--help"]
