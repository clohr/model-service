.PHONY:	clean install install-apoc start-neo4j-empty start-neo4j-ppmi load-ppmi run setup-indexes test test-container test-ci lint typecheck docs

LOG_LEVEL          ?= "INFO"
USE_CACHE          ?= "0"
NEO4J_APOC_VERSION ?= "3.5.0.13"
IMAGE_TAG          ?= "latest"
PORT               ?= 8080

QUOTED_NEXUS_USER := $(shell python3 -c "import urllib.parse; print(urllib.parse.quote('$(PENNSIEVE_NEXUS_USER)'))")
QUOTED_NEXUS_PW   := $(shell python3 -c "import urllib.parse; print(urllib.parse.quote('$(PENNSIEVE_NEXUS_PW)'))")

JWT = $(eval JWT := $(shell python bin/generate_jwt.py))$(JWT)

# JWT for the dataset in the `neo4j-ppmi` Docker image
PPMI_JWT = $(eval JWT := $(shell python bin/generate_jwt.py \
	--organization_node_id='N:organization:c905919f-56f5-43ae-9c2a-8d5d542c133b' \
	--dataset_node_id='N:dataset:4a2c9688-cb64-4b45-97a5-f10da00ff41f'))$(PPMI_JWT)

all: start-neo4j-empty

install-apoc:
	mkdir -p $(PWD)/plugins
	bin/install_neo4j_apoc.sh $(PWD)/plugins $(NEO4J_APOC_VERSION)

neo4j:
	docker-compose up -d neo4j

start-neo4j-empty: docker-clean install-apoc
	docker-compose --compatibility up neo4j

start-neo4j-ppmi: docker-clean install-apoc
	docker-compose up neo4j-ppmi

load-ppmi: clean install-apoc
	docker-compose build data-loader
	docker-compose down
	LOG_LEVEL=$(LOG_LEVEL) DATASET_NAME=ppmi USE_CACHE=$(USE_CACHE) docker-compose up --remove-orphans data-loader

docker-clean:
	docker-compose down

clean: docker-clean
	[ -d conf/ ] && rm -rf conf/* || return 0
	rm -rf data/*
	rm -f plugins/*
	rm -f generator/output/*
	$(MAKE) clean -C docs

install:
	pip install --upgrade pip
	pip install --upgrade --extra-index-url "https://$(QUOTED_NEXUS_USER):$(QUOTED_NEXUS_PW)@nexus.pennsieve.cc/repository/pypi-prod/simple" --pre -r requirements.txt -r requirements-dev.txt

setup-indexes:
	python -m server.db.index

test: typecheck format
	pytest -x -s -v tests

jwt:
	@echo $(JWT)

format:
	isort -rc .
	black .

typecheck:
	@mypy -p server -p publish

lint:
	@isort -rc --check-only .
	@black --check .

docs:
	 $(MAKE) -C docs

test-container:
	@IMAGE_TAG=$(IMAGE_TAG) docker-compose build model-service-test

test-ci: install-apoc
	docker-compose down --remove-orphans
	mkdir -p data plugins conf logs
	chmod -R 777 conf
	@IMAGE_TAG=$(IMAGE_TAG) docker-compose up --exit-code-from=model-service-test model-service-test

lint-ci:
	docker-compose down --remove-orphans
	@IMAGE_TAG=$(IMAGE_TAG) docker-compose up --exit-code-from=model-service-lint model-service-lint

containers:
	@IMAGE_TAG=$(IMAGE_TAG) docker-compose build model-service model-publish

run:
	python main.py --port=$(PORT)

run-service-container: containers
	docker run -it -e ENVIRONMENT=dev -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_SESSION_TOKEN -p "$(PORT):$(PORT)" pennsieve/model-service:latest

run-profiler:
	sudo py-spy --flame flame.svg --nonblocking --duration 90 -- python main.py localhost $(PORT)

run-upload-test:
	@JWT=$(JWT) ./load-tests/artillery run load-tests/upload.yml

# Launch the profiler in the background, then start the load test
# This must be run as root, eg `sudo make profile`
profile-upload:
	if (( $EUID != 0 )); then echo "Please run as root"; exit 1; fi
	$(MAKE) run-profiler& sleep 5 && $(MAKE) run-upload-test

setup-download:
	python -m generator.seed -m 1 -r 100000 --out download-seed.json
	python -m generator.load download-seed.json

run-download-test:
	@if [[ -z "${MODEL_ID}" ]]; then echo "Set MODEL_ID variable with target model id"; exit 1; fi
	@JWT=$(JWT) ./load-tests/artillery run load-tests/download.yml

profile-download:
	@if (( $EUID != 0 )); then echo "Please run as root"; exit 1; fi
	@if [[ -z "${MODEL_ID}" ]]; then echo "Set MODEL_ID variable with target model id"; exit 1; fi
	$(MAKE) run-profiler& sleep 5 && $(MAKE) run-download-test


SSH_KEY := "~/.ssh/neptune-export.pem"
NEPTUNE_EXPORT_HOST := "ubuntu@172.31.14.140"

build-neptune-export:
	cd ../pennsieve-api && sbt neptune-export/assembly

deploy-neptune-export:
	ssh -i $(SSH_KEY) $(NEPTUNE_EXPORT_HOST) 'mkdir -p ~/model-service'
	rsync -rave "ssh -i $(SSH_KEY)"  \
		--exclude="__pycache__/" \
		--exclude="*.pyc" \
		--exclude="*.csv" \
		--exclude=".git/" \
		--exclude=".mypy_cache/" \
		--exclude="node_modules/*" \
		--exclude="datasets/"  \
		--exclude="data/" \
		--exclude=".pytest_cache/" \
		./ $(NEPTUNE_EXPORT_HOST):/home/ubuntu/model-service/

	ssh -i $(SSH_KEY) $(NEPTUNE_EXPORT_HOST) 'mkdir -p ~/pennsieve-api/neptune-export/target/scala-2.12/'
	rsync -rave "ssh -i $(SSH_KEY)" \
		--exclude="classes/" \
		--exclude="test-classes/" \
		../pennsieve-api/neptune-export/target/scala-2.12/ \
		$(NEPTUNE_EXPORT_HOST):/home/ubuntu/pennsieve-api/neptune-export/target/scala-2.12/

	ssh -i $(SSH_KEY) $(NEPTUNE_EXPORT_HOST) 'mkdir -p ~/.local/lib/python3.6/site-packages/auth_middleware/'
	rsync -rave "ssh -i $(SSH_KEY)" \
		--exclude="__pycache__/" \
		--exclude="*.pyc" \
		~/.pyenv/versions/neo4j-poc/lib/python3.7/site-packages/auth_middleware/ \
		$(NEPTUNE_EXPORT_HOST):/home/ubuntu/.local/lib/python3.6/site-packages/auth_middleware/

setup-import-tests:
	aws s3 cp tests/data/ s3://dev-neptune-export-use1/ --recursive \
		--exclude=".pytest_cache/*" \
        --exclude="parsed/*"
