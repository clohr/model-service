#!/usr/bin/env sh

set  -e

function print_usage_and_exit() {
  echo "Usage: $0 -e|--environment=ENVIRONMENT -j|--jumpbox=JUMPBOX -d|--dataset=DATASET "
  exit 1
}

PARAMS=""
DATASET=
ENVIRONMENT=
IS_LOCAL=
JUMPBOX=
CUTOVER=0
REMOVE_EXISTING=0
PRODUCE_STATISTICS=0
RUN_SMOKE_TEST=1
USE_CACHE=0
USE_S3_BUCKET=
REMAP_IDS=0

function check_and_set_env() {
  if [ -z "$ENVIRONMENT" ]; then
    print_usage_and_exit
  fi
  case "$ENVIRONMENT" in
    local)
      ENVIRONMENT=local
      IS_LOCAL=1
      ;;
    dev|development)
      ENVIRONMENT=dev
      ;;
    prod|production)
      ENVIRONMENT=prod
      ;;
    *)
      echo "Bad environment: $ENVIRONMENT (use \"local\", \"dev\", \"development\", \"prod\", \"production\")"
      exit 1
      ;;
   esac
}

while (( "$#" )); do
  case "$1" in
    -d|--dataset)
      DATASET=$2
      shift # past argument
      shift # past value
      ;;

    -e|--environment)
      ENVIRONMENT=$2
      shift # past argument
      shift # past value
      ;;

    -j|--jumpbox)
      JUMPBOX=$2
      shift # past argument
      shift # past value
      ;;

    -c|--cutover)
      CUTOVER=1
      shift # past argument
      ;;

    -b|--bucket)
      USE_S3_BUCKET=$2
      shift # past argument
      shift # past value
      ;;

    -r|--remove-existing)
      REMOVE_EXISTING=1
      shift # past argument
      ;;

    -u|--use-cache)
      USE_CACHE=1
      shift # past argument
      ;;

    --statistics)
      PRODUCE_STATISTICS=1
      shift # past argument
      ;;

    --skip-smoke-test)
      RUN_SMOKE_TEST=0
      shift # past argument
      ;;

    --remap-ids)
      REMAP_IDS=1
      shift # past argument
      ;;

    --) # end argument parsing
      shift
      break
      ;;

    -*|--*=) # unsupported flags
      print_usage_and_exit
      ;;

    *) # preserve positional arguments
      PARAMS="$PARAMS $1"
      shift
      ;;
  esac
done

# set positional arguments in their proper place
eval set -- "$PARAMS"

if [ -z "$AWS_SESSION_TOKEN" ]; then
  echo "Missing AWS session token. Call assume-role first"
  exit 1
fi

if [ -z "$DATASET" ]; then
  print_usage_and_exit
fi

check_and_set_env

if [ -z "$IS_LOCAL" ] && [ -z "$JUMPBOX" ]; then
  print_usage_and_exit
fi

get_param() {
  aws ssm get-parameter --with-decryption --output=text --query Parameter.Value --name $1
}

if [ -z "$IS_LOCAL" ]; then
  echo "Getting SSM parameters..."

  neo4j_url=$(get_param "/$ENVIRONMENT/model-service/neo4j-bolt-url")
  neo4j_user=$(get_param "/$ENVIRONMENT/model-service/neo4j-bolt-user")
  neo4j_password=$(get_param "/$ENVIRONMENT/model-service/neo4j-bolt-password")
  local_neo4j_port=$(expr $RANDOM + 1000)
else
  echo "Importing locally"

  neo4j_url="bolt://localhost:7687"
  neo4j_user="neo4j"
  neo4j_password="blackandwhite"
  local_neo4j_port=7687
fi

if [ -z "$USE_S3_BUCKET" ]; then
  if [ "$ENVIRONMENT" = "local" ]; then
    S3_ENV="dev"
  else
    S3_ENV="$ENVIRONMENT"
  fi
  export_bucket=$(get_param "/$S3_ENV/neptune-export/s3-export-bucket")
else
  export_bucket="$USE_S3_BUCKET"
fi

neo4j_host_and_port=$(python -c "import urllib.parse; print(urllib.parse.urlparse('$neo4j_url').netloc)")

echo "neo4j_url = $neo4j_url"
echo "neo4j_user = $neo4j_user"
echo "neo4j_host_and_port = $neo4j_host_and_port"
echo "local_neo4j_port = $local_neo4j_port"
echo "export_bucket = $export_bucket"

if [ -z "$IS_LOCAL" ]; then
  echo "Opening SSH tunnel for Neo4j to $JUMPBOX..."
  ssh -f -o ExitOnForwardFailure=yes -L "$local_neo4j_port:$neo4j_host_and_port" $JUMPBOX sleep 600
fi

export NEO4J_BOLT_URL="bolt://localhost:$local_neo4j_port"
export NEO4J_BOLT_USER="$neo4j_user"
export NEO4J_BOLT_PASSWORD="$neo4j_password"
python -m loader --bucket "$export_bucket" --dataset "$DATASET" --cutover "$CUTOVER" --remove-existing "$REMOVE_EXISTING" --use-cache "$USE_CACHE" --statistics "$PRODUCE_STATISTICS" --smoke-test "$RUN_SMOKE_TEST" --remap-ids "$REMAP_IDS"
