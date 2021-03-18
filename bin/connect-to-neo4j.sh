#!/usr/bin/env sh

set  -e

function print_usage_and_exit() {
  echo "Usage: $0 -e|--environment=ENVIRONMENT -j|--jumpbox=JUMPBOX"
  exit 1
}

if [ -z "$AWS_SESSION_TOKEN" ]; then
  echo "Missing AWS session token. Call assume-role first"
  exit 1
fi

ENVIRONMENT=
JUMPBOX=

while (( "$#" )); do
  case "$1" in
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

if [ -z "$ENVIRONMENT" ]; then
  echo "Missing ENVIRONMENT argument"
  print_usage_and_exit
fi

if [ -z "$JUMPBOX" ]; then
  echo "Missing JUMPBOX argument"
  print_usage_and_exit
fi

get_param() {
  aws ssm get-parameter --with-decryption --output=text --query Parameter.Value --name $1
}

echo "Getting SSM parameters..."

neo4j_url=$(get_param "/$ENVIRONMENT/model-service/neo4j-bolt-url")
neo4j_user=$(get_param "/$ENVIRONMENT/model-service/neo4j-bolt-user")
neo4j_password=$(get_param "/$ENVIRONMENT/model-service/neo4j-bolt-password")

neo4j_host_and_port=$(python -c "import urllib.parse; print(urllib.parse.urlparse('$neo4j_url').netloc)")

local_neo4j_port=$(expr $RANDOM + 1000)

echo "Opening SSH tunnel for Neo4j to $JUMPBOX..."
ssh -f -o ExitOnForwardFailure=yes -L "$local_neo4j_port:$neo4j_host_and_port" $JUMPBOX sleep 60

export NEO4J_BOLT_URL="bolt://localhost:$local_neo4j_port"
export NEO4J_BOLT_USER="$neo4j_user"
export NEO4J_BOLT_PASSWORD="$neo4j_password"


cypher-shell -a $NEO4J_BOLT_URL -u $NEO4J_BOLT_USER -p $NEO4J_BOLT_PASSWORD
