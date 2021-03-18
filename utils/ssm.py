import urllib.parse

import boto3


class SSMParameters:
    """
    Lifted from base_processor and timeseries_processor
    """

    def __init__(self, env="dev", aws_region="us-east-1"):
        self.aws_region = aws_region

        self._load_values(
            postgres_db=f"{env}-pennsieve-postgres-db",
            postgres_host=f"{env}-pennsieve-postgres-host",
            postgres_port=f"{env}-pennsieve-postgres-port",
            postgres_user=f"{env}-pennsieve-postgres-user",
            postgres_password=f"{env}-pennsieve-postgres-password",
        )

        self._load_values(
            neo4j_url=f"/{env}/model-service/neo4j-bolt-url",
            neo4j_user=f"/{env}/model-service/neo4j-bolt-user",
            neo4j_password=f"/{env}/model-service/neo4j-bolt-password",
            # export_bucket=f"/{env}/neptune-export/s3-export-bucket",
            # neptune_host=f"/{env}/neptune-export/neptune-host",
            # redis_host=f"/{env}/neptune-export/redis-host",
        )

        self._load_values(
            api_host=f"/{env}/model-service/pennsieve-api-host",
            jwt_secret_key=f"/{env}/model-service/jwt-secret-key",
        )

        self.neo4j_host, self.neo4j_port = urllib.parse.urlparse(
            self.neo4j_url
        ).netloc.split(":")

        self.neptune_port = 8182
        self.redis_port = 6379

    def _load_values(self, **keys):
        """
        Load values from SSM for all keys. Keys should be in the
        format key=ssm_variable. This method injects the value of
        each key into class variables, such that self.key = <ssm-value>

        For example, if:

        keys = {
            'key1': 'ssm-field1',
            'key2:  'ssm-field2'
        }

        It results in:

        self.key1 = ssm-field1-value
        self.key2 = ssm-field2-value

        """
        ssm_client = boto3.client("ssm", region_name=self.aws_region)
        response = ssm_client.get_parameters(
            Names=list(keys.values()), WithDecryption=True
        )

        if len(response["InvalidParameters"]) > 0:
            raise Exception(
                "Invalid SSM parameters: {}".format(response["InvalidParameters"])
            )

        for i, resp in enumerate(response["Parameters"]):
            var = list(keys.keys())[list(keys.values()).index(resp["Name"])]
            value = resp["Value"]
            if str(var) in self.__dict__:
                raise Exception("Settings variable already exists: {}".format(var))
            self.__dict__[var] = value
