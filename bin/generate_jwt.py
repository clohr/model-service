#!/usr/bin/env python

import os

if __name__ == "__main__":

    import argparse

    from auth_middleware import Claim, JwtConfig, UserClaim
    from auth_middleware.models import RoleType
    from auth_middleware.role import (
        DatasetId,
        DatasetRole,
        OrganizationId,
        OrganizationRole,
    )

    parser = argparse.ArgumentParser(description="Generate a JWT")
    parser.add_argument(
        "--dataset_node_id", type=str, default="N:dataset:A-B", required=False
    )
    parser.add_argument("--dataset_id", type=int, default=1, required=False)
    parser.add_argument(
        "--organization_node_id", type=str, default="N:organization:1", required=False
    )
    parser.add_argument("--organization_id", type=int, default=1, required=False)
    parser.add_argument("--user_id", type=int, default=12345, required=False)
    parser.add_argument(
        "--user_node_id", type=str, default="N:user:U-S-E-R", required=False
    )
    parser.add_argument(
        "--jwt_key",
        type=str,
        default=os.environ.get("JWT_SECRET_KEY", "test-key"),
        required=False,
    )

    args = parser.parse_args()

    claim = Claim.from_claim_type(
        UserClaim(
            id=args.user_id,
            node_id=args.user_node_id,
            roles=[
                OrganizationRole(
                    id=OrganizationId(args.organization_id),
                    node_id=args.organization_node_id,
                    role=RoleType.OWNER,
                ),
                DatasetRole(
                    id=DatasetId(args.dataset_id),
                    node_id=args.dataset_node_id,
                    role=RoleType.OWNER,
                ),
            ],
        ),
        60 * 60,
    )
    token = claim.encode(JwtConfig(args.jwt_key))

    print(token.decode())  # convert bytes -> string
