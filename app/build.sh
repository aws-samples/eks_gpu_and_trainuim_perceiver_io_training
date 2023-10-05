#!/bin/bash -x
#export AWS_ACCOUNT_ID="133573815653"
#export AWS_REGION="us-west-2"
#export BASE_REPO="perceiver"
#export BASE_IMAGE_TAG="base"

BASE_IMAGE=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$BASE_REPO:$BASE_IMAGE_TAG
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com
docker build -t $BASE_IMAGE .
docker logout 763104351884.dkr.ecr.us-east-1.amazonaws.com
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $BASE_IMAGE
docker push $BASE_IMAGE
