
* Fork https://github.com/aws-samples/eks_gpu_and_trainuim_perceiver_io_training/ and populate the `GITHUB_USER`.
* Export the following variables

```bash
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --output text --query Account)
export AWS_REGION=us-west-2
export BASE_REPO=perceiver
export BASE_TAG=multiarch-ubuntu
export BASE_ARM_TAG=arm64
export BASE_AMD_TAG=amd64
export GITHUB_BRANCH=master
export GITHUB_USER=yahavb
export GITHUB_REPO=eks_gpu_and_trainuim_perceiver_io_training
```

```bash
cd ci-build
./deploy-pipeline.sh
```
