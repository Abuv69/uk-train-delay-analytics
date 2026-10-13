#!/bin/bash
# lambda/deploy_lambda.sh
# Packages the Lambda function and deploys it to AWS.
# Run from the project root: bash lambda/deploy_lambda.sh
#
# Prerequisites:
#   - AWS CLI installed: https://aws.amazon.com/cli/
#   - AWS credentials configured (~/.aws/credentials)
#   - pip install boto3 pandas openpyxl odfpy (for the layer)

set -e

FUNCTION_NAME="uk-train-delay-pipeline"
REGION="${AWS_REGION:-eu-west-2}"   # must match your bucket region
BUCKET="${S3_BUCKET:-uk-train-delay-abu}"   # export S3_BUCKET=your-bucket-name
RUNTIME="python3.12"
ROLE_NAME="lambda-train-delay-role"

echo "======================================================"
echo " UK Train Delay Analytics — Lambda Deploy"
echo "======================================================"

# ── Step 1: Create IAM role for Lambda (skip if already exists) ───────────────
echo ""
echo "[1/5] Checking IAM role..."
ROLE_ARN=$(aws iam get-role --role-name $ROLE_NAME \
    --query 'Role.Arn' --output text 2>/dev/null || echo "NOT_FOUND")

if [ "$ROLE_ARN" = "NOT_FOUND" ]; then
    echo "  Creating IAM role: $ROLE_NAME"
    TRUST_POLICY='{
        "Version":"2012-10-17",
        "Statement":[{
            "Effect":"Allow",
            "Principal":{"Service":"lambda.amazonaws.com"},
            "Action":"sts:AssumeRole"
        }]
    }'
    ROLE_ARN=$(aws iam create-role \
        --role-name $ROLE_NAME \
        --assume-role-policy-document "$TRUST_POLICY" \
        --query 'Role.Arn' --output text)
    aws iam attach-role-policy \
        --role-name $ROLE_NAME \
        --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
    aws iam attach-role-policy \
        --role-name $ROLE_NAME \
        --policy-arn arn:aws:iam::aws:policy/CloudWatchLogsFullAccess
    echo "  Role created: $ROLE_ARN"
    echo "  Waiting 10s for IAM to propagate..."
    sleep 10
else
    echo "  Role already exists: $ROLE_ARN"
fi

# ── Step 2: Build the deployment package ──────────────────────────────────────
echo ""
echo "[2/5] Building deployment package..."
BUILD_DIR="/tmp/lambda_build"
rm -rf $BUILD_DIR && mkdir -p $BUILD_DIR

# Copy Lambda handler
cp lambda/lambda_function.py $BUILD_DIR/

# Install dependencies into the build dir
# Lambda runs Linux x86_64 -- download Linux wheels, NOT Mac ones
pip install pandas openpyxl \
    --target $BUILD_DIR \
    --platform manylinux2014_x86_64 \
    --implementation cp \
    --python-version 3.12 \
    --only-binary=:all: \
    --quiet \
    --no-cache-dir

# odfpy is pure Python (no wheel on PyPI) -- safe to install without platform flags
pip install odfpy \
    --target $BUILD_DIR \
    --quiet \
    --no-cache-dir

# Zip it up
cd $BUILD_DIR
zip -r /tmp/lambda_package.zip . -x "*.pyc" -x "__pycache__/*" > /dev/null
cd -
echo "  Package size: $(du -sh /tmp/lambda_package.zip | cut -f1)"
aws s3 cp /tmp/lambda_package.zip s3://$BUCKET/lambda/lambda_package.zip --region $REGION --only-show-errors
echo "  Package uploaded to s3://$BUCKET/lambda/"

# ── Step 3: Create or update the Lambda function ──────────────────────────────
echo ""
echo "[3/5] Deploying Lambda function..."
EXISTING=$(aws lambda get-function --function-name $FUNCTION_NAME \
    --region $REGION --query 'Configuration.FunctionName' \
    --output text 2>/dev/null || echo "NOT_FOUND")

if [ "$EXISTING" = "NOT_FOUND" ]; then
    echo "  Creating new function: $FUNCTION_NAME"
    aws lambda create-function \
        --function-name $FUNCTION_NAME \
        --runtime $RUNTIME \
        --role $ROLE_ARN \
        --handler lambda_function.lambda_handler \
        --code S3Bucket=$BUCKET,S3Key=lambda/lambda_package.zip \
        --timeout 900 \
        --memory-size 2048 \
        --region $REGION \
        --environment "Variables={S3_BUCKET=$BUCKET,AWS_REGION_NAME=$REGION}" \
        --output text > /dev/null
    echo "  Function created."
else
    echo "  Updating existing function..."
    aws lambda update-function-code \
        --function-name $FUNCTION_NAME \
        --s3-bucket $BUCKET --s3-key lambda/lambda_package.zip \
        --region $REGION \
        --output text > /dev/null
    echo "  Function updated."
fi

# ── Step 4: Add S3 trigger (skip if already configured) ──────────────────────
echo ""
echo "[4/5] Configuring S3 trigger..."
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
LAMBDA_ARN="arn:aws:lambda:$REGION:$ACCOUNT_ID:function:$FUNCTION_NAME"

# Allow S3 to invoke Lambda
aws lambda add-permission \
    --function-name $FUNCTION_NAME \
    --statement-id "S3InvokeLambda" \
    --action "lambda:InvokeFunction" \
    --principal s3.amazonaws.com \
    --source-arn "arn:aws:s3:::$BUCKET" \
    --region $REGION 2>/dev/null || echo "  (Permission already exists)"

# Set S3 bucket notification
NOTIFICATION_CONFIG=$(cat <<EOF
{
  "LambdaFunctionConfigurations": [
    {
      "LambdaFunctionArn": "$LAMBDA_ARN",
      "Events": ["s3:ObjectCreated:*"],
      "Filter": {
        "Key": {
          "FilterRules": [
            { "Name": "prefix", "Value": "raw/" }
          ]
        }
      }
    }
  ]
}
EOF
)
aws s3api put-bucket-notification-configuration \
    --bucket $BUCKET \
    --notification-configuration "$NOTIFICATION_CONFIG"
echo "  S3 trigger set: any upload to s3://$BUCKET/raw/ will run the pipeline"

# ── Step 5: Done ──────────────────────────────────────────────────────────────
echo ""
echo "[5/5] All done!"
echo ""
echo "======================================================"
echo " ✓ Lambda deployed: $FUNCTION_NAME"
echo " ✓ Trigger: s3://$BUCKET/raw/* → Lambda → s3://$BUCKET/processed/*"
echo " ✓ Logs: https://console.aws.amazon.com/cloudwatch/home?region=$REGION"
echo "======================================================"
echo ""
echo "Test it by uploading a raw ORR file:"
echo "  aws s3 cp data/raw/<your_file>.xlsx s3://$BUCKET/raw/"
echo "Then watch CloudWatch Logs for the output."