#!/usr/bin/env bash
# Deploy the voice studio to a single EC2 instance and copy the existing projects up.
#
#   ./deploy/aws_deploy.sh            provision, build, upload data, start
#   ./deploy/aws_deploy.sh --no-data  same, but skip the audio upload
#
# The instance keeps its database and audio on a persistent EBS volume mounted at
# /opt/appdata, so stopping and starting the instance loses nothing. Stop it when you are
# not reviewing: you then pay only for the disk.
set -euo pipefail

PROFILE="${AWS_PROFILE:-personal}"
REGION="${AWS_REGION:-ap-southeast-1}"
NAME="${NAME:-sinhala-voice}"
INSTANCE_TYPE="${INSTANCE_TYPE:-t4g.small}"
DISK_GB="${DISK_GB:-20}"
PORT=8000
KEY_PATH="${HOME}/.ssh/${NAME}.pem"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEND_DATA=1
[[ "${1:-}" == "--no-data" ]] && SEND_DATA=0

aws() { command aws --profile "$PROFILE" --region "$REGION" "$@"; }
say() { printf "\n\033[1m==> %s\033[0m\n" "$*"; }

say "Account and region"
aws sts get-caller-identity --query '[Account,Arn]' --output text
echo "region: $REGION   instance: $INSTANCE_TYPE   disk: ${DISK_GB}GB"

say "Key pair"
if [[ -f "$KEY_PATH" ]] && aws ec2 describe-key-pairs --key-names "$NAME" >/dev/null 2>&1; then
  echo "reusing $KEY_PATH"
else
  aws ec2 delete-key-pair --key-name "$NAME" >/dev/null 2>&1 || true
  aws ec2 create-key-pair --key-name "$NAME" --query KeyMaterial --output text > "$KEY_PATH"
  chmod 400 "$KEY_PATH"
  echo "created $KEY_PATH"
fi

say "Security group"
MY_IP="$(curl -s --max-time 10 https://checkip.amazonaws.com)/32"
SG=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=$NAME" \
       --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo None)
if [[ "$SG" == "None" || -z "$SG" ]]; then
  SG=$(aws ec2 create-security-group --group-name "$NAME" \
        --description "Sinhala English voice studio" --query GroupId --output text)
fi
# The app itself: reachable from anywhere, as chosen. It has no login of its own.
aws ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port $PORT \
    --cidr 0.0.0.0/0 >/dev/null 2>&1 || true
# SSH stays restricted to whoever runs this script.
aws ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 22 \
    --cidr "$MY_IP" >/dev/null 2>&1 || true
echo "$SG  (app open to 0.0.0.0/0, ssh limited to $MY_IP)"

say "Instance"
ID=$(aws ec2 describe-instances \
      --filters "Name=tag:Name,Values=$NAME" "Name=instance-state-name,Values=running,stopped,pending" \
      --query 'Reservations[0].Instances[0].InstanceId' --output text 2>/dev/null || echo None)
if [[ "$ID" == "None" || -z "$ID" ]]; then
  AMI=$(aws ssm get-parameters --names \
        /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-arm64 \
        --query 'Parameters[0].Value' --output text)
  echo "launching from $AMI"
  ID=$(aws ec2 run-instances --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
        --key-name "$NAME" --security-group-ids "$SG" \
        --block-device-mappings "[{\"DeviceName\":\"/dev/xvda\",\"Ebs\":{\"VolumeSize\":$DISK_GB,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME}]" \
        --user-data '#!/bin/bash
dnf install -y docker rsync
systemctl enable --now docker
usermod -aG docker ec2-user
mkdir -p /opt/appdata /opt/app && chown ec2-user:ec2-user /opt/appdata /opt/app
touch /opt/ready' \
        --query 'Instances[0].InstanceId' --output text)
else
  echo "reusing $ID"
  state=$(aws ec2 describe-instances --instance-ids "$ID" --query 'Reservations[0].Instances[0].State.Name' --output text)
  [[ "$state" == "stopped" ]] && aws ec2 start-instances --instance-ids "$ID" >/dev/null
fi
aws ec2 wait instance-running --instance-ids "$ID"
HOST=$(aws ec2 describe-instances --instance-ids "$ID" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "$ID at $HOST"

SSH="ssh -i $KEY_PATH -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 ec2-user@$HOST"
say "Waiting for the machine to finish booting"
for _ in $(seq 1 60); do $SSH 'test -f /opt/ready' 2>/dev/null && break; sleep 5; done
$SSH 'test -f /opt/ready' || { echo "instance never became ready"; exit 1; }
echo "ready"

say "Preparing directories"
$SSH 'sudo mkdir -p /opt/app /opt/appdata && sudo chown -R ec2-user:ec2-user /opt/app /opt/appdata' 2>/dev/null
echo "/opt/app and /opt/appdata owned by ec2-user"

say "Copying the application"
rsync -az --delete -e "ssh -i $KEY_PATH -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
  --exclude '.venv' --exclude '.git' --exclude 'data' --exclude '__pycache__' --exclude '*.pyc' \
  "$ROOT/app" "$ROOT/pyproject.toml" "$ROOT/README.md" "$ROOT/Dockerfile" ec2-user@"$HOST":/opt/app/

say "Installing the API key"
KEY=$(grep -E '^OPENAI_API_KEY=' "$ROOT/.env" | cut -d= -f2- | tr -d '"'"'"'`' | tr -d '\r\n')
[[ -n "$KEY" ]] || { echo "no OPENAI_API_KEY in .env"; exit 1; }
printf 'OPENAI_API_KEY=%s\n' "$KEY" | $SSH 'cat > /tmp/app.env && sudo install -m 600 -o root -g root /tmp/app.env /opt/app.env && rm /tmp/app.env'
echo "written to /opt/app.env, root only"

if [[ $SEND_DATA -eq 1 ]]; then
  say "Uploading projects (skipping _raw.wav intermediates, which nothing references)"
  rsync -a --info=progress2 --exclude '*_raw.wav' \
    -e "ssh -i $KEY_PATH -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
    "$ROOT/data/app.db" "$ROOT/data/projects" ec2-user@"$HOST":/opt/appdata/
fi

say "Building and starting"
$SSH "cd /opt/app && sudo docker build -q -t $NAME . && \
      sudo docker rm -f $NAME >/dev/null 2>&1; \
      sudo docker run -d --name $NAME --restart unless-stopped \
        --env-file /opt/app.env -e DATA_DIR=/app/data \
        -v /opt/appdata:/app/data -p $PORT:$PORT $NAME"

say "Checking it answers"
for _ in $(seq 1 30); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://$HOST:$PORT/api/health" || true)
  [[ "$code" == "200" ]] && break; sleep 3
done
curl -s "http://$HOST:$PORT/api/health"; echo

cat <<EOF

  Address:  http://$HOST:$PORT
  SSH:      ssh -i $KEY_PATH ec2-user@$HOST
  Stop:     aws --profile $PROFILE --region $REGION ec2 stop-instances --instance-ids $ID
  Start:    aws --profile $PROFILE --region $REGION ec2 start-instances --instance-ids $ID
  Destroy:  aws --profile $PROFILE --region $REGION ec2 terminate-instances --instance-ids $ID

  The public IP changes each time the instance is stopped and started.
EOF
