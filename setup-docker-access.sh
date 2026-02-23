#!/bin/bash
# One-time setup: give Ava permanent docker access without sudo
# Run once with: sudo bash setup-docker-access.sh
set -e

echo "▶ Adding noufal to docker group..."
usermod -aG docker noufal

echo "▶ Adding NOPASSWD sudoers rule for docker (belt + suspenders)..."
cat > /etc/sudoers.d/99-docker-nopasswd << 'EOF'
# Allow noufal to run docker/docker-compose without password
noufal ALL=(ALL) NOPASSWD: /usr/bin/docker, /usr/bin/docker-compose, /usr/bin/docker-buildx
EOF
chmod 440 /etc/sudoers.d/99-docker-nopasswd

echo "▶ Fixing docker socket permissions for current session..."
chmod 666 /var/run/docker.sock

echo ""
echo "✓ Done. Ava can now run docker without any help."
echo "  Group change takes full effect on next login/openclaw restart."
echo "  NOPASSWD rule is effective immediately."
