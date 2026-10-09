#!/bin/sh
# As root, once per environment: uv, and unzip for bun's installer.
set -eu
command -v unzip > /dev/null || (apt-get update -qq && apt-get install -y -qq unzip > /dev/null)
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
