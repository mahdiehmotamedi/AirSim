#!/usr/bin/env bash
# ============================================================================
# Ubuntu 20.04 helper -- build Unreal Engine 5.3 + Colosseum from source.
# Review before running! Every heavy step can be skipped by answering n.
# ============================================================================
set -euo pipefail

WORK_DIR="${WORK_DIR:-$HOME/colosseum-build}"   # needs ~150 GB free
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

echo "== 1/4  System packages =="
sudo apt update
sudo apt install -y git build-essential clang-12 clang++-12 cmake ninja-build \
                    unzip xdg-user-dirs python3-pip

echo "== 2/4  Register GitHub with Epic (manual, one-time) =="
echo "   -> https://www.epicgames.com/id/account  : link your GitHub account"
echo "   -> accept the invitation mail from EpicGames/UnrealEngine"
read -rp "Done? [y/N] " ok && [ "$ok" = y ]

echo "== 3/4  Unreal Engine 5.3 (1-2 h) =="
if [ ! -d UnrealEngine ]; then
  git clone -b 5.3 https://github.com/EpicGames/UnrealEngine.git
  cd UnrealEngine
  ./Setup.sh
  ./GenerateProjectFiles.sh
  make -j"$(nproc)"
  cd ..
fi

echo "== 4/4  Colosseum (20-40 min) =="
if [ ! -d Colosseum ]; then
  git clone https://github.com/CodexLabs-LLC/Colosseum.git
  cd Colosseum
  ./setup.sh          # installs the right cmake / llvm / eigen versions
  ./build.sh
  cd ..
fi

echo
echo "== done =="
echo "Launch the editor with the Blocks environment:"
echo "  $WORK_DIR/UnrealEngine/Engine/Binaries/Linux/UE4Editor \\"
echo "      $WORK_DIR/Colosseum/Unreal/Environments/BlocksV2/BlocksV2.uproject"
echo
echo "Python client (use a 3.10 venv):"
echo "  python3 -m venv ~/.venvs/airsim && source ~/.venvs/airsim/bin/activate"
echo "  pip install airsim msgpack-rpc-python opencv-python numpy"
