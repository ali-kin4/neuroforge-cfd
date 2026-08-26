#!/usr/bin/env bash
# Install OpenFOAM (ESI) inside a WSL2 Ubuntu distro, for the Paper-2 warm-start
# experiment (see docs/ROADMAP_paper2.md).
#
# Run from Windows:
#     wsl -d Ubuntu -- bash /mnt/d/Codes/Github/neuroforge-cfd/scripts/install_openfoam_wsl.sh
#
# It will prompt for your sudo password. Downloads ~1-2 GB.
set -euo pipefail

echo "== distro =="
lsb_release -ds || true
echo

if [ -f /etc/os-release ]; then
    . /etc/os-release
    echo "codename: ${VERSION_CODENAME:-unknown}"
fi

echo
echo "== 1/4 adding the OpenFOAM apt repository =="
curl -fsSL https://dl.openfoam.com/add-debian-repo.sh | sudo bash

echo
echo "== 2/4 apt-get update =="
sudo apt-get update

echo
echo "== 3/4 selecting the newest openfoam*-default package =="
PKG=$(apt-cache search 'openfoam.*-default' \
      | awk '{print $1}' \
      | grep -E '^openfoam[0-9]+-default$' \
      | sort -V | tail -1)

if [ -z "${PKG}" ]; then
    echo "ERROR: no openfoam<version>-default package found in the repository." >&2
    echo "Available openfoam packages:" >&2
    apt-cache search openfoam | head -20 >&2
    exit 1
fi
echo "selected: ${PKG}"

echo
echo "== 4/4 installing ${PKG} (this is the big download) =="
sudo apt-get install -y "${PKG}"

echo
echo "== verifying =="
BASHRC=$(for f in /usr/lib/openfoam/openfoam*/etc/bashrc /opt/openfoam*/etc/bashrc; do
             [ -f "$f" ] && echo "$f"
         done | sort -V | tail -1)

if [ -z "${BASHRC}" ]; then
    echo "ERROR: installed, but no etc/bashrc found where expected." >&2
    exit 1
fi

# shellcheck disable=SC1090
source "${BASHRC}" >/dev/null 2>&1 || true
echo "bashrc  : ${BASHRC}"
echo "version : ${WM_PROJECT_VERSION:-unknown}"
echo "simpleFoam : $(command -v simpleFoam || echo 'NOT ON PATH')"
echo "blockMesh  : $(command -v blockMesh  || echo 'NOT ON PATH')"
echo "subsetMesh : $(command -v subsetMesh || echo 'NOT ON PATH')"
echo "topoSet    : $(command -v topoSet    || echo 'NOT ON PATH')"

echo
echo "Done. Back in Windows, run:"
echo "    .\\.venv\\Scripts\\python.exe scripts/openfoam_warm_start.py --check"
