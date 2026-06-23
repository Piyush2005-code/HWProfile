#!/bin/bash
set -e

echo "================================================="
echo "  ArchScope Lite — Dependency Installer"
echo "================================================="
echo ""

# 1. Detect OS and install system packages
if command -v apt-get >/dev/null 2>&1; then
    echo "[*] Detected Debian/Ubuntu-based system (e.g., Jetson Nano)"
    echo "[*] Updating package lists..."
    sudo apt-get update
    
    echo "[*] Installing build tools (GCC, Make)..."
    sudo apt-get install -y build-essential gcc make
    
    echo "[*] Installing Python 3 and plotting libraries..."
    # Installing via apt avoids PEP-668 "externally-managed-environment" 
    # errors that occur when using pip on modern Ubuntu/Debian systems.
    sudo apt-get install -y python3 python3-pandas python3-matplotlib python3-numpy

elif command -v brew >/dev/null 2>&1; then
    echo "[*] Detected macOS (Homebrew)"
    echo "[*] Installing build tools and Python 3..."
    brew install gcc make python
    
    echo "[*] Installing Python plotting libraries via pip..."
    pip3 install pandas matplotlib numpy

else
    echo "[!] Unsupported or undetected package manager."
    echo "[!] Please manually install:"
    echo "    1. A C compiler (gcc or clang) and Make"
    echo "    2. Python 3"
    echo "    3. Python packages: pandas, matplotlib, numpy"
    echo ""
    exit 1
fi

echo ""
echo "================================================="
echo "  Dependencies installed successfully! ✅"
echo "================================================="
echo "You can now run:"
echo "  make          # to build the profiler"
echo "  make run      # to build, run benchmarks, and generate plots"
