#!/bin/bash
# Self-contained: image torch (working CUDA) + pip faiss + clone + run the diagnostic controller.
# No SSH needed; monitor via the :8000 http proxy (DIAG.md, STATUS.txt, controller.out under /workspace).
mkdir -p /workspace && cd /workspace
nohup python3 -m http.server 8000 --directory /workspace >/dev/null 2>&1 &
exec > /workspace/setup.log 2>&1; set -x
echo "SETUP START"; date; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo SETUP > /workspace/setup.status
python3 -c "import torch;print('CUDA_BEFORE',torch.cuda.is_available())"
pip install -q "numpy<2" faiss-cpu networkx crafter sheeprl 2>&1 | tail -2
python3 -c "import torch;print('CUDA_AFTER',torch.cuda.is_available())"
git clone --depth 1 https://github.com/izmarel/mc-gpu-smoke /workspace/proj
cd /workspace/proj
cp research/p2e/cloud/faulthandler_sitecustomize.py /workspace/proj/sitecustomize.py
cp research/p2e/cloud/diag_controller.py /workspace/diag_controller.py
python research/p2e/patches/apply_integration.py 2>&1 | tail -1
echo READY > /workspace/setup.status
cd /workspace
setsid env PYTHONPATH=/workspace/proj python3 /workspace/diag_controller.py </dev/null >/workspace/controller.out 2>&1 &
echo "CONTROLLER LAUNCHED $(date -u)" >> /workspace/setup.log
sleep infinity
