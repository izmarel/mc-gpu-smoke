ssh-keygen -A 2>/dev/null
mkdir -p ~/.ssh && echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
( /usr/sbin/sshd 2>/dev/null || service ssh start 2>/dev/null ) || true
mkdir -p /workspace && cd /workspace
nohup python3 -m http.server 8000 --directory /workspace >/dev/null 2>&1 &
exec > /workspace/setup.log 2>&1; set -x
echo "SETUP START"; date; nvidia-smi -L
echo SETUP > /workspace/setup.status
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /workspace bin/micromamba
export MAMBA_ROOT_PREFIX=/workspace/mamba
/workspace/bin/micromamba create -y -p /workspace/env -c pytorch -c nvidia -c conda-forge python=3.11 pytorch pytorch-cuda=11.8 mkl=2024.0 faiss-cpu 2>&1 | tail -5
RUN="/workspace/bin/micromamba run -p /workspace/env"
$RUN pip install -q "numpy<2" "setuptools<80" sheeprl crafter networkx py-spy 2>&1 | tail -3
$RUN python -c "import torch,faiss;print('CUDA',torch.cuda.is_available(),'faiss_threads',faiss.omp_get_max_threads())"
git clone --depth 1 https://github.com/izmarel/mc-gpu-smoke /workspace/proj
cd /workspace/proj && $RUN python research/p2e/patches/apply_integration.py 2>&1 | tail -1
echo "SETUP DONE"; date; echo READY > /workspace/setup.status
sleep infinity
