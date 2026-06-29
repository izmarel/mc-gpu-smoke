#!/bin/bash
echo "pod_launch started $(date -u)" >> /workspace/launch.log
for i in $(seq 1 120); do [ "$(cat /workspace/setup.status 2>/dev/null)" = "READY" ] && break; sleep 20; done
echo "READY seen $(date -u)" >> /workspace/launch.log
cp /workspace/sitecustomize.py /workspace/proj/sitecustomize.py 2>>/workspace/launch.log
cp /workspace/lp_memory_reward.py /workspace/proj/research/p2e/drive/lp_memory_reward.py 2>>/workspace/launch.log
cp /workspace/inject.py /workspace/proj/research/p2e/retrieval/inject.py 2>>/workspace/launch.log
pgrep -f diag_controller.py >/dev/null && { echo "already running" >>/workspace/launch.log; exit 0; }
cd /workspace
setsid /workspace/bin/micromamba run -p /workspace/env python /workspace/diag_controller.py </dev/null >/workspace/controller.out 2>&1 &
echo "controller launched $(date -u)" >> /workspace/launch.log
