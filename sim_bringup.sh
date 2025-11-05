#!/usr/bin/env bash
# sim_bringup.sh
set -e
if ! pgrep -f "roscore" >/dev/null; then
  echo "[sim] starting roscore..."
  roscore >/tmp/roscore_sim.log 2>&1 &
  sleep 2
fi
echo "[sim] starting sim_ridgeback..."
python3 sim_ridgeback.py &
echo "[sim] starting sim_mocap..."
python3 sim_mocap.py &
# echo "[sim] launched. Try:  rostopic list"
# echo "Then run your controller in another shell:"
# echo "python3 CollaborativeGameRealtime.py --mocap ros --human Human --robot Ridgeback_base --theta-deg 30 --x1-des-x 7 --x1-des-y 0 --tf 30"
