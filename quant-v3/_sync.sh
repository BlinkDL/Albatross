#!/bin/bash
# Copy from windows workspace mirror to WSL
SRC_DIR="/mnt/d/code/workspace"
cd "$SRC_DIR"
for d in *; do
  if [[ "$d" == Albatross-BACKUP* ]]; then continue; fi
  if [[ "$d" == Albatross* ]]; then
    cp -f "$SRC_DIR/$d/faster3a_2605/cuda/rwkv7_int8_ops.cu" /home/njzy/Albatross/faster3a_2605/cuda/rwkv7_int8_ops.cu
    cp -f "$SRC_DIR/$d/faster3a_2605/cuda/rwkv7_int8_ops.cpp" /home/njzy/Albatross/faster3a_2605/cuda/rwkv7_int8_ops.cpp
    cp -f "$SRC_DIR/$d/faster3a_2605/quant/int8_linear.py" /home/njzy/Albatross/faster3a_2605/quant/int8_linear.py
    cp -f "$SRC_DIR/$d/faster3a_2605/rwkv7_fast_v3a.py" /home/njzy/Albatross/faster3a_2605/rwkv7_fast_v3a.py
    cp -rf "$SRC_DIR/$d/faster3a_2605/quant-v3/" /home/njzy/Albatross/quant-v3/
    echo "copied from $d"
    break
  fi
done
# fix double nesting
if [ -d /home/njzy/Albatross/quant-v3/quant-v3 ]; then
  cp -rf /home/njzy/Albatross/quant-v3/quant-v3/* /home/njzy/Albatross/quant-v3/
  rm -rf /home/njzy/Albatross/quant-v3/quant-v3
fi
