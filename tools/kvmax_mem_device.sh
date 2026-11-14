#!/system/bin/sh
# Ticket kv-maxlen: sample neuronrt resident memory while it runs the decode graph for one
# cache length C (one load, many Runs).  Same method as tools/mem_q06_device.sh.
#   sh kvmax_mem_device.sh <C> [iterations]
C="$1"; ITER="$2"
[ -z "$ITER" ] && ITER=20
DEV=/data/local/tmp/npuops
cd "$DEV" || exit 1
D=kvmax/c$C
echo "=== C $C BASELINE meminfo ==="
grep -E "MemFree|MemAvailable" /proc/meminfo
LD_LIBRARY_PATH=$PWD/lib ./bin/neuronrt -m hw -a $D/model.dla \
  -i $D/x.bin -i $D/mask.bin -i $D/cos14.bin -i $D/sin14.bin \
  -i $D/cos2.bin -i $D/sin2.bin -i $D/k_in.bin -i $D/v_in.bin \
  -o $D/out_hidden.bin -o $D/out_k.bin -o $D/out_v.bin \
  -c $ITER > kvmax/mem_c$C.runlog 2>&1 &
PID=$!
sleep 0.3
echo "=== PID $PID ==="
i=0
while [ -d "/proc/$PID" ] && [ $i -lt 10 ]; do
  echo "=== SAMPLE $i ==="
  grep -E "^(Rss|Pss|Pss_Anon|Pss_File|Pss_Shmem|Private_Dirty):" /proc/$PID/smaps_rollup 2>/dev/null
  grep -E "rchar|read_bytes" /proc/$PID/io 2>/dev/null
  echo "--- dmabuf vmas ---"
  grep dmabuf /proc/$PID/maps 2>/dev/null
  i=$((i+1))
  sleep 0.5
done
wait $PID
echo "=== AFTER meminfo ==="
grep -E "MemFree|MemAvailable" /proc/meminfo
echo "=== RUNLOG tail ==="
tail -4 kvmax/mem_c$C.runlog
