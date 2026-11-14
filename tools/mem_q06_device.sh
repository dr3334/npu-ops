#!/system/bin/sh
# Ticket 06: sample neuronrt resident memory while it runs the DECODE graph over the shared
# bin (one load, many Runs) -- same method as tools/mem_q05_device.sh.
#   sh mem_q06_device.sh <dla-name> [iterations]
NAME="$1"; ITER="$2"
[ -z "$ITER" ] && ITER=200
DEV=/data/local/tmp/npuops; SUB=$DEV/q06
cd "$DEV" || exit 1
echo "=== BASELINE meminfo ==="
grep -E 'MemFree|MemAvailable' /proc/meminfo
LD_LIBRARY_PATH=$PWD/lib ./bin/neuronrt -m hw -a q06/$NAME \
  -i q06/live/x.bin -i q06/live/mask.bin -i q06/live/cos14.bin -i q06/live/sin14.bin \
  -i q06/live/cos2.bin -i q06/live/sin2.bin -i q06/live/k_in.bin -i q06/live/v_in.bin \
  -i q06/shared.bin -o q06/live/out_hidden.bin -o q06/live/out_k.bin -o q06/live/out_v.bin \
  -c $ITER > q06/mem.runlog 2>&1 &
PID=$!
sleep 0.3
echo "=== PID $PID ==="
i=0
while [ -d "/proc/$PID" ] && [ $i -lt 14 ]; do
  echo "=== SAMPLE $i ==="
  grep -E '^(Rss|Pss|Pss_Anon|Pss_File|Pss_Shmem|Private_Dirty|Private_Clean):' /proc/$PID/smaps_rollup 2>/dev/null
  grep -E 'rchar|read_bytes' /proc/$PID/io 2>/dev/null
  echo "--- dmabuf vmas ---"
  grep dmabuf /proc/$PID/maps 2>/dev/null
  echo "--- dma heap (exporter totals) ---"
  dmabuf_dump -b 2>/dev/null | grep -E 'Dmabuf Inode|apusys|mtk-apu|mtk_mm' | head -25
  i=$((i+1))
  sleep 0.5
done
wait $PID
echo "=== AFTER meminfo ==="
grep -E 'MemFree|MemAvailable' /proc/meminfo
echo "=== RUNLOG tail ==="
tail -4 q06/mem.runlog
