#!/system/bin/sh
# Ticket 05: sample neuronrt resident memory while it runs the 24-layer shared-weight graph
# (docs/OP-vendor-runtime.md sections 5-6 method: smaps_rollup + dmabuf maps + io).
#   sh mem_q05_device.sh <graph-name> [iterations]
NAME="$1"; ITER="${2:-400}"
DEV=/data/local/tmp/npuops; SUB=$DEV/q05
cd "$DEV" || exit 1
echo "=== BASELINE meminfo ==="
grep -E 'MemFree|MemAvailable' /proc/meminfo
LD_LIBRARY_PATH=$PWD/lib ./bin/neuronrt -m hw -a q05/$NAME.dla -i q05/$NAME.in.bin \
    -i q05/shared.bin -o q05/$NAME.mem.out.bin -c $ITER > q05/$NAME.mem.runlog 2>&1 &
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
tail -4 q05/$NAME.mem.runlog
