#!/system/bin/sh
# Device-side helper: run neuronrt under the ioctl recorder.
#   sh run_shim.sh <dla> <in.bin> [weights.bin] <out.bin> <log> <bufdir>
# Prints the decoder output path so the caller can pull it.
set -u
DLA="$1"; IN="$2"
if [ "$#" -eq 6 ]; then
  W="$3"; OUT="$4"; LOG="$5"; BUF="$6"; EXTRA="-i $W"
else
  W=""; OUT="$3"; LOG="$4"; BUF="$5"; EXTRA=""
fi

rm -rf "$BUF"; mkdir -p "$BUF"
cd /data/local/tmp/npuops || exit 1

IOLOG="$LOG" IODIR="$BUF" \
LD_PRELOAD=/data/local/tmp/npuops/probe/libiolog2.so \
LD_LIBRARY_PATH=/data/local/tmp/npuops/lib \
./bin/neuronrt -m hw -a "$DLA" -i "$IN" $EXTRA -o "$OUT" -c 1
rc=$?
echo "neuronrt_rc=$rc log=$LOG buf=$BUF"
ls -l "$LOG" 2>/dev/null
exit $rc
