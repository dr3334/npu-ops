# ops/_probe/bb08-prof — ticket 08 evidence (profiling levels on the board)

Ticket: `.scratch/compiler-blackbox/issues/08-profile-on-board.md`
Report: `reports/compiler-blackbox-08.md`

## What is here

| path | what |
|---|---|
| `00-build.sh` | host (WSL): build the two `.tflite` inputs, compile **8** DLAs (`fc_p0..p3`, `dec_p0..p3`) |
| `build/*.compile.log` | per product: cmd / ncc version / **exec plan** (layer index + op class) / memory summary / size / sha256 |
| `10-stage.sh` | push the 8 products + inputs to `/data/local/tmp/npuops/bb08` |
| `20-run-one.sh` | ONE profiled batch (`MTKNN_ENABLE_PROFILER=1 MTKNN_PROFILING_LEVEL=L`), console + `logcat -s neuron`, pulls the level-3 JSON, then reads dmesg for the `MDLA timeout` bad state |
| `21-latency.sh` | clean `-c N` totals, **paired alternating** p0,p1,p2,p3 per round (timings.md discipline) |
| `22-mismatch.sh` | level-gating text on the board (console + logcat) |
| `30-host-levels.sh` | the same matrix on the **host** runtime (`-m null`), which prints the gating text and the LEVEL 1/2 summary tables |
| `40-board-profqos.sh` | runtime's own `getProfiledQoSData` (`exec_us`) — clean and **with level 3 in the same run** (pins the cycle->us clock, 07 §9 left this open) |
| `90-perop.py` | level-3 dump -> per-command table + by-op-class rollup + reconciliation vs the wall clock |
| `23-c-sweep.sh` | control for 08a: same product, only `-c` changes, paired alternating (does `ms/inf` depend on N?) |
| `24-latency-08a.sh` | 08a: `-c 200` (fc) / `-c 50` (dec) × 9 rounds, paired alternating, writes only into `board/08a/` |
| `92-latency.py` | median / MAD + per-round paired ratios for `21-latency.sh` output |
| `92b-paired.py` | 08a: median of the **per-round** ratios, its MAD, and the exact two-sided sign test |
| `board/` | raw evidence pulled from the device |
| `host/` | raw evidence from the host runtime (`-m null`) |

## Discipline (why the scripts look like this)

* **one level at a time**, healthcheck before/after each level (`tools/healthcheck.sh --quick`);
* `MDLA timeout` / `mdw_cmd_complete ... ret(-5)` in dmesg => the board is in the bad state
  (`docs/OP-toolchain.md` 6.1): `adb reboot`, wait, **`adb root`**, healthcheck, only then continue;
* every board invocation goes through `tools/devlock.sh run -- ...` with `NPU_LOCK_HELD=1`;
* long outputs land in `/mnt/d/...` (this directory), never `/tmp`, and each batch finishes inside
  a single call (no `nohup`).
