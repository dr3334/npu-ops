#!/usr/bin/env python3
"""The int8 requant contract, as integers -- shared by the golden builders.

Device behaviour (inferred from dumps, then confirmed out of sample; see
docs/OP-MATRIX.md \u00a73 and docs/OP-convpool.md \u00a75 CP-05):

    out = RoundingDivideByPOT( round_half_up(acc * M0 / 2^31), shift )

i.e. TFLite/gemmlowp MultiplyByQuantizedMultiplier(acc, M0, shift) -- TWO
rounding steps at integer granularity, not one rounding of the exact rational.
A golden written as np.round(acc * ratio) therefore differs from the device on
the values that sit near a half-LSB boundary (measured: 55 of 145,296 int8
FC/BMM values with the exact ratio; 0 with this module -- see
tools/verify_int8_requant.py and reports/Q02-int8-bitexact.md).

Two rules keep M0 honest:

  * M0 comes from the DECLARED multiplier as a short rational (1 -> M0=2^30,
    shift=+1 ; 1/32 -> M0=2^30, shift=-4), never from the float32 product of the
    stored scales: that product is 1/32 * (1 + 1.1e-8), whose QuantizeMultiplier
    is M0=1073741840, and replaying that M0 against 35 device dumps misses 69
    values (0 for the integer M0).
  * the ratio is snapped with Fraction.limit_denominator(), so a builder that
    carries 1/32 as a float64 product still lands on the exact power of two.

Only tensor-level (per-tensor) scales are covered. Per-channel scales,
asymmetric zero points and int16 accumulators are NOT verified -- they keep the
+-1 code allowance (see the record in the report).
"""
import numpy as np
from fractions import Fraction

__all__ = ["quantize_multiplier", "multiply_by_quantized_multiplier",
           "requant_int8", "requant_int8_num_den"]


def quantize_multiplier(ratio, max_den=1 << 20):
    """(M0, shift): ratio ~= M0 * 2^-31 * 2^shift, TFLite's sign convention.

    +shift = left shift applied *before* the high-multiply, -shift = right shift
    applied after it (genuine TFLite QuantizeMultiplier + MultiplyByQuantizedMultiplier
    pairing; an exact multiplier of 1 comes out as M0=2^30, shift=+1, i.e. the identity).
    """
    m = Fraction(float(ratio)).limit_denominator(max_den)
    shift = 0
    while m < Fraction(1, 2):
        m *= 2
        shift += 1
    while m >= 1:
        m /= 2
        shift -= 1
    m0 = int(np.floor(m * (1 << 31) + Fraction(1, 2)))
    return m0, -shift


def multiply_by_quantized_multiplier(acc, m0, shift):
    """gemmlowp MultiplyByQuantizedMultiplier on int64 (no int32 saturation)."""
    acc = np.asarray(acc, np.int64)
    left = shift if shift > 0 else 0
    right = 0 if shift > 0 else -shift
    x = acc << left
    # step 1: SaturatingRoundingDoublingHighMul -- round_half_up(x * M0 / 2^31)
    x = (x * np.int64(m0) + np.int64(1 << 30)) >> 31
    # step 2: RoundingDivideByPOT -- round_half_away_from_zero(x / 2**right)
    if right == 0:
        return x
    mask = (1 << right) - 1
    thr = np.int64(mask >> 1) + (x < 0).astype(np.int64)
    return (x >> right) + ((x & mask) > thr).astype(np.int64)


def requant_int8_num_den(acc, num, den):
    """Contract requant with an explicit rational multiplier, clipped to int8."""
    m0, shift = quantize_multiplier(Fraction(int(num), int(den)))
    return np.clip(multiply_by_quantized_multiplier(acc, m0, shift),
                   -128, 127).astype(np.int8)


def requant_int8(acc, ratio):
    """Contract requant for an already-known scale ratio (snapped to a short rational)."""
    m0, shift = quantize_multiplier(ratio)
    return np.clip(multiply_by_quantized_multiplier(acc, m0, shift),
                   -128, 127).astype(np.int8)
