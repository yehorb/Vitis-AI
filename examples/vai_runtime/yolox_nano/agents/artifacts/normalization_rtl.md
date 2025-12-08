# STFT Normalization - RTL Implementation Notes

This document describes how to implement the dB normalization operation in RTL (Verilog/VHDL) for FPGA deployment.

## Python Reference

The normalization is defined in `src/python/stft_dataset/src/stft_dataset/normalization.py`.

```python
def normalize_db_scalar(x: float, vmin_db: float, vmax_db: float) -> float:
    scale = 1.0 / (vmax_db - vmin_db)
    y = (x - vmin_db) * scale
    if y < 0.0:
        y = 0.0
    elif y > 1.0:
        y = 1.0
    return y
```

## RTL Implementation

The operation consists of three stages:

### Stage 1: Subtract Offset
```
y1 = x - vmin_db
```
- `vmin_db` is a constant (e.g., -90.0 dB)
- For fixed-point: precompute `vmin_db` in the target format

### Stage 2: Multiply by Scale
```
y2 = y1 * scale
```
- `scale = 1.0 / (vmax_db - vmin_db)` is a precomputed constant
- For the default range [-90, -20] dB: `scale = 1/70 = 0.0142857...`
- For fixed-point: represent as `scale_fixed = round(scale * 2^N)` and shift result

### Stage 3: Clamp to [0, 1]
```
if y2 < 0.0:
    y3 = 0.0
elif y2 > 1.0:
    y3 = 1.0
else:
    y3 = y2
```
- Simple saturation logic
- For fixed-point [0, 1] represented as [0, 2^N-1]: saturate to min=0, max=2^N-1

## Parameters from meta.json

The normalization constants come from `meta.json`:

```json
{
  "render": {
    "vmin_db": -90,
    "vmax_db": -20
  }
}
```

### Precomputed Constants

| Parameter | Value | Notes |
|-----------|-------|-------|
| `vmin_db` | -90.0 | Offset to subtract |
| `vmax_db` | -20.0 | Upper bound |
| `range` | 70.0 | `vmax_db - vmin_db` |
| `scale` | 0.014285714... | `1.0 / range` |

## Fixed-Point Example (Q8.8 format)

For 16-bit fixed-point with 8 fractional bits:

```
vmin_db_fixed = round(-90.0 * 256) = -23040  (signed 16-bit)
scale_fixed   = round(0.0142857 * 256) = 4   (but loses precision)
```

For better precision, use more fractional bits for scale:

```
// Q1.15 for scale (15 fractional bits)
scale_q15 = round(0.0142857 * 32768) = 468

// Computation:
y1 = x - vmin_db_fixed;           // Q8.8 subtract
y2 = (y1 * scale_q15) >> 15;      // Q8.8 * Q1.15 -> Q9.23, shift to Q8.8
y3 = clamp(y2, 0, 256);           // 256 = 1.0 in Q8.8
```

## Pipeline Structure

```
┌─────────┐    ┌─────────┐    ┌─────────┐
│  SUB    │───▶│  MUL    │───▶│ CLAMP   │───▶ y_out
│ x-vmin  │    │ y*scale │    │ [0, 1]  │
└─────────┘    └─────────┘    └─────────┘
     │              │              │
   1 cycle       1 cycle        1 cycle
```

Total latency: 3 clock cycles (pipelined, 1 sample/cycle throughput)

## Verilog Pseudocode

```verilog
module normalize_db #(
    parameter DATA_WIDTH = 16,
    parameter FRAC_BITS = 8,
    parameter SCALE_FRAC_BITS = 15
)(
    input  wire clk,
    input  wire signed [DATA_WIDTH-1:0] x_in,
    input  wire signed [DATA_WIDTH-1:0] vmin_db,      // constant
    input  wire signed [SCALE_FRAC_BITS:0] scale,     // constant, Q1.15
    output reg  [DATA_WIDTH-1:0] y_out
);

    // Stage 1: Subtract
    reg signed [DATA_WIDTH:0] y1;
    always @(posedge clk) begin
        y1 <= x_in - vmin_db;
    end

    // Stage 2: Multiply
    reg signed [DATA_WIDTH+SCALE_FRAC_BITS:0] y2_full;
    reg signed [DATA_WIDTH-1:0] y2;
    always @(posedge clk) begin
        y2_full <= y1 * scale;
        y2 <= y2_full >>> SCALE_FRAC_BITS;
    end

    // Stage 3: Clamp
    localparam ONE = (1 << FRAC_BITS);  // 1.0 in fixed-point
    always @(posedge clk) begin
        if (y2 < 0)
            y_out <= 0;
        else if (y2 > ONE)
            y_out <= ONE;
        else
            y_out <= y2[DATA_WIDTH-1:0];
    end

endmodule
```
