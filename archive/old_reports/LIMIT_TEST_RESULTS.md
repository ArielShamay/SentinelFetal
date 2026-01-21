# Limit Test Results (Synthetic Simulator V2.0)

- Failure point (patients): **150**
- Failure mode: **Latency exceeded 2000ms (observed 764672ms)**
- Safe operational limit (80% rule): **120 patients**
- Engine: PyTorch backend (CPU only, Intel i5, no GPU)

## Summary Metrics
- Avg latency: 43 ms
- Max latency: 764672 ms
- CPU avg/max: 43.0% / 312.5%
- RAM max: 53 MB
- Accuracy final/min: 0.96 / 0.00
- TP/FP/FN: 76165/421334/2973

## Notes
- Signals include heavy noise and dropouts.
- Late decelerations injected into 20% of patients at random intervals.
- Termination criteria: latency>2000ms, CPU>95% for 30s, accuracy<70%, or crash.