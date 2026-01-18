# Demo Cheat Sheet (Printable)

## Launch
```
py -m venv .venv
.\.venv\Scripts\activate
set PYTHONPATH=.
py scripts/run_simulation.py
```

## Scenario 1 – Normal (Green)
- Patient: 1 (no injections)
- Expect: Category 1 (Green); baseline ~140 bpm; moderate variability; no alerts.

## Scenario 2 – Sinusoidal (Red Alert)
- Injection: Sinusoidal Pattern, Severity: Severe, Duration: 5 min
- Expect: Category 3 (Red) within ~20–40s; alert explicitly mentions sinusoidal pattern.

## Scenario 3 – Late Decelerations (Safety Override)
- Injection: Late Decelerations, Severity: Severe
- Expect: Category 2/3 after recurrent events; alert notes late decelerations and elevated category.

## Quick Tips
- Reload the browser tab if UI stalls—the backend continues running.
- Stop injections in the UI or restart the simulator to reset a patient.
- Increase patient count to show load handling; watch tick stability indicator.
