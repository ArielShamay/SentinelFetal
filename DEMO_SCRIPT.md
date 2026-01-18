# SentinelFetal Demo Script

This guide is for the live presentation. Use a clean terminal with the virtual environment activated.

## Setup
1) Activate env: `.\.venv\Scripts\activate` (PowerShell).
2) Set path: `set PYTHONPATH=.`
3) Start simulator UI: `py scripts/run_simulation.py`

## Scenario A: Normal (Green)
- Narrate: “Here is a stable patient with no interventions.”
- Action: Select **Patient 1** (default profile, no injections).
- Expectation: Category 1 (Green), baseline ~140 bpm, moderate variability, no alerts.

## Scenario B: The Danger (Sinusoidal → Red)
- Narrate: “We inject a sinusoidal pattern—this is pathological.”
- Action: Inject **Sinusoidal Pattern** (Severity: Severe, Duration: 5 min) into the active patient.
- Expectation: Within ~20–40 seconds, override triggers **Category 3 (Red)**; alert mentions sinusoidal pattern.

## Scenario C: The Safety Net (Late Decelerations → Orange/Red)
- Narrate: “Recurrent late decels trigger the safety net.”
- Action: Inject **Late Decelerations** (Severity: Severe) into the active patient.
- Expectation: Recurrent late decels elevate to **Category 2/3**; alert shows late decelerations and elevated category.

## Tips
- If the dashboard pauses, reload the browser tab; the backend keeps running.
- To reset a patient, stop injections in the UI or restart the simulator.
- For higher load, increase patient count in settings and watch tick stability.
