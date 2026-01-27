# UI Reference (Truth-Aligned)

Date: 2026-01-27
Scope: React UI + FastAPI/WebSocket contract **as implemented in code**.

## Overview
- Frontend: `frontend/` (React + TypeScript)
- Backend: `api/` (FastAPI)
- Real-time updates: WebSocket (`/ws/stream`) + REST for snapshots and control

## How the UI connects
- REST base URL: `VITE_API_URL` (frontend env) or `/api` by default (`frontend/src/services/api.ts`).
- WebSocket URL: `ws://localhost:8001/ws/stream` by default (`frontend/src/services/websocket.ts`).
- WebSocket query params: `client_id` (optional), `format` (`json` or `msgpack`).

## REST endpoints (current)
- `GET /api/health` → `{ status, version, timestamp }`
- `GET /api/patients?duration_minutes=5` → `{ patients: PatientSnapshot[], count, timestamp }`
- `GET /api/patients/summary` → `{ patients: PatientSummary[], count, timestamp }`
- `GET /api/patients/{patient_id}?duration_minutes=5` → `PatientSnapshot`
- `GET /api/patients/{patient_id}/history?duration_minutes=30` → `{ patient_id, fhr[], uc[], timestamps[], duration_minutes }`
- `POST /api/patients/{patient_id}/event` → `EventInjectionResponse`
- `GET /api/simulation/status` → `SimulationStatus`
- `POST /api/simulation/start|stop|pause|resume|reset` → `SimulationResponse`
- `POST /api/simulation/command` → `SimulationResponse`
- `PATCH /api/simulation/config` → `SimulationResponse`
- `GET /ws/stats` (not under `/api`) → `{ connected_clients, queue_size, running }`
- `GET /ws/health` → `{ status, clients, queue_size }`

## JSON payload contract (verified)

### PatientSnapshot (REST)
Fields come from `api/models/schemas.py` and are used in `frontend/src/types/index.ts`:
- `patient_id: string`
- `bed_number: number`
- `category: 1|2|3` and `category_name: string`
- `metrics: { baseline_fhr, current_fhr, variability, current_uc, acceleration_count?, deceleration_count? }`
- `fhr_history: number[]`, `uc_history: number[]`, `timestamps: number[]`
- `alerts: { type, message, severity, timestamp }[]`
- `trend_data?: object | null`
- `explanation?: object | null`
- `highlight_regions?: { start_idx, end_idx, region_type, severity, label, color? }[]`
- `fsqi_score: number`
- `has_active_event: boolean`
- `last_update: number`

### EventInjection (REST request)
- `event_type: "LATE_DECEL"|"VARIABLE_DECEL"|"PROLONGED_DECEL"|"BRADYCARDIA"|"TACHYCARDIA"|"MINIMAL_VARIABILITY"|"HYPERSTIM"|"SINUSOIDAL"|"RECOVERY"`
- `severity: "mild"|"moderate"|"severe"`
- `duration_seconds?: number`
- `duration_minutes?: number`
- `params?: object`

### WebSocket messages (stream)
The backend **does not** wrap messages in the `WSMessage.payload` schema; it sends raw dicts.

#### Connected message (sent on connect)
```json
{
  "type": "connected",
  "client_id": "8chars",
  "format": "json",
  "timestamp": 1737940000.0,
  "subscribed_to": "P1" // only on /ws/stream/{patient_id}
}
```

#### Patient update (streamed)
`api/services/orchestrator_adapter.py` sends a `type: "patient_update"` payload:
```json
{
  "type": "patient_update",
  "timestamp": 1737940000.0,
  "patient_id": "P1",
  "category": 1,
  "baseline": 140.0,
  "variability": 10.0,
  "fhr_latest": [/* usually last 16 samples (4s @ 4Hz) */],
  "uc_latest": [/* usually last 16 samples (4s @ 4Hz) */],
  "fsqi": 1.0,
  "confidence": 0.0,
  "findings": { /* dict */ },
  "mhr_alert": { "is_mhr": false, "confidence": 0.0, "recommended_action": "NONE", "detection_methods": [] },
  "trend_score": 0.0,
  "trend_slope": 0.0,
  "explanation": null,
  "highlight_regions": []
}
```

#### Client → server control messages
- Subscribe: `{ "type": "subscribe", "patient_ids": ["P1", "P2"] }`
- Unsubscribe: `{ "type": "unsubscribe" }`
- Pong: `{ "type": "pong" }`

## Known limitations / assumptions
- **MessagePack negotiation is partial**: welcome/heartbeat uses `encode_message(...)`, but patient updates are sent as JSON text (`send_text`).
- **Batch updates are not emitted** by the backend today (frontend can handle `patients[]`, but producer sends single-patient updates).
- `fhr_latest`/`uc_latest` are truncated to the last 16 samples in `_push_websocket_update`, while `_on_tick` can push full arrays (both are currently `type: \"patient_update\"`).
- **Optional fields** (`explanation`, `highlight_regions`, `mhr_alert`, trend fields) are only present when upstream modules supply them.
