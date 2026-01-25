# Phase 2: WebSocket Stream Implementation - PRD

**Phase:** 2 of 6
**Duration:** 2-3 days
**Priority:** Critical (Real-time foundation)
**Risk Level:** Medium
**Dependencies:** Phase 1 Complete

---

## 1. Overview

### 1.1 Purpose
Phase 2 implements the real-time WebSocket streaming system that pushes patient data from the backend to connected frontend clients. This replaces Streamlit's polling-based architecture with a push-based system, enabling <10ms latency updates.

### 1.2 Goals
1. Implement AsyncBroadcaster for efficient multi-client updates
2. Create WebSocket endpoint with proper connection lifecycle
3. Implement binary serialization (MessagePack) for performance
4. Add connection health monitoring (heartbeat/ping-pong)
5. Handle graceful client disconnection and reconnection

### 1.3 Non-Goals
- Frontend WebSocket client (Phase 3)
- Chart rendering (Phase 4)
- Load balancing across multiple servers (future scope)

---

## 2. User Stories

### US-2.1: Real-Time Patient Updates
**As a** frontend application
**I want** to receive patient data via WebSocket
**So that** I can update the UI in real-time without polling

**Acceptance Criteria:**
- [ ] WebSocket connection established at `/ws/stream`
- [ ] Receive patient snapshots at 4Hz (250ms intervals)
- [ ] Latency from backend to browser < 10ms
- [ ] All 20 patients' data included in each update

### US-2.2: Single Patient Subscription
**As a** frontend application
**I want** to subscribe to a specific patient
**So that** I can receive high-fidelity data for the detail view

**Acceptance Criteria:**
- [ ] Optional `patient_id` parameter to filter updates
- [ ] Receive only specified patient's data when subscribed
- [ ] Ability to change subscription without reconnecting

### US-2.3: Connection Health
**As a** frontend application
**I want** heartbeat messages from the server
**So that** I can detect and recover from stale connections

**Acceptance Criteria:**
- [ ] Server sends ping every 30 seconds
- [ ] Client can respond with pong
- [ ] Connection closed after 3 missed pongs
- [ ] Frontend notified of connection state changes

### US-2.4: Binary Serialization
**As a** system architect
**I want** MessagePack instead of JSON
**So that** bandwidth is reduced and parsing is faster

**Acceptance Criteria:**
- [ ] All WebSocket messages use MessagePack format
- [ ] Message size reduced by ~50% vs JSON
- [ ] Parse time < 1ms for 20-patient update

### US-2.5: Graceful Reconnection
**As a** frontend application
**I want** to reconnect automatically
**So that** temporary network issues don't require page refresh

**Acceptance Criteria:**
- [ ] Server assigns client ID on connect
- [ ] Reconnection uses same client ID
- [ ] No duplicate messages during reconnection
- [ ] Missed messages during disconnect are not replayed (latest-only)

---

## 3. Functional Requirements

### FR-2.1: WebSocket Endpoints

| Endpoint | Purpose |
|----------|---------|
| `/ws/stream` | Full ward stream (all patients) |
| `/ws/stream/{patient_id}` | Single patient stream |

### FR-2.2: Message Types

| Type | Direction | Purpose |
|------|-----------|---------|
| `patient_update` | Server→Client | Patient snapshot data |
| `ping` | Server→Client | Heartbeat check |
| `pong` | Client→Server | Heartbeat response |
| `subscribe` | Client→Server | Change subscription |
| `error` | Server→Client | Error notification |

### FR-2.3: AsyncBroadcaster

The AsyncBroadcaster shall:
- Maintain a registry of connected clients
- Support topic-based subscriptions (patient IDs)
- Broadcast updates to all relevant subscribers
- Handle backpressure (slow clients)
- Clean up disconnected clients automatically

### FR-2.4: Message Format (Patient Update)

```python
{
    "type": "patient_update",
    "timestamp": 1706101800.123,
    "patients": [
        {
            "patient_id": "Patient-1",
            "category": 1,
            "baseline": 142.5,
            "variability": 12.3,
            "fhr_latest": [141, 143, 142, 144],  # Last 4 samples
            "uc_latest": [10, 12, 15, 18],
            "mhr_alert": false,
            "highlight_regions": [],
            "trend_score": 15,
            "active_event": null
        },
        // ... more patients
    ]
}
```

### FR-2.5: Connection Lifecycle

```
1. Client connects → Server assigns client_id
2. Server sends "connected" message with client_id
3. Server adds client to broadcast registry
4. Server pushes updates at 4Hz
5. Server sends ping every 30s
6. Client responds with pong
7. On disconnect → Server removes from registry
8. On reconnect → Client sends previous client_id
```

---

## 4. Non-Functional Requirements

### NFR-2.1: Performance
- Latency: < 10ms from orchestrator push to client receive
- Throughput: Support 100 concurrent connections
- Memory: < 1MB per connection

### NFR-2.2: Reliability
- Auto-cleanup of dead connections within 90 seconds
- No memory leaks over 24-hour operation
- Graceful degradation under load

### NFR-2.3: Monitoring
- Log connection/disconnection events
- Metrics for active connections, message rate
- Alert on high error rate

---

## 5. Technical Constraints

### 5.1 Async Requirements
- All WebSocket handlers must be async
- Orchestrator data pushed via async queue
- No blocking operations in broadcast path

### 5.2 Thread Safety
- Orchestrator runs in separate thread
- AsyncBroadcaster accessed from async context
- Use thread-safe queue for cross-thread communication

### 5.3 Serialization
- MessagePack primary format
- JSON fallback for debugging
- Negotiated on connection (Accept header)

---

## 6. Dependencies

### 6.1 External Dependencies
- websockets >= 12.0
- msgpack >= 1.0.7
- anyio >= 4.0.0

### 6.2 Internal Dependencies
- Phase 1 complete (FastAPI app structure)
- OrchestratorAdapter with push capability
- DataBridge integration

---

## 7. Acceptance Criteria Summary

| ID | Criteria | Verification Method |
|----|----------|---------------------|
| AC-2.1 | WebSocket connects successfully | Browser DevTools |
| AC-2.2 | Receives updates at 4Hz | Message rate measurement |
| AC-2.3 | Latency < 10ms | Network timing |
| AC-2.4 | MessagePack messages decode correctly | Unit test |
| AC-2.5 | Ping/pong cycle works | Manual test |
| AC-2.6 | Disconnection handled gracefully | Kill connection test |
| AC-2.7 | 100 concurrent connections stable | Load test |

---

## 8. Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Backpressure from slow clients | Medium | Medium | Drop messages for slow clients |
| Memory leak on disconnect | Low | High | Careful resource cleanup |
| Cross-thread race conditions | Medium | High | Use thread-safe queue |
| MessagePack compatibility | Low | Low | Test with frontend early |

---

## 9. Timeline

| Day | Tasks |
|-----|-------|
| Day 1 | AsyncBroadcaster, basic WebSocket endpoint |
| Day 2 | MessagePack, orchestrator integration |
| Day 3 | Heartbeat, reconnection, testing |

---

## 10. Deliverables Checklist

- [x] `api/services/broadcaster.py` - AsyncBroadcaster class
- [x] `api/routers/websocket.py` - Full WebSocket implementation
- [x] `api/services/message_encoder.py` - MessagePack serialization
- [x] `api/services/orchestrator_bridge.py` - Thread-safe data transfer
- [x] Integration with orchestrator callback (via OrchestratorAdapter)
- [x] `api/main.py` - Lifespan handler for broadcaster & bridge

---

## 11. Implementation Status

### ✅ Phase 2 Complete - January 2025

**Files Implemented:**

| File | Status | Description |
|------|--------|-------------|
| `api/services/broadcaster.py` | ✅ Complete | AsyncBroadcaster singleton with client management, subscription filtering, heartbeat |
| `api/services/message_encoder.py` | ✅ Complete | JSON/MessagePack encoding with format negotiation |
| `api/services/orchestrator_bridge.py` | ✅ Complete | Thread-safe queue bridge between sync orchestrator and async broadcaster |
| `api/routers/websocket.py` | ✅ Complete | `/ws/stream`, `/ws/stream/{patient_id}`, `/ws/stats`, `/ws/health` |
| `api/main.py` | ✅ Updated | Lifespan starts/stops broadcaster and bridge |
| `api/services/orchestrator_adapter.py` | ✅ Updated | `_push_websocket_update()` pushes to bridge |

**WebSocket Endpoints:**

| Endpoint | Status | Description |
|----------|--------|-------------|
| `WS /ws/stream` | ✅ | All patients stream |
| `WS /ws/stream/{patient_id}` | ✅ | Single patient stream |
| `GET /ws/stats` | ✅ | Broadcaster statistics |
| `GET /ws/health` | ✅ | WebSocket subsystem health |

**Architecture:**
```
┌─────────────────────┐         ┌─────────────────────┐         ┌─────────────────────┐
│  Orchestrator       │  sync   │  OrchestratorBridge │  async  │  AsyncBroadcaster   │
│  (_tick() loop)     │ ──────▶ │  (thread-safe queue)│ ──────▶ │  (client mgmt)      │
└─────────────────────┘         └─────────────────────┘         └─────────────────────┘
                                                                          │
                                                                          ▼
                                                                 ┌─────────────────────┐
                                                                 │  WebSocket Clients  │
                                                                 └─────────────────────┘
```

**Features Implemented:**
- ✅ Client registration with UUID
- ✅ Patient-based subscription filtering
- ✅ Heartbeat/ping-pong for connection health
- ✅ Backpressure handling (drop messages for slow clients)
- ✅ Dead connection cleanup
- ✅ JSON format (MessagePack optional)
- ✅ Format negotiation via query param or Accept header

**Deviations from Plan:**
1. Default format is JSON (not MessagePack) for broader browser compatibility
2. Added `/ws/health` endpoint for health checks
3. `orchestrator_bridge.py` is separate from `broadcaster.py` for cleaner separation

---

*End of Phase 2 PRD*
