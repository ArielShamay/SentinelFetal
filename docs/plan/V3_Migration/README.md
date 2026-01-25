# SentinelFetal V3.0 - Full-Stack Migration Master Plan

**Version:** 1.0.0
**Date:** January 24, 2026
**Status:** Ready for Execution
**Total Estimated Effort:** 23-30 Person-Days

---

## Executive Summary

This directory contains the complete implementation plan for migrating SentinelFetal from Streamlit to a professional real-time stack (FastAPI + React + Lightweight-Charts). The migration is divided into 7 phases, each with its own PRD (Product Requirements Document) and SPECS (Technical Specifications).

### Why This Migration?

| Current State (Streamlit) | Target State (V3) |
|---------------------------|-------------------|
| ~5 FPS effective | 60 FPS guaranteed |
| 190ms frame latency | <17ms frame latency |
| Polling-based updates | Push-based WebSocket |
| SVG rendering (slow) | Canvas/WebGL (fast) |
| Single-threaded UI | Async, non-blocking |
| 20 patients max | 50+ patients supported |

---

## Phase Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        V3 MIGRATION TIMELINE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Phase 0        Phase 1        Phase 2        Phase 3        Phase 4       │
│  ┌──────┐      ┌──────┐       ┌──────┐       ┌──────┐       ┌──────┐       │
│  │Setup │ ──▶  │ API  │  ──▶  │  WS  │  ──▶  │React │  ──▶  │Charts│       │
│  │ 2d   │      │ 4d   │       │ 3d   │       │ 4d   │       │ 5d   │       │
│  └──────┘      └──────┘       └──────┘       └──────┘       └──────┘       │
│                                                                             │
│                                              Phase 5        Phase 6        │
│                                              ┌──────┐       ┌──────┐       │
│                                         ──▶  │Polish│  ──▶  │ QA   │       │
│                                              │ 4d   │       │ 3d   │       │
│                                              └──────┘       └──────┘       │
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════   │
│                         TOTAL: 23-30 DAYS                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase Breakdown

| Phase | Name | Duration | Key Deliverables | Risk |
|-------|------|----------|------------------|------|
| **0** | Setup & Foundation | 2 days | Dev environment, project structure, CI/CD | Low |
| **1** | Backend API (FastAPI) | 3-4 days | REST endpoints, Pydantic models, routing | Low |
| **2** | WebSocket Stream | 2-3 days | Real-time data push, AsyncBroadcaster | Medium |
| **3** | Frontend Scaffold | 3-4 days | React app, routing, state management | Low |
| **4** | Canvas Charts | 4-5 days | Lightweight-Charts CTG monitor, 60FPS | Medium |
| **5** | Feature Parity | 3-4 days | God Mode, i18n, panels, alerts | Low |
| **6** | Testing & Deployment | 2-3 days | E2E tests, Docker, documentation | Low |

---

## Directory Structure

```
V3_Migration/
├── README.md                          # This file
│
├── Phase_0_Setup/
│   ├── PRD.md                         # Requirements for setup phase
│   └── SPECS.md                       # Technical specs for setup
│
├── Phase_1_Backend_API/
│   ├── PRD.md                         # FastAPI requirements
│   └── SPECS.md                       # API specs, endpoints, models
│
├── Phase_2_WebSocket/
│   ├── PRD.md                         # WebSocket requirements
│   └── SPECS.md                       # Protocol, broadcaster, serialization
│
├── Phase_3_Frontend_Scaffold/
│   ├── PRD.md                         # React app requirements
│   └── SPECS.md                       # Components, routing, state
│
├── Phase_4_Canvas_Charts/
│   ├── PRD.md                         # Chart requirements
│   └── SPECS.md                       # Lightweight-Charts integration
│
├── Phase_5_Feature_Parity/
│   ├── PRD.md                         # Feature requirements
│   └── SPECS.md                       # God Mode, i18n, panels specs
│
└── Phase_6_Testing_Deployment/
    ├── PRD.md                         # QA requirements
    └── SPECS.md                       # Testing strategy, Docker, CI/CD
```

---

## Dependencies Between Phases

```
Phase 0 ─────────┐
                 │
                 ▼
Phase 1 ────────────────┐
(Backend API)           │
                        ▼
Phase 2 ◀───────── Phase 3
(WebSocket)        (Frontend)
     │                  │
     └────────┬─────────┘
              ▼
          Phase 4
        (Charts) ────────┐
                         │
                         ▼
                    Phase 5
                   (Polish) ─────┐
                                 │
                                 ▼
                            Phase 6
                            (Deploy)
```

**Critical Path:** 0 → 1 → 2 → 4 → 5 → 6

**Parallelizable:** Phase 2 and Phase 3 can run in parallel after Phase 1.

---

## Technology Stack Summary

### Backend (Python)
| Component | Technology | Version |
|-----------|------------|---------|
| API Framework | FastAPI | >=0.109.0 |
| ASGI Server | Uvicorn | >=0.27.0 |
| Validation | Pydantic | >=2.5.0 |
| Serialization | msgpack-python | >=1.0.7 |
| Async | asyncio + anyio | Built-in |

### Frontend (TypeScript)
| Component | Technology | Version |
|-----------|------------|---------|
| Framework | React | ^18.2.0 |
| Build Tool | Vite | ^5.0.0 |
| Charts | lightweight-charts | ^4.1.0 |
| State | Zustand | ^4.4.0 |
| Styling | TailwindCSS | ^3.4.0 |
| HTTP | @tanstack/react-query | ^5.0.0 |

### Infrastructure
| Component | Technology | Purpose |
|-----------|------------|---------|
| Containerization | Docker | Deployment |
| Orchestration | Docker Compose | Multi-service |
| Reverse Proxy | Nginx | Static files + API routing |

---

## Success Criteria

### Performance Metrics
- [ ] CTG chart renders at **60 FPS** with 20 patients
- [ ] WebSocket latency < **10ms** (server to browser)
- [ ] Initial page load < **2 seconds**
- [ ] Memory usage < **500MB** per browser tab

### Functional Metrics
- [ ] 100% feature parity with Streamlit V2.0
- [ ] All V2.0 test cases pass on V3.0
- [ ] Hebrew + English language support
- [ ] Mobile-responsive design

### Operational Metrics
- [ ] Single-command deployment (`docker-compose up`)
- [ ] Zero downtime updates supported
- [ ] Centralized logging and monitoring

---

## How to Use This Plan

### For Project Managers
1. Review each Phase's **PRD.md** to understand scope
2. Create JIRA/Linear tickets based on PRD sections
3. Track progress against PRD acceptance criteria

### For Developers
1. Start with **Phase_0_Setup/SPECS.md** for environment setup
2. Follow phases in order (respect dependencies)
3. Each SPECS.md contains exact code examples and file structures

### For QA Engineers
1. Review **Phase_6_Testing_Deployment/PRD.md** for test requirements
2. Each phase's PRD has acceptance criteria for validation
3. Performance benchmarks are in Phase 6 SPECS

---

## Risk Mitigation

| Risk | Probability | Mitigation Strategy |
|------|-------------|---------------------|
| React learning curve | Medium | Start with Phase 0 tutorials, use component libraries |
| WebSocket stability | Low | Implement reconnection logic in Phase 2 |
| Chart performance | Low | Benchmark in Phase 4, have uPlot as fallback |
| Feature regression | Medium | Automated E2E tests in Phase 6 |
| Integration issues | Medium | Continuous integration from Phase 1 |

---

## Getting Started

```bash
# After Phase 0 is complete, the development workflow is:

# Terminal 1: Backend
cd api && uvicorn main:app --reload --port 8000

# Terminal 2: Frontend
cd frontend && npm run dev

# Terminal 3: Original Pipeline (unchanged)
python -m src.simulation.core.orchestrator
```

---

## Document Conventions

### PRD Documents
- **Goal:** Define WHAT needs to be built
- **Audience:** Product managers, stakeholders, QA
- **Format:** User stories, acceptance criteria, non-functional requirements

### SPECS Documents
- **Goal:** Define HOW to build it
- **Audience:** Developers, architects
- **Format:** Code examples, API definitions, file structures, diagrams

---

## Change Log

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-24 | 1.0.0 | Initial plan creation |

---

*Generated by AI Chief Software Architect*
*SentinelFetal V3.0 Migration Project*
