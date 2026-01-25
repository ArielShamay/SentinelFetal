# Phase 3: Frontend Scaffold (React + Vite) - PRD

**Phase:** 3 of 6
**Duration:** 3-4 days
**Priority:** High
**Risk Level:** Low
**Dependencies:** Phase 0 Complete, Phase 2 Parallel

---

## ✅ IMPLEMENTATION STATUS: COMPLETE

**Implemented:** January 2026
**Developer:** AI Assistant (Claude)

### Implementation Summary:
Phase 3 has been fully implemented with all deliverables completed. The React frontend scaffold is now complete with:
- Full TypeScript types matching API schemas
- Zustand stores for patient and UI state (with persist middleware)
- WebSocket integration with auto-reconnect
- REST API client for all endpoints
- Complete component library (Layout, Header, PatientCard, CategoryBadge, etc.)
- WardView with filtering, sorting, and search
- DetailView with vitals panel and events display
- React Query for server state management

### Files Created:
- `frontend/src/types/index.ts` - Comprehensive TypeScript types
- `frontend/src/stores/patientStore.ts` - Zustand patient state with Map
- `frontend/src/stores/uiStore.ts` - UI preferences with persist
- `frontend/src/services/api.ts` - REST API client
- `frontend/src/services/websocket.ts` - WebSocket manager with reconnect
- `frontend/src/hooks/usePatientStream.ts` - WebSocket connection hook
- `frontend/src/hooks/useSimulation.ts` - Simulation control hook
- `frontend/src/components/layout/Layout.tsx` - Main layout with Outlet
- `frontend/src/components/layout/Header.tsx` - Header with controls
- `frontend/src/components/status/ConnectionStatus.tsx` - Connection indicator
- `frontend/src/components/status/SimulationControls.tsx` - Start/stop/pause
- `frontend/src/components/patient/PatientCard.tsx` - Patient grid card
- `frontend/src/components/patient/CategoryBadge.tsx` - FIGO category badge
- `frontend/src/pages/WardView.tsx` - Multi-patient grid with toolbar
- `frontend/src/pages/DetailView.tsx` - Patient detail with vitals
- `frontend/src/App.tsx` - Updated with routes and providers
- `frontend/src/main.tsx` - Simplified entry point
- `frontend/tailwind.config.js` - Extended with custom animations

### Deviations from Plan:
1. Added `useSimulation` hook beyond spec for simulation control
2. Enhanced WardView with search and advanced filtering
3. Added signal quality visualization in PatientCard
4. Added CategoryBadge utility functions (getCategoryPriority, sortByCategory)

---

## 1. Overview

### 1.1 Purpose
Phase 3 establishes the React frontend application structure, including routing, state management, and WebSocket integration. This phase creates the foundation upon which the CTG charts (Phase 4) and features (Phase 5) will be built.

### 1.2 Goals
1. Create production-ready React application with Vite
2. Implement routing for Ward View and Detail View
3. Set up Zustand state management
4. Create WebSocket connection hook
5. Build basic UI layout and navigation

### 1.3 Non-Goals
- CTG chart implementation (Phase 4)
- Hebrew localization (Phase 5)
- God Mode controls (Phase 5)
- Production deployment (Phase 6)

---

## 2. User Stories

### US-3.1: Application Shell
**As a** user
**I want** a responsive application shell
**So that** I can navigate between views

**Acceptance Criteria:**
- [ ] Header with application title and status indicator
- [ ] Navigation between Ward View and Detail View
- [ ] Responsive design works on desktop and tablet
- [ ] Dark/light mode toggle (optional)

### US-3.2: Ward View Layout
**As a** clinician
**I want** to see all patients in a grid layout
**So that** I can monitor the entire ward at a glance

**Acceptance Criteria:**
- [ ] Grid layout displays up to 20 patient cards
- [ ] Each card shows patient ID, category, and key metrics
- [ ] Cards are color-coded by category (green/orange/red)
- [ ] Clicking a card navigates to Detail View
- [ ] Grid is responsive (4 columns on large screens, 2 on mobile)

### US-3.3: Detail View Layout
**As a** clinician
**I want** a focused view for a single patient
**So that** I can analyze their CTG trace in detail

**Acceptance Criteria:**
- [ ] Large area for CTG chart (placeholder in Phase 3)
- [ ] Side panel with patient info and metrics
- [ ] Back button to return to Ward View
- [ ] Patient selector to switch without going back

### US-3.4: Real-Time Connection
**As a** frontend application
**I want** to connect to the WebSocket stream
**So that** I can display live patient data

**Acceptance Criteria:**
- [ ] WebSocket connection established on app load
- [ ] Connection status indicator (connected/reconnecting/error)
- [ ] Automatic reconnection on disconnect
- [ ] Patient data updates reflected in state

### US-3.5: State Management
**As a** developer
**I want** centralized state management
**So that** components can access patient data efficiently

**Acceptance Criteria:**
- [ ] Zustand store holds all patient data
- [ ] Store updates trigger minimal re-renders
- [ ] Selected patient ID persisted in URL
- [ ] Simulation status available globally

---

## 3. Functional Requirements

### FR-3.1: Application Routes

| Route | Component | Purpose |
|-------|-----------|---------|
| `/` | WardView | Multi-patient monitoring grid |
| `/patient/:id` | DetailView | Single patient focus |
| `/settings` | SettingsView | Configuration (optional) |

### FR-3.2: Component Hierarchy

```
App
├── Header
│   ├── Logo
│   ├── ConnectionStatus
│   └── Navigation
├── Main (React Router Outlet)
│   ├── WardView
│   │   └── PatientCard (x N)
│   └── DetailView
│       ├── PatientInfo
│       ├── CTGChartPlaceholder
│       └── EventsPanel
└── Footer (optional)
```

### FR-3.3: State Shape

```typescript
interface AppState {
  // Connection
  connected: boolean;
  lastUpdate: number;

  // Simulation
  simulationRunning: boolean;
  simulationPaused: boolean;

  // Patients
  patients: Map<string, PatientData>;
  selectedPatientId: string | null;

  // UI
  viewMode: 'ward' | 'detail';
}

interface PatientData {
  patient_id: string;
  category: 1 | 2 | 3;
  baseline: number;
  variability: number;
  fhr_latest: number[];
  uc_latest: number[];
  mhr_alert: boolean;
  highlight_regions: HighlightRegion[];
  trend_score: number;
  active_event: string | null;
}
```

### FR-3.4: WebSocket Hook API

```typescript
// Hook usage
const { patients, connected, reconnect } = usePatientStream();

// Connection lifecycle
// 1. Connect on mount
// 2. Reconnect on disconnect (max 5 retries)
// 3. Exponential backoff (1s, 2s, 4s, 8s, 16s)
// 4. Manual reconnect available
```

### FR-3.5: Patient Card Display

Each patient card shall display:
- Patient ID (e.g., "Patient-1")
- Category badge with color
- Baseline FHR (bpm)
- Variability (bpm)
- Active event indicator (if any)
- MHR alert badge (if applicable)

---

## 4. Non-Functional Requirements

### NFR-3.1: Performance
- Initial page load < 2 seconds
- State updates < 16ms (60fps capable)
- Bundle size < 500KB gzipped

### NFR-3.2: Accessibility
- All interactive elements keyboard accessible
- ARIA labels on status indicators
- Color not sole means of conveying category

### NFR-3.3: Browser Support
- Chrome 90+
- Firefox 90+
- Safari 14+
- Edge 90+

### NFR-3.4: Code Quality
- TypeScript strict mode
- ESLint with no warnings
- Component tests for key flows

---

## 5. Technical Design Decisions

### 5.1 State Management: Zustand

**Why Zustand over Redux/Context:**
- Minimal boilerplate
- No provider wrapping needed
- Built-in devtools
- Excellent TypeScript support
- Works well with WebSocket updates

### 5.2 Styling: TailwindCSS

**Why Tailwind:**
- Utility-first for rapid development
- Small production bundle (purged)
- Easy responsive design
- Consistent design tokens

### 5.3 HTTP Client: React Query

**For REST endpoints:**
- Automatic caching
- Background refetching
- Loading/error states
- DevTools integration

---

## 6. UI Design Specifications

### 6.1 Color Palette

| Usage | Color | Hex |
|-------|-------|-----|
| Category 1 (Normal) | Green | #28a745 |
| Category 2 (Intermediate) | Orange | #fd7e14 |
| Category 3 (Pathological) | Red | #dc3545 |
| FHR Line | Blue | #1E90FF |
| UC Line | Orange | #FF8C00 |
| Background | Light Gray | #f8f9fa |
| Text Primary | Dark Gray | #212529 |

### 6.2 Typography

- Font Family: Inter, system-ui
- Headings: Semi-bold
- Body: Regular
- Monospace (metrics): JetBrains Mono

### 6.3 Spacing

- Base unit: 4px
- Component padding: 16px (4 units)
- Grid gap: 24px (6 units)

---

## 7. Dependencies

### 7.1 External Dependencies
```json
{
  "react": "^18.2.0",
  "react-dom": "^18.2.0",
  "react-router-dom": "^6.21.0",
  "zustand": "^4.4.0",
  "@tanstack/react-query": "^5.17.0",
  "@ygoe/msgpack": "^1.0.0"
}
```

### 7.2 Dev Dependencies
```json
{
  "vite": "^5.0.0",
  "typescript": "^5.3.0",
  "tailwindcss": "^3.4.0",
  "@types/react": "^18.2.0"
}
```

### 7.3 Internal Dependencies
- Phase 0 complete (project structure)
- Phase 2 WebSocket endpoints (can develop in parallel with mocks)

---

## 8. Acceptance Criteria Summary

| ID | Criteria | Verification Method |
|----|----------|---------------------|
| AC-3.1 | App loads without errors | Browser console |
| AC-3.2 | Ward view displays patient grid | Visual inspection |
| AC-3.3 | Clicking card navigates to detail | Manual test |
| AC-3.4 | Back navigation works | Manual test |
| AC-3.5 | WebSocket connection indicator accurate | Network tab |
| AC-3.6 | Patient data updates in real-time | With Phase 2 running |
| AC-3.7 | Responsive on mobile viewport | Chrome DevTools |
| AC-3.8 | No TypeScript errors | `npm run type-check` |

---

## 9. Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| React learning curve | Low | Medium | Use well-documented patterns |
| MessagePack browser support | Low | Low | Fallback to JSON |
| WebSocket reliability | Medium | Medium | Robust reconnection logic |
| State update performance | Low | Medium | Zustand selectors |

---

## 10. Timeline

| Day | Tasks |
|-----|-------|
| Day 1 | Vite setup, routing, Tailwind config |
| Day 2 | Zustand store, WebSocket hook |
| Day 3 | WardView, PatientCard components |
| Day 4 | DetailView layout, testing |

---

## 11. Deliverables Checklist

- [x] `frontend/src/App.tsx` - Root component with router
- [x] `frontend/src/pages/WardView.tsx` - Ward grid page
- [x] `frontend/src/pages/DetailView.tsx` - Patient detail page
- [x] `frontend/src/components/PatientCard.tsx` - Card component (in patient folder)
- [x] `frontend/src/components/Header.tsx` - App header (in layout folder)
- [x] `frontend/src/components/ConnectionStatus.tsx` - WebSocket indicator (in status folder)
- [x] `frontend/src/hooks/usePatientStream.ts` - WebSocket hook
- [x] `frontend/src/stores/patientStore.ts` - Zustand store
- [x] `frontend/src/types/index.ts` - TypeScript types
- [x] Responsive layout verified

### Additional Deliverables (Beyond Original Spec):
- [x] `frontend/src/stores/uiStore.ts` - UI preferences store
- [x] `frontend/src/services/api.ts` - REST API client
- [x] `frontend/src/services/websocket.ts` - WebSocket manager
- [x] `frontend/src/hooks/useSimulation.ts` - Simulation control hook
- [x] `frontend/src/components/status/SimulationControls.tsx` - Control buttons
- [x] `frontend/src/components/patient/CategoryBadge.tsx` - Category display
- [x] `frontend/src/components/layout/Layout.tsx` - Layout wrapper

---

*End of Phase 3 PRD*
