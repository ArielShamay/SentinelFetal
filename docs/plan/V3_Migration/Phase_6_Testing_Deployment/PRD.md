# Phase 6: Testing & Deployment - Product Requirements Document

**Phase:** 6 of 6 (Final)
**Duration:** 3-4 days
**Priority:** Critical - Production Readiness
**Prerequisites:** All Phases 0-5 completed

---

## 1. Executive Summary

Phase 6 delivers the testing infrastructure, containerization, and deployment pipeline that transforms the V3 Full-Stack Migration from a development prototype into a production-ready system. This phase ensures reliability, reproducibility, and safe continuous delivery.

### 1.1 Phase Objectives

| Objective | Description | Metric |
|-----------|-------------|--------|
| E2E Test Coverage | Comprehensive UI flow testing | ≥80% critical path coverage |
| Container Ready | Docker-based deployment | `docker compose up` works first try |
| CI/CD Pipeline | Automated build/test/deploy | PR→Merge→Deploy in <10 minutes |
| Documentation | Complete deployment guide | New dev onboards in <1 hour |

---

## 2. Stakeholders

| Role | Interest | Deliverable |
|------|----------|-------------|
| Developers | Fast feedback loop | CI runs in <5 minutes |
| DevOps | Easy deployment | One-command deploy |
| QA | Regression safety | Automated tests |
| Medical Staff | System reliability | 99.9% uptime target |

---

## 3. Functional Requirements

### 3.1 End-to-End Testing

#### FR-6.1.1: Playwright Test Framework
- **SHALL** use Playwright for browser automation
- **SHALL** support Chrome and Firefox
- **SHALL** run tests in headless mode in CI
- **SHALL** support local headed mode for debugging

#### FR-6.1.2: Critical User Flows
All critical flows **MUST** have E2E test coverage:

| Flow | Description | Priority |
|------|-------------|----------|
| Ward View Load | Start simulation, see all patient tiles | P0 |
| Patient Detail Navigation | Click tile → see detail view → back | P0 |
| Real-time Updates | Verify chart updates continuously | P0 |
| God Mode Injection | Inject event → verify detection | P1 |
| Language Toggle | Switch EN/HE → verify RTL | P1 |
| Simulation Controls | Start/Pause/Resume/Reset cycle | P1 |
| Reconnection | Disconnect WS → auto-reconnect | P2 |

#### FR-6.1.3: Visual Regression
- **SHOULD** capture screenshots of key views
- **SHOULD** compare against baseline images
- **SHOULD** flag visual differences >1%

### 3.2 Container Architecture

#### FR-6.2.1: Backend Container
- **SHALL** use Python 3.11 slim base image
- **SHALL** install dependencies via pip
- **SHALL** expose port 8000
- **SHALL** run via Uvicorn
- **SHALL** support health checks

#### FR-6.2.2: Frontend Container
- **SHALL** use Node 20 for build stage
- **SHALL** use Nginx for serving static files
- **SHALL** expose port 80
- **SHALL** proxy `/api` and `/ws` to backend

#### FR-6.2.3: Compose Orchestration
- **SHALL** define both services in docker-compose.yml
- **SHALL** handle service dependencies
- **SHALL** support volume mounts for development
- **SHALL** support `.env` file for configuration

### 3.3 CI/CD Pipeline

#### FR-6.3.1: GitHub Actions Workflow
- **SHALL** trigger on PR to `main` and `develop`
- **SHALL** trigger on push to `main`
- **SHALL** run linting, type checking, unit tests
- **SHALL** run E2E tests in headless mode
- **SHALL** build Docker images on success

#### FR-6.3.2: Deployment (Optional)
- **MAY** deploy to staging on PR merge
- **MAY** deploy to production on release tag
- **MAY** integrate with cloud provider (AWS/Azure/GCP)

### 3.4 Documentation

#### FR-6.4.1: Developer Documentation
- **SHALL** include README with quick start
- **SHALL** include architecture overview
- **SHALL** include API documentation (auto-generated)
- **SHALL** include troubleshooting guide

#### FR-6.4.2: Operations Documentation
- **SHALL** include deployment instructions
- **SHALL** include environment variables reference
- **SHALL** include monitoring and logging guide
- **SHALL** include backup and recovery procedures

---

## 4. Non-Functional Requirements

### 4.1 Performance
| Metric | Requirement |
|--------|-------------|
| CI Pipeline Duration | <5 minutes for PR checks |
| Docker Build Time | <3 minutes backend, <2 minutes frontend |
| E2E Test Suite | <3 minutes total |
| Container Startup | <10 seconds to healthy |

### 4.2 Reliability
| Metric | Requirement |
|--------|-------------|
| CI Success Rate | >95% for passing code |
| Test Flakiness | <2% flaky test rate |
| Container Restart | Auto-restart on failure |

### 4.3 Security
| Requirement | Implementation |
|-------------|----------------|
| No secrets in code | Use environment variables |
| Container scanning | Trivy in CI pipeline |
| Dependency auditing | npm audit, safety check |
| HTTPS in production | Nginx SSL termination |

---

## 5. Test Scenarios

### 5.1 E2E Test Scenarios

#### TC-6.1: Simulation Start Flow
```gherkin
Feature: Simulation Start
  Scenario: User starts simulation and sees patient data
    Given the application is loaded
    When I click the "Start" button
    Then I should see the simulation status as "Running"
    And I should see patient tiles appear within 5 seconds
    And each tile should show FHR data updating
```

#### TC-6.2: Patient Detail Navigation
```gherkin
Feature: Patient Detail
  Scenario: User views patient details
    Given the simulation is running with 4 patients
    When I click on patient "P001" tile
    Then I should see the patient detail view
    And I should see the full CTG chart
    And I should see the trend panel
    When I click the back button
    Then I should return to the ward view
```

#### TC-6.3: God Mode Event Injection
```gherkin
Feature: God Mode
  Scenario: User injects a late deceleration
    Given the simulation is running
    And I am viewing patient "P001" details
    When I open the God Mode panel
    And I select "Late Deceleration" event type
    And I select "Moderate" severity
    And I click "Inject Event"
    Then I should see a success notification
    And within 20 seconds I should see category change to 2 or 3
```

#### TC-6.4: Language Toggle
```gherkin
Feature: Language Toggle
  Scenario: User switches to Hebrew
    Given the application is loaded in English
    When I click the language toggle button
    Then the page should switch to Hebrew
    And the layout should become RTL
    And all text should be in Hebrew
```

### 5.2 Container Test Scenarios

#### TC-6.5: Docker Compose Startup
```bash
# Test scenario: Fresh environment startup
docker compose down -v  # Clean slate
docker compose up -d    # Start services
curl -f http://localhost:8000/api/health  # Backend health
curl -f http://localhost/                 # Frontend loads
```

#### TC-6.6: Container Restart Recovery
```bash
# Test scenario: Backend restart
docker compose restart backend
sleep 5
curl -f http://localhost:8000/api/health  # Should recover
```

---

## 6. Deliverables

### 6.1 Files to Create

| File | Description |
|------|-------------|
| `e2e/playwright.config.ts` | Playwright configuration |
| `e2e/tests/ward.spec.ts` | Ward view tests |
| `e2e/tests/detail.spec.ts` | Patient detail tests |
| `e2e/tests/godmode.spec.ts` | God Mode tests |
| `e2e/tests/i18n.spec.ts` | Language toggle tests |
| `backend/Dockerfile` | Backend container |
| `frontend/Dockerfile` | Frontend container |
| `docker-compose.yml` | Compose orchestration |
| `.github/workflows/ci.yml` | CI pipeline |
| `.github/workflows/deploy.yml` | CD pipeline (optional) |
| `docs/DEPLOYMENT.md` | Deployment guide |
| `docs/TROUBLESHOOTING.md` | Common issues and fixes |

### 6.2 Commands to Support

```bash
# Development
npm run test:e2e          # Run E2E tests locally
npm run test:e2e:headed   # Run E2E tests with browser visible
npm run test:e2e:debug    # Debug E2E tests

# Docker
docker compose up         # Start all services
docker compose up -d      # Start detached
docker compose logs -f    # View logs
docker compose down       # Stop services

# CI (runs automatically)
# - Lint check
# - Type check
# - Unit tests
# - E2E tests
# - Docker build
```

---

## 7. Acceptance Criteria

### 7.1 E2E Testing
- [x] Playwright configured with Chrome and Firefox ✅
- [x] Ward view tests created ✅
- [x] Simulation control tests created ✅
- [x] Patient detail tests created ✅
- [ ] All P0 test scenarios passing (requires running backend)
- [ ] Tests run in <3 minutes
- [ ] No flaky tests (3 consecutive runs)

### 7.2 Containerization
- [x] Dockerfile.backend created ✅
- [x] Dockerfile.frontend created ✅
- [x] nginx.conf created ✅
- [x] docker-compose.yml created ✅
- [ ] `docker compose up` starts system (needs testing)
- [ ] Health checks passing
- [ ] Frontend accessible at `http://localhost`
- [ ] Backend API accessible at `http://localhost/api`
- [ ] WebSocket connects at `ws://localhost/ws/stream`

### 7.3 CI/CD Pipeline
- [x] GitHub Actions workflow updated ✅
- [x] E2E test job added ✅
- [x] Docker build job added ✅
- [ ] All checks pass for clean code
- [ ] Docker images build successfully
- [ ] Pipeline completes in <5 minutes

### 7.4 Documentation
- [ ] README quick start works
- [ ] API docs auto-generated
- [ ] Deployment guide complete
- [ ] Troubleshooting covers common issues

---

## 13. Implementation Status

### Completed (2025-01-XX)
- **Playwright Setup**: playwright.config.ts with Chrome/Firefox/Mobile
- **E2E Tests**: ward-view.spec.ts, simulation-controls.spec.ts, patient-detail.spec.ts
- **Docker Backend**: Multi-stage Python 3.12-slim, health checks, non-root user
- **Docker Frontend**: Node 20 build, Nginx serve, health checks
- **Nginx Config**: API/WebSocket proxy, gzip, caching, SPA fallback
- **Docker Compose**: Full stack with health checks and networking
- **GitHub Actions**: Updated with E2E and Docker jobs

### Build Status
- **Frontend**: 474KB (gzipped: 151KB)
- **E2E Framework**: Playwright with 3 test files
- **Docker**: Configs created, not yet tested

### Pending
- [ ] Run E2E tests with backend running
- [ ] Test Docker builds
- [ ] Test docker-compose up
- [ ] Update README with quick start

---

## 8. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Flaky E2E tests | Medium | Use explicit waits, retry mechanism |
| Docker build failures | High | Multi-stage builds, cache optimization |
| CI timeout | Medium | Parallelize tests, optimize steps |
| Security vulnerabilities | High | Automated scanning, regular updates |

---

## 9. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Test Coverage | ≥80% critical paths | Playwright report |
| CI Pass Rate | ≥95% | GitHub Actions history |
| Deploy Time | <10 minutes | Workflow duration |
| Onboarding Time | <1 hour | New dev feedback |
| Container Uptime | 99.9% | Monitoring dashboard |

---

## 10. Timeline

| Day | Activity |
|-----|----------|
| Day 1 | Playwright setup, first E2E tests |
| Day 2 | Complete E2E test suite |
| Day 3 | Docker containers, docker-compose |
| Day 4 | CI/CD pipeline, documentation |

---

## 11. Dependencies

### 11.1 From Previous Phases
- Phase 1: Backend API running
- Phase 2: WebSocket streaming working
- Phase 3: Frontend application built
- Phase 4: Charts rendering correctly
- Phase 5: All features complete

### 11.2 External Dependencies
- GitHub Actions (free tier sufficient)
- Docker Hub (optional, for image hosting)
- Playwright browsers (auto-installed)

---

## 12. Post-Phase Activities

After Phase 6 completion:
1. **Performance Testing** - Load test with multiple clients
2. **Security Audit** - Penetration testing
3. **User Acceptance** - Medical staff validation
4. **Production Deploy** - Go-live planning

---

*End of Phase 6 PRD*
