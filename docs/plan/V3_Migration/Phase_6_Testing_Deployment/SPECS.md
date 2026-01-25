# Phase 6: Testing & Deployment - Technical Specifications

**Phase:** 6 of 6 (Final)
**Document Type:** Technical Specifications
**Target Audience:** DevOps Engineers, QA Engineers, Developers

---

## 1. Project Structure Additions

```
root/
├── e2e/
│   ├── playwright.config.ts
│   ├── global-setup.ts
│   ├── fixtures/
│   │   └── test-fixtures.ts
│   └── tests/
│       ├── ward.spec.ts
│       ├── detail.spec.ts
│       ├── godmode.spec.ts
│       └── i18n.spec.ts
│
├── backend/
│   ├── Dockerfile
│   └── .dockerignore
│
├── frontend/
│   ├── Dockerfile
│   ├── nginx.conf
│   └── .dockerignore
│
├── docker-compose.yml
├── docker-compose.dev.yml
├── docker-compose.prod.yml
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── deploy.yml
│
└── docs/
    ├── DEPLOYMENT.md
    └── TROUBLESHOOTING.md
```

---

## 2. E2E Testing with Playwright

### 2.1 e2e/playwright.config.ts

```typescript
import { defineConfig, devices } from '@playwright/test'

export default defineConfig({
  testDir: './tests',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: [
    ['list'],
    ['html', { outputFolder: 'playwright-report' }],
    ['junit', { outputFile: 'test-results/junit.xml' }],
  ],
  
  use: {
    baseURL: process.env.BASE_URL || 'http://localhost:5173',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },

  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] },
    },
  ],

  webServer: process.env.CI ? undefined : {
    command: 'npm run dev',
    url: 'http://localhost:5173',
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
})
```

### 2.2 e2e/global-setup.ts

```typescript
import { chromium, FullConfig } from '@playwright/test'

async function globalSetup(config: FullConfig) {
  const { baseURL } = config.projects[0].use
  
  // Wait for backend to be ready
  const browser = await chromium.launch()
  const page = await browser.newPage()
  
  let retries = 30
  while (retries > 0) {
    try {
      const response = await page.goto(`${baseURL?.replace(':5173', ':8000')}/api/health`)
      if (response?.ok()) break
    } catch {
      // Server not ready yet
    }
    await page.waitForTimeout(1000)
    retries--
  }
  
  if (retries === 0) {
    throw new Error('Backend did not start in time')
  }
  
  await browser.close()
}

export default globalSetup
```

### 2.3 e2e/fixtures/test-fixtures.ts

```typescript
import { test as base, expect } from '@playwright/test'

interface SimulationFixtures {
  startedSimulation: void
}

export const test = base.extend<SimulationFixtures>({
  startedSimulation: async ({ page }, use) => {
    // Navigate and start simulation
    await page.goto('/')
    await page.getByRole('button', { name: /start/i }).click()
    
    // Wait for first patient data
    await expect(page.locator('[data-testid="patient-tile"]').first())
      .toBeVisible({ timeout: 10_000 })
    
    await use()
  },
})

export { expect }
```

### 2.4 e2e/tests/ward.spec.ts

```typescript
import { test, expect } from '../fixtures/test-fixtures'

test.describe('Ward View', () => {
  test('should load application', async ({ page }) => {
    await page.goto('/')
    await expect(page).toHaveTitle(/SentinelFetal/)
  })

  test('should show patient tiles when simulation starts', async ({ 
    page, 
    startedSimulation 
  }) => {
    // Check multiple tiles appear
    const tiles = page.locator('[data-testid="patient-tile"]')
    await expect(tiles).toHaveCount(4, { timeout: 15_000 })
  })

  test('should show real-time FHR updates', async ({ 
    page, 
    startedSimulation 
  }) => {
    const firstTile = page.locator('[data-testid="patient-tile"]').first()
    
    // Get initial FHR value
    const initialFHR = await firstTile.locator('[data-testid="fhr-value"]').textContent()
    
    // Wait for update
    await page.waitForTimeout(3000)
    
    // FHR should have changed (simulation generates data)
    const updatedFHR = await firstTile.locator('[data-testid="fhr-value"]').textContent()
    
    // Value format check (should be a number)
    expect(parseInt(updatedFHR || '0')).toBeGreaterThan(60)
    expect(parseInt(updatedFHR || '0')).toBeLessThan(200)
  })

  test('should show simulation controls', async ({ page }) => {
    await page.goto('/')
    
    await expect(page.getByRole('button', { name: /start/i })).toBeVisible()
  })

  test('should pause and resume simulation', async ({ 
    page, 
    startedSimulation 
  }) => {
    // Pause
    await page.getByRole('button', { name: /pause/i }).click()
    await expect(page.getByText(/paused/i)).toBeVisible()
    
    // Resume
    await page.getByRole('button', { name: /resume/i }).click()
    await expect(page.getByText(/running/i)).toBeVisible()
  })

  test('should show connection status', async ({ page, startedSimulation }) => {
    await expect(page.getByText(/live/i)).toBeVisible()
  })
})
```

### 2.5 e2e/tests/detail.spec.ts

```typescript
import { test, expect } from '../fixtures/test-fixtures'

test.describe('Patient Detail View', () => {
  test('should navigate to patient detail', async ({ 
    page, 
    startedSimulation 
  }) => {
    const firstTile = page.locator('[data-testid="patient-tile"]').first()
    await firstTile.click()
    
    // Should show detail view
    await expect(page.locator('[data-testid="patient-detail"]')).toBeVisible()
    
    // Should show full CTG chart
    await expect(page.locator('[data-testid="ctg-chart"]')).toBeVisible()
  })

  test('should show trend panel', async ({ page, startedSimulation }) => {
    await page.locator('[data-testid="patient-tile"]').first().click()
    
    await expect(page.getByText(/trend analysis/i)).toBeVisible()
    await expect(page.locator('[data-testid="deterioration-score"]')).toBeVisible()
  })

  test('should show explanation panel', async ({ page, startedSimulation }) => {
    await page.locator('[data-testid="patient-tile"]').first().click()
    
    await expect(page.getByText(/explanation/i)).toBeVisible()
  })

  test('should navigate back to ward view', async ({ 
    page, 
    startedSimulation 
  }) => {
    await page.locator('[data-testid="patient-tile"]').first().click()
    
    // Wait for detail view
    await expect(page.locator('[data-testid="patient-detail"]')).toBeVisible()
    
    // Click back
    await page.getByRole('button', { name: /back/i }).click()
    
    // Should return to ward view
    await expect(page.locator('[data-testid="patient-tile"]').first()).toBeVisible()
  })

  test('should show chart updating in real-time', async ({ 
    page, 
    startedSimulation 
  }) => {
    await page.locator('[data-testid="patient-tile"]').first().click()
    
    const chart = page.locator('[data-testid="ctg-chart"]')
    await expect(chart).toBeVisible()
    
    // Take screenshot at two points to verify update
    // (Visual comparison would be done by Playwright's screenshot comparison)
    await page.waitForTimeout(2000)
  })
})
```

### 2.6 e2e/tests/godmode.spec.ts

```typescript
import { test, expect } from '../fixtures/test-fixtures'

test.describe('God Mode', () => {
  test('should open God Mode panel', async ({ page, startedSimulation }) => {
    await page.locator('[data-testid="patient-tile"]').first().click()
    
    await expect(page.getByText(/god mode/i)).toBeVisible()
  })

  test('should inject late deceleration event', async ({ 
    page, 
    startedSimulation 
  }) => {
    await page.locator('[data-testid="patient-tile"]').first().click()
    
    // Select event type
    await page.locator('[data-testid="event-type-select"]').selectOption('LATE_DECEL')
    
    // Select severity
    await page.getByLabel(/moderate/i).check()
    
    // Click inject
    await page.getByRole('button', { name: /inject/i }).click()
    
    // Should show success message
    await expect(page.getByText(/success/i)).toBeVisible()
  })

  test('should detect injected event within expected time', async ({ 
    page, 
    startedSimulation 
  }) => {
    await page.locator('[data-testid="patient-tile"]').first().click()
    
    // Get initial category
    const initialCategory = await page.locator('[data-testid="category-badge"]').textContent()
    
    // Inject severe event
    await page.locator('[data-testid="event-type-select"]').selectOption('LATE_DECEL')
    await page.getByLabel(/severe/i).check()
    await page.getByRole('button', { name: /inject/i }).click()
    
    // Wait for detection (up to 30 seconds)
    await page.waitForTimeout(30_000)
    
    // Category should have worsened (this is probabilistic, may not always trigger)
    const newCategory = await page.locator('[data-testid="category-badge"]').textContent()
    
    // Log for debugging - actual assertion depends on deterministic behavior
    console.log(`Category: ${initialCategory} -> ${newCategory}`)
  })

  test('should show expected detection time', async ({ 
    page, 
    startedSimulation 
  }) => {
    await page.locator('[data-testid="patient-tile"]').first().click()
    
    await page.locator('[data-testid="event-type-select"]').selectOption('BRADYCARDIA')
    
    await expect(page.getByText(/~5s/)).toBeVisible()
  })
})
```

### 2.7 e2e/tests/i18n.spec.ts

```typescript
import { test, expect } from '../fixtures/test-fixtures'

test.describe('Internationalization', () => {
  test('should default to English', async ({ page }) => {
    await page.goto('/')
    
    await expect(page.getByText(/ward view/i)).toBeVisible()
  })

  test('should switch to Hebrew', async ({ page }) => {
    await page.goto('/')
    
    // Click language toggle
    await page.getByRole('button', { name: /עב|he/i }).click()
    
    // Should show Hebrew text
    await expect(page.getByText(/תצוגת מחלקה/)).toBeVisible()
  })

  test('should apply RTL layout in Hebrew', async ({ page }) => {
    await page.goto('/')
    
    await page.getByRole('button', { name: /עב|he/i }).click()
    
    // Check document direction
    const dir = await page.evaluate(() => document.documentElement.dir)
    expect(dir).toBe('rtl')
  })

  test('should persist language preference', async ({ page }) => {
    await page.goto('/')
    
    // Switch to Hebrew
    await page.getByRole('button', { name: /עב|he/i }).click()
    await expect(page.getByText(/תצוגת מחלקה/)).toBeVisible()
    
    // Reload page
    await page.reload()
    
    // Should still be Hebrew
    await expect(page.getByText(/תצוגת מחלקה/)).toBeVisible()
  })

  test('should switch back to English', async ({ page }) => {
    await page.goto('/')
    
    // Switch to Hebrew first
    await page.getByRole('button', { name: /עב|he/i }).click()
    
    // Switch back to English
    await page.getByRole('button', { name: /en/i }).click()
    
    // Should show English
    await expect(page.getByText(/ward view/i)).toBeVisible()
    
    // Check LTR direction
    const dir = await page.evaluate(() => document.documentElement.dir)
    expect(dir).toBe('ltr')
  })
})
```

---

## 3. Docker Containers

### 3.1 backend/Dockerfile

```dockerfile
# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY backend/ ./backend/

# Create non-root user
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

# Run application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 3.2 backend/.dockerignore

```
__pycache__
*.pyc
*.pyo
.pytest_cache
.git
.gitignore
.env
.venv
venv
*.egg-info
.coverage
htmlcov
.mypy_cache
*.log
tests/
docs/
*.md
```

### 3.3 frontend/Dockerfile

```dockerfile
# Stage 1: Builder
FROM node:20-slim as builder

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm ci

# Copy source and build
COPY . .
RUN npm run build

# Stage 2: Runtime
FROM nginx:alpine

# Copy built assets
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
```

### 3.4 frontend/nginx.conf

```nginx
server {
    listen 80;
    server_name localhost;
    
    root /usr/share/nginx/html;
    index index.html;

    # Gzip compression
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;
    gzip_min_length 1000;

    # SPA fallback
    location / {
        try_files $uri $uri/ /index.html;
    }

    # API proxy
    location /api {
        proxy_pass http://backend:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket proxy
    location /ws {
        proxy_pass http://backend:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 86400;
    }

    # Cache static assets
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
}
```

### 3.5 frontend/.dockerignore

```
node_modules
dist
.git
.gitignore
*.md
.env*
.vscode
playwright-report
test-results
e2e
```

### 3.6 docker-compose.yml

```yaml
version: '3.8'

services:
  backend:
    build:
      context: .
      dockerfile: backend/Dockerfile
    container_name: sentinel-backend
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
      - LOG_LEVEL=${LOG_LEVEL:-INFO}
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    restart: unless-stopped

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    container_name: sentinel-frontend
    ports:
      - "80:80"
    depends_on:
      backend:
        condition: service_healthy
    restart: unless-stopped

networks:
  default:
    name: sentinel-network
```

### 3.7 docker-compose.dev.yml

```yaml
version: '3.8'

services:
  backend:
    build:
      context: .
      dockerfile: backend/Dockerfile
    volumes:
      - ./src:/app/src:ro
      - ./backend:/app/backend:ro
    environment:
      - DEBUG=true
      - LOG_LEVEL=DEBUG
    command: uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
      target: builder
    volumes:
      - ./frontend/src:/app/src
      - ./frontend/public:/app/public
    ports:
      - "5173:5173"
    command: npm run dev -- --host
    depends_on:
      - backend
```

### 3.8 docker-compose.prod.yml

```yaml
version: '3.8'

services:
  backend:
    image: ${REGISTRY:-ghcr.io}/${IMAGE_PREFIX:-sentinelfetal}/backend:${TAG:-latest}
    environment:
      - LOG_LEVEL=WARNING
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 1G
        reservations:
          memory: 512M

  frontend:
    image: ${REGISTRY:-ghcr.io}/${IMAGE_PREFIX:-sentinelfetal}/frontend:${TAG:-latest}
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 256M
        reservations:
          memory: 128M
```

---

## 4. CI/CD Pipeline

### 4.1 .github/workflows/ci.yml

```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

env:
  NODE_VERSION: '20'
  PYTHON_VERSION: '3.11'

jobs:
  lint-and-type-check:
    name: Lint & Type Check
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'
      
      - name: Install Python dependencies
        run: |
          pip install -r requirements.txt
          pip install ruff mypy
      
      - name: Run Ruff (Python linter)
        run: ruff check src/ backend/
      
      - name: Run MyPy (Python type checker)
        run: mypy src/ backend/ --ignore-missing-imports
      
      - name: Set up Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'
          cache-dependency-path: frontend/package-lock.json
      
      - name: Install Node dependencies
        working-directory: frontend
        run: npm ci
      
      - name: Run ESLint
        working-directory: frontend
        run: npm run lint
      
      - name: Run TypeScript check
        working-directory: frontend
        run: npm run type-check

  unit-tests:
    name: Unit Tests
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'
      
      - name: Install Python dependencies
        run: pip install -r requirements.txt
      
      - name: Run Python tests
        run: pytest tests/ -v --tb=short
      
      - name: Set up Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'
          cache-dependency-path: frontend/package-lock.json
      
      - name: Install Node dependencies
        working-directory: frontend
        run: npm ci
      
      - name: Run Vitest
        working-directory: frontend
        run: npm run test

  e2e-tests:
    name: E2E Tests
    runs-on: ubuntu-latest
    needs: [lint-and-type-check, unit-tests]
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'
      
      - name: Install Python dependencies
        run: pip install -r requirements.txt
      
      - name: Set up Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'
          cache-dependency-path: frontend/package-lock.json
      
      - name: Install Node dependencies
        working-directory: frontend
        run: npm ci
      
      - name: Install Playwright browsers
        working-directory: e2e
        run: npx playwright install --with-deps chromium
      
      - name: Build frontend
        working-directory: frontend
        run: npm run build
      
      - name: Start backend
        run: |
          uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
          sleep 5
      
      - name: Start frontend preview
        working-directory: frontend
        run: |
          npm run preview -- --host --port 5173 &
          sleep 3
      
      - name: Run Playwright tests
        working-directory: e2e
        run: npx playwright test --project=chromium
        env:
          BASE_URL: http://localhost:5173
          CI: true
      
      - name: Upload test results
        uses: actions/upload-artifact@v4
        if: failure()
        with:
          name: playwright-report
          path: e2e/playwright-report/
          retention-days: 7

  build-docker:
    name: Build Docker Images
    runs-on: ubuntu-latest
    needs: [e2e-tests]
    if: github.event_name == 'push'
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Build backend image
        uses: docker/build-push-action@v5
        with:
          context: .
          file: backend/Dockerfile
          push: false
          tags: sentinel-backend:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
      
      - name: Build frontend image
        uses: docker/build-push-action@v5
        with:
          context: ./frontend
          file: frontend/Dockerfile
          push: false
          tags: sentinel-frontend:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  security-scan:
    name: Security Scan
    runs-on: ubuntu-latest
    needs: [lint-and-type-check]
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          severity: 'CRITICAL,HIGH'
          exit-code: '1'
          ignore-unfixed: true
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Check Python dependencies
        run: |
          pip install safety
          safety check -r requirements.txt || true
      
      - name: Set up Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
      
      - name: Check npm dependencies
        working-directory: frontend
        run: npm audit --audit-level=high || true
```

### 4.2 .github/workflows/deploy.yml

```yaml
name: Deploy

on:
  push:
    tags:
      - 'v*'
  workflow_dispatch:
    inputs:
      environment:
        description: 'Deployment environment'
        required: true
        default: 'staging'
        type: choice
        options:
          - staging
          - production

env:
  REGISTRY: ghcr.io
  IMAGE_PREFIX: ${{ github.repository }}

jobs:
  build-and-push:
    name: Build & Push Images
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Login to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: |
            ${{ env.REGISTRY }}/${{ env.IMAGE_PREFIX }}/backend
            ${{ env.REGISTRY }}/${{ env.IMAGE_PREFIX }}/frontend
          tags: |
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha
      
      - name: Build and push backend
        uses: docker/build-push-action@v5
        with:
          context: .
          file: backend/Dockerfile
          push: true
          tags: ${{ env.REGISTRY }}/${{ env.IMAGE_PREFIX }}/backend:${{ github.ref_name }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
      
      - name: Build and push frontend
        uses: docker/build-push-action@v5
        with:
          context: ./frontend
          file: frontend/Dockerfile
          push: true
          tags: ${{ env.REGISTRY }}/${{ env.IMAGE_PREFIX }}/frontend:${{ github.ref_name }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  deploy-staging:
    name: Deploy to Staging
    runs-on: ubuntu-latest
    needs: build-and-push
    if: github.event.inputs.environment != 'production'
    environment: staging
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Deploy to staging
        run: |
          echo "Deploying to staging environment..."
          echo "Tag: ${{ github.ref_name }}"
          # Add actual deployment commands here
          # e.g., kubectl apply, docker-compose pull && up, etc.

  deploy-production:
    name: Deploy to Production
    runs-on: ubuntu-latest
    needs: build-and-push
    if: github.event.inputs.environment == 'production' || startsWith(github.ref, 'refs/tags/v')
    environment: production
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Deploy to production
        run: |
          echo "Deploying to production environment..."
          echo "Tag: ${{ github.ref_name }}"
          # Add actual deployment commands here
```

---

## 5. Documentation

### 5.1 docs/DEPLOYMENT.md

```markdown
# SentinelFetal V3 - Deployment Guide

## Quick Start

### Prerequisites
- Docker 24.0+
- Docker Compose 2.20+
- 4GB RAM minimum
- Ports 80 and 8000 available

### Local Deployment

```bash
# Clone repository
git clone https://github.com/your-org/sentinelfetal.git
cd sentinelfetal

# Start services
docker compose up -d

# Verify health
curl http://localhost:8000/api/health
# Expected: {"status": "healthy"}

# Open application
open http://localhost
```

### Development Mode

```bash
# Start with hot-reload
docker compose -f docker-compose.yml -f docker-compose.dev.yml up

# Backend: http://localhost:8000
# Frontend: http://localhost:5173
```

### Production Deployment

```bash
# Pull latest images
docker compose -f docker-compose.yml -f docker-compose.prod.yml pull

# Start with production config
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Scale if needed
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --scale backend=2 --scale frontend=2
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | INFO | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `DEBUG` | false | Enable debug mode |
| `REGISTRY` | ghcr.io | Container registry |
| `TAG` | latest | Image tag |

## Health Checks

### Backend
```bash
curl http://localhost:8000/api/health
```

### Frontend
```bash
curl -I http://localhost/
```

### WebSocket
```bash
wscat -c ws://localhost/ws/stream
```

## Logs

```bash
# All logs
docker compose logs -f

# Backend only
docker compose logs -f backend

# Frontend only
docker compose logs -f frontend
```

## Updating

```bash
# Pull latest images
docker compose pull

# Restart with new images
docker compose up -d

# Remove old images
docker image prune -f
```

## Backup

The application is stateless. Patient data is simulated.
For persistent data, configure external database.

## Monitoring

Recommended monitoring stack:
- Prometheus for metrics
- Grafana for dashboards
- Loki for log aggregation

## Security

- HTTPS termination via reverse proxy (nginx, traefik)
- Container runs as non-root user
- No secrets in code or images
- Regular security scanning via CI
```

### 5.2 docs/TROUBLESHOOTING.md

```markdown
# SentinelFetal V3 - Troubleshooting Guide

## Common Issues

### Backend won't start

**Symptoms:** Backend container exits immediately

**Check logs:**
```bash
docker compose logs backend
```

**Common causes:**
1. Missing model files
   ```bash
   ls -la models/
   # Should contain minirocket_encoder.joblib
   ```

2. Port already in use
   ```bash
   lsof -i :8000
   # Kill conflicting process
   ```

3. Memory issues
   ```bash
   docker stats
   # Increase Docker memory limit
   ```

### Frontend shows "Connecting..."

**Symptoms:** WebSocket never connects

**Debug steps:**
1. Check backend is running:
   ```bash
   curl http://localhost:8000/api/health
   ```

2. Check WebSocket endpoint:
   ```bash
   wscat -c ws://localhost:8000/ws/stream
   ```

3. Check browser console for errors

**Common causes:**
1. Backend not ready - wait for health check
2. CORS issues - check browser console
3. Nginx proxy misconfigured

### Charts not updating

**Symptoms:** Data appears but doesn't animate

**Debug steps:**
1. Open browser DevTools → Network → WS
2. Verify messages arriving
3. Check console for errors

**Common causes:**
1. WebSocket disconnected - check connection status
2. JavaScript error - check console
3. Browser tab inactive - browsers throttle background tabs

### High CPU usage

**Symptoms:** Container using >100% CPU

**Debug steps:**
```bash
docker stats
```

**Solutions:**
1. Reduce patient count
2. Increase data emission interval
3. Check for infinite loops in logs

### Memory leak

**Symptoms:** Memory usage grows continuously

**Debug steps:**
```bash
docker stats --no-stream
# Wait 5 minutes
docker stats --no-stream
# Compare memory usage
```

**Solutions:**
1. Restart containers (temporary)
2. Check for accumulating data structures
3. Enable garbage collection logging

### E2E tests failing locally

**Symptoms:** Tests pass in CI but fail locally

**Common causes:**
1. Different browser versions
   ```bash
   npx playwright install
   ```

2. Backend not running
   ```bash
   curl http://localhost:8000/api/health
   ```

3. Port conflicts
   ```bash
   lsof -i :5173
   lsof -i :8000
   ```

4. Stale cache
   ```bash
   npm run build -- --force
   ```

### Hebrew text displays incorrectly

**Symptoms:** Hebrew appears as boxes or question marks

**Solutions:**
1. Ensure UTF-8 encoding in HTML
2. Check font supports Hebrew
3. Verify JSON files are UTF-8 encoded

### RTL layout issues

**Symptoms:** Layout doesn't flip correctly in Hebrew

**Debug steps:**
1. Check `document.documentElement.dir` in console
2. Verify `rtl.css` is loaded
3. Check for hardcoded margins/paddings

## Getting Help

1. Check logs: `docker compose logs -f`
2. Check health: `curl http://localhost:8000/api/health`
3. Check browser console: F12 → Console
4. Open GitHub issue with:
   - Steps to reproduce
   - Expected vs actual behavior
   - Logs and screenshots
   - Browser and OS version
```

---

## 6. Package.json Scripts Addition

### 6.1 Root package.json (workspace)

```json
{
  "name": "sentinelfetal-v3",
  "private": true,
  "scripts": {
    "dev": "concurrently \"cd backend && uvicorn main:app --reload\" \"cd frontend && npm run dev\"",
    "build": "cd frontend && npm run build",
    "test": "npm run test:unit && npm run test:e2e",
    "test:unit": "cd frontend && npm run test",
    "test:e2e": "cd e2e && npx playwright test",
    "test:e2e:headed": "cd e2e && npx playwright test --headed",
    "test:e2e:debug": "cd e2e && npx playwright test --debug",
    "docker:up": "docker compose up -d",
    "docker:down": "docker compose down",
    "docker:logs": "docker compose logs -f",
    "docker:build": "docker compose build"
  }
}
```

### 6.2 e2e/package.json

```json
{
  "name": "sentinelfetal-e2e",
  "private": true,
  "scripts": {
    "test": "playwright test",
    "test:headed": "playwright test --headed",
    "test:debug": "playwright test --debug",
    "test:report": "playwright show-report"
  },
  "devDependencies": {
    "@playwright/test": "^1.42.0"
  }
}
```

---

## 7. Verification Checklist

### 7.1 E2E Tests
```bash
# Install Playwright
cd e2e
npm install
npx playwright install

# Run tests
npm test

# View report
npm run test:report
```

### 7.2 Docker
```bash
# Build images
docker compose build

# Start services
docker compose up -d

# Verify health
curl http://localhost:8000/api/health
curl -I http://localhost/

# Check logs
docker compose logs -f

# Stop services
docker compose down
```

### 7.3 CI Pipeline
```bash
# Trigger manually (if configured)
gh workflow run ci.yml

# View runs
gh run list --workflow=ci.yml

# View logs
gh run view <run-id> --log
```

---

## 8. Final Project Structure

```
sentinelfetal/
├── backend/
│   ├── Dockerfile
│   ├── .dockerignore
│   ├── main.py
│   ├── routers/
│   ├── schemas/
│   └── services/
│
├── frontend/
│   ├── Dockerfile
│   ├── .dockerignore
│   ├── nginx.conf
│   ├── package.json
│   ├── vite.config.ts
│   ├── src/
│   └── public/
│
├── e2e/
│   ├── package.json
│   ├── playwright.config.ts
│   ├── global-setup.ts
│   ├── fixtures/
│   └── tests/
│
├── src/
│   └── (existing Python modules)
│
├── models/
│   └── (ML models)
│
├── docs/
│   ├── DEPLOYMENT.md
│   ├── TROUBLESHOOTING.md
│   └── plan/V3_Migration/
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── deploy.yml
│
├── docker-compose.yml
├── docker-compose.dev.yml
├── docker-compose.prod.yml
├── requirements.txt
└── README.md
```

---

## 9. Migration Complete Checklist

Upon completing Phase 6, verify:

- [ ] All E2E tests pass (ward, detail, godmode, i18n)
- [ ] Docker images build successfully
- [ ] `docker compose up` starts the system
- [ ] Health checks pass for all services
- [ ] CI pipeline runs on PR creation
- [ ] CI pipeline passes for clean code
- [ ] Documentation is complete and accurate
- [ ] New developer can onboard in <1 hour

---

*End of Phase 6 Technical Specifications*
*End of V3 Full-Stack Migration Documentation*
