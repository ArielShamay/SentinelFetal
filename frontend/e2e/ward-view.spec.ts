/**
 * Ward View E2E Tests
 * Tests for the main ward view page
 */

import { test, expect } from '@playwright/test'

test.describe('Ward View', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
  })

  test('displays the application header', async ({ page }) => {
    // Check for the app title
    await expect(page.locator('header')).toBeVisible()
    await expect(page.getByText('SentinelFetal')).toBeVisible()
    await expect(page.getByText('V3.0')).toBeVisible()
  })

  test('shows simulation controls in header', async ({ page }) => {
    // Check for start button (when stopped)
    const startButton = page.getByRole('button', { name: /start/i })
    await expect(startButton).toBeVisible()
  })

  test('shows language toggle', async ({ page }) => {
    // Check for EN/HE toggle
    await expect(page.getByText('EN')).toBeVisible()
    await expect(page.getByText('עב')).toBeVisible()
  })

  test('shows connection status', async ({ page }) => {
    // Should show either connected or disconnected status
    const connectionStatus = page.locator('[class*="ConnectionStatus"]')
    await expect(connectionStatus.or(page.getByText(/Live|Offline|Reconnecting/i))).toBeVisible()
  })

  test('displays filter controls', async ({ page }) => {
    // Check for search input
    await expect(page.getByPlaceholder(/search/i)).toBeVisible()
    
    // Check for filter buttons
    await expect(page.getByRole('button', { name: /all/i })).toBeVisible()
  })

  test('switches language to Hebrew', async ({ page }) => {
    // Click language toggle
    await page.getByText('עב').click()
    
    // Wait for language change
    await page.waitForTimeout(500)
    
    // Check that document direction is RTL
    const dir = await page.locator('html').getAttribute('dir')
    expect(dir).toBe('rtl')
    
    // Switch back to English
    await page.getByText('EN').click()
    await page.waitForTimeout(500)
    
    const dirAfter = await page.locator('html').getAttribute('dir')
    expect(dirAfter).toBe('ltr')
  })
})
