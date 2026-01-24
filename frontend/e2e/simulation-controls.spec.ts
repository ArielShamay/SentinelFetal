/**
 * Simulation Controls E2E Tests
 * Tests for simulation start/pause/resume/reset functionality
 */

import { test, expect } from '@playwright/test'

test.describe('Simulation Controls', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
  })

  test('can start simulation', async ({ page }) => {
    // Find and click start button
    const startButton = page.getByRole('button', { name: /start/i })
    await expect(startButton).toBeVisible()
    
    // Click start
    await startButton.click()
    
    // Should show running status or pause button
    await expect(
      page.getByText(/running/i).or(page.getByRole('button', { name: /pause/i }))
    ).toBeVisible({ timeout: 10000 })
  })

  test('can pause running simulation', async ({ page }) => {
    // Start simulation first
    const startButton = page.getByRole('button', { name: /start/i })
    await startButton.click()
    
    // Wait for running state
    const pauseButton = page.getByRole('button', { name: /pause/i })
    await expect(pauseButton).toBeVisible({ timeout: 10000 })
    
    // Pause
    await pauseButton.click()
    
    // Should show paused status or resume button
    await expect(
      page.getByText(/paused/i).or(page.getByRole('button', { name: /resume/i }))
    ).toBeVisible({ timeout: 5000 })
  })

  test('can resume paused simulation', async ({ page }) => {
    // Start and pause
    await page.getByRole('button', { name: /start/i }).click()
    await page.getByRole('button', { name: /pause/i }).click({ timeout: 10000 })
    
    // Wait for resume button
    const resumeButton = page.getByRole('button', { name: /resume/i })
    await expect(resumeButton).toBeVisible({ timeout: 5000 })
    
    // Resume
    await resumeButton.click()
    
    // Should be running again
    await expect(
      page.getByText(/running/i).or(page.getByRole('button', { name: /pause/i }))
    ).toBeVisible({ timeout: 5000 })
  })

  test('shows status indicator', async ({ page }) => {
    // Initially should show stopped
    await expect(page.getByText(/stopped/i)).toBeVisible()
    
    // Start simulation
    await page.getByRole('button', { name: /start/i }).click()
    
    // Should show running
    await expect(page.getByText(/running/i)).toBeVisible({ timeout: 10000 })
  })
})
