/**
 * Patient Detail E2E Tests
 * Tests for the patient detail view
 */

import { test, expect } from '@playwright/test'

test.describe('Patient Detail View', () => {
  // These tests require a running simulation with patients
  // In CI, we may need to mock the API responses

  test('navigates back to ward view', async ({ page }) => {
    // Go directly to a patient detail page
    await page.goto('/patient/test-patient-1')
    
    // Should show back button
    const backButton = page.getByRole('button', { name: /back/i })
    await expect(backButton).toBeVisible()
    
    // Click back
    await backButton.click()
    
    // Should be on ward view
    await expect(page).toHaveURL('/')
  })

  test('displays patient vitals panel', async ({ page }) => {
    await page.goto('/patient/test-patient-1')
    
    // Should show vitals section
    await expect(page.getByText(/Current Vitals|Vitals/i)).toBeVisible()
    
    // Should show FHR metric
    await expect(page.getByText(/FHR/i)).toBeVisible()
    
    // Should show baseline
    await expect(page.getByText(/Baseline/i)).toBeVisible()
  })

  test('displays CTG chart', async ({ page }) => {
    await page.goto('/patient/test-patient-1')
    
    // Should show CTG Monitor section
    await expect(page.getByText(/CTG Monitor/i)).toBeVisible()
    
    // Should have chart controls
    await expect(page.getByRole('button', { name: /fit/i })).toBeVisible()
    await expect(page.getByRole('button', { name: /live/i })).toBeVisible()
  })

  test('displays events panel', async ({ page }) => {
    await page.goto('/patient/test-patient-1')
    
    // Should show events section
    await expect(page.getByText(/Recent Events/i)).toBeVisible()
  })

  test('displays alerts panel', async ({ page }) => {
    await page.goto('/patient/test-patient-1')
    
    // Should show alerts section
    await expect(page.getByText(/Alerts/i)).toBeVisible()
  })

  test('shows category badge for patient', async ({ page }) => {
    await page.goto('/patient/test-patient-1')
    
    // Should show category badge (Normal, Intermediate, or Pathological)
    await expect(
      page.getByText(/Normal|Intermediate|Pathological|תקין|בינוני|פתולוגי/i)
    ).toBeVisible()
  })
})
