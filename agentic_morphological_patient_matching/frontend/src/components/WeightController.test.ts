import { describe, it, expect } from 'vitest'
import fc from 'fast-check'
import { enforceWeightSum } from './WeightController'

const FIELDS = ['alpha', 'beta', 'gamma'] as const
type WeightKey = (typeof FIELDS)[number]

// ── Unit tests ────────────────────────────────────────────────────────────────

describe('enforceWeightSum', () => {
  it('keeps sum = 1 when changing alpha with non-zero others', () => {
    const result = enforceWeightSum('alpha', 0.6, { alpha: 0.33, beta: 0.33, gamma: 0.34 })
    expect(result.alpha + result.beta + result.gamma).toBeCloseTo(1.0, 10)
    expect(result.alpha).toBeCloseTo(0.6)
  })

  it('distributes remainder equally when other two weights are zero (req 5.3)', () => {
    const result = enforceWeightSum('alpha', 0.4, { alpha: 1.0, beta: 0.0, gamma: 0.0 })
    expect(result.alpha).toBeCloseTo(0.4)
    expect(result.beta).toBeCloseTo(0.3)
    expect(result.gamma).toBeCloseTo(0.3)
  })

  it('scales proportionally when others are non-zero (req 5.2)', () => {
    const result = enforceWeightSum('alpha', 0.5, { alpha: 0.2, beta: 0.4, gamma: 0.4 })
    expect(result.alpha).toBeCloseTo(0.5)
    // beta and gamma should remain equal since they started equal
    expect(result.beta).toBeCloseTo(0.25)
    expect(result.gamma).toBeCloseTo(0.25)
  })

  it('clamps new value to [0, 1] (req 5.4)', () => {
    const over = enforceWeightSum('beta', 1.5, { alpha: 0.5, beta: 0.3, gamma: 0.2 })
    expect(over.beta).toBe(1.0)
    expect(over.alpha + over.beta + over.gamma).toBeCloseTo(1.0, 10)

    const under = enforceWeightSum('gamma', -0.2, { alpha: 0.5, beta: 0.3, gamma: 0.2 })
    expect(under.gamma).toBe(0.0)
    expect(under.alpha + under.beta + under.gamma).toBeCloseTo(1.0, 10)
  })

  it('all values remain in [0, 1] after adjustment (req 5.4)', () => {
    const result = enforceWeightSum('gamma', 0.9, { alpha: 0.3, beta: 0.4, gamma: 0.3 })
    for (const f of FIELDS) {
      expect(result[f]).toBeGreaterThanOrEqual(0)
      expect(result[f]).toBeLessThanOrEqual(1)
    }
  })
})

// ── Property-based tests ──────────────────────────────────────────────────────

/**
 * Validates: Requirements 5.2, 5.3, 5.4
 *
 * Property 1: sum invariant — alpha + beta + gamma always equals 1.0
 */
describe('enforceWeightSum property: sum invariant', () => {
  it('sum is always 1.0 for any valid input', () => {
    fc.assert(
      fc.property(
        fc.constantFrom(...FIELDS),
        fc.float({ min: 0, max: 1, noNaN: true }),
        fc.float({ min: 0, max: 1, noNaN: true }),
        fc.float({ min: 0, max: 1, noNaN: true }),
        fc.float({ min: 0, max: 1, noNaN: true }),
        (field: WeightKey, newVal, a, b, g) => {
          const result = enforceWeightSum(field, newVal, { alpha: a, beta: b, gamma: g })
          const sum = result.alpha + result.beta + result.gamma
          expect(sum).toBeCloseTo(1.0, 10)
        },
      ),
    )
  })
})

/**
 * Validates: Requirements 5.4
 *
 * Property 2: range invariant — all weights remain in [0, 1]
 */
describe('enforceWeightSum property: range invariant', () => {
  it('all weights are in [0, 1] for any valid input', () => {
    fc.assert(
      fc.property(
        fc.constantFrom(...FIELDS),
        fc.float({ min: 0, max: 1, noNaN: true }),
        fc.float({ min: 0, max: 1, noNaN: true }),
        fc.float({ min: 0, max: 1, noNaN: true }),
        fc.float({ min: 0, max: 1, noNaN: true }),
        (field: WeightKey, newVal, a, b, g) => {
          const result = enforceWeightSum(field, newVal, { alpha: a, beta: b, gamma: g })
          for (const f of FIELDS) {
            expect(result[f]).toBeGreaterThanOrEqual(0)
            expect(result[f]).toBeLessThanOrEqual(1)
          }
        },
      ),
    )
  })
})
