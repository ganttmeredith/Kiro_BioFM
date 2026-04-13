import React from 'react'
import FormField from '@cloudscape-design/components/form-field'
import Input from '@cloudscape-design/components/input'
import SpaceBetween from '@cloudscape-design/components/space-between'

export interface WeightControllerProps {
  alpha: number
  beta: number
  gamma: number
  onChange: (weights: { alpha: number; beta: number; gamma: number }) => void
}

type WeightKey = 'alpha' | 'beta' | 'gamma'

/**
 * Enforces alpha + beta + gamma = 1.0 after changing one weight.
 * Exported as a pure function for independent testing.
 */
export function enforceWeightSum(
  changedField: WeightKey,
  newValue: number,
  current: { alpha: number; beta: number; gamma: number },
): { alpha: number; beta: number; gamma: number } {
  const clamped = Math.min(1, Math.max(0, newValue))
  const updated = { ...current, [changedField]: clamped }
  const remainder = 1.0 - clamped
  const others = (['alpha', 'beta', 'gamma'] as WeightKey[]).filter((f) => f !== changedField)
  const totalOther = others.reduce((sum, f) => sum + updated[f], 0)

  if (totalOther === 0) {
    for (const f of others) {
      updated[f] = remainder / 2
    }
  } else {
    for (const f of others) {
      updated[f] = updated[f] * (remainder / totalOther)
    }
  }

  return updated
}

const WEIGHT_LABELS: Record<WeightKey, string> = {
  alpha: 'Slide similarity (α)',
  beta: 'Composition similarity (β)',
  gamma: 'Metadata similarity (γ)',
}

export default function WeightController({
  alpha,
  beta,
  gamma,
  onChange,
}: WeightControllerProps): React.ReactElement {
  const weights = { alpha, beta, gamma }

  const handleChange = (field: WeightKey, raw: string) => {
    const parsed = parseFloat(raw)
    if (isNaN(parsed)) return
    const updated = enforceWeightSum(field, parsed, weights)
    onChange(updated)
  }

  return (
    <SpaceBetween size="s">
      {(['alpha', 'beta', 'gamma'] as WeightKey[]).map((field) => (
        <FormField key={field} label={WEIGHT_LABELS[field]}>
          <Input
            type="number"
            inputMode="decimal"
            value={weights[field].toFixed(3)}
            onChange={({ detail }) => handleChange(field, detail.value)}
            step={0.05}
          />
        </FormField>
      ))}
    </SpaceBetween>
  )
}
