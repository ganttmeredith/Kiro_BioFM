import React, { useRef, useCallback } from 'react'
import Multiselect from '@cloudscape-design/components/multiselect'
import SpaceBetween from '@cloudscape-design/components/space-between'
import type { ColumnMeta, MetadataFilters } from '../types'

interface FilterPanelProps {
  columns: ColumnMeta[]
  activeFilters: MetadataFilters
  onChange: (filters: MetadataFilters) => void
}

export default function FilterPanel({
  columns,
  activeFilters,
  onChange,
}: FilterPanelProps): React.ReactElement {
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const handleChange = useCallback(
    (columnName: string, selectedValues: string[]) => {
      if (debounceRef.current !== null) {
        clearTimeout(debounceRef.current)
      }
      debounceRef.current = setTimeout(() => {
        const updated: MetadataFilters = { ...activeFilters }
        if (selectedValues.length === 0) {
          delete updated[columnName]
        } else {
          updated[columnName] = selectedValues
        }
        onChange(updated)
      }, 300)
    },
    [activeFilters, onChange],
  )

  return (
    <SpaceBetween size="m">
      {columns.map((col) => {
        const selected = (activeFilters[col.name] ?? []).map((v) => ({ value: v, label: v }))
        const options = col.values.map((v) => ({ value: v, label: v }))
        return (
          <Multiselect
            key={col.name}
            placeholder={`Filter by ${col.label}`}
            options={options}
            selectedOptions={selected}
            onChange={({ detail }) =>
              handleChange(
                col.name,
                detail.selectedOptions.map((o) => o.value as string),
              )
            }
          />
        )
      })}
    </SpaceBetween>
  )
}
