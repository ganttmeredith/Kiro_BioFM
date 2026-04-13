import React, { useRef, useState, useEffect, useCallback } from 'react'
import Plotly from 'plotly.js-dist'
import Button from '@cloudscape-design/components/button'
import Container from '@cloudscape-design/components/container'
import Header from '@cloudscape-design/components/header'
import SpaceBetween from '@cloudscape-design/components/space-between'
import FormField from '@cloudscape-design/components/form-field'
import Slider from '@cloudscape-design/components/slider'
import Select, { SelectProps } from '@cloudscape-design/components/select'
import Flashbar, { FlashbarProps } from '@cloudscape-design/components/flashbar'
import Spinner from '@cloudscape-design/components/spinner'
import Box from '@cloudscape-design/components/box'
import { runUMAP, ApiError } from '../api/client'
import { exportUmapHtml } from '../utils/exportHtml'
import type { MetadataFilters, ColumnMeta, UMAPResponse } from '../types'

interface UMAPPanelProps {
  filters: MetadataFilters
  onQuerySelect: (patientId: string) => void
  highlightIds?: string[]
  queryHighlightId?: string
  highlightSlideName?: string
  neighbourSlideNames?: string[]
  /** Increments each time "Show on UMAP" is clicked — forces re-trigger even for same patient */
  highlightVersion?: number
  columns?: ColumnMeta[]
  umapResult?: UMAPResponse | null
  onUmapResult?: (result: UMAPResponse) => void
  onNotLoadedError?: () => void
  onFiltersRestrictive?: () => void
}

const DEFAULT_N_NEIGHBORS = 15
const DEFAULT_MIN_DIST = 0.1
const DEFAULT_COLOR_BY = 'primary_tumor_site'

export default function UMAPPanel({
  filters,
  onQuerySelect,
  highlightIds = [],
  queryHighlightId,
  highlightSlideName,
  neighbourSlideNames = [],
  highlightVersion = 0,
  columns = [],
  umapResult: externalUmapResult,
  onUmapResult,
  onNotLoadedError,
  onFiltersRestrictive,
}: UMAPPanelProps): React.ReactElement {
  const plotDivRef = useRef<HTMLDivElement>(null)

  const [nNeighbors, setNNeighbors] = useState(DEFAULT_N_NEIGHBORS)
  const [minDist, setMinDist] = useState(DEFAULT_MIN_DIST)
  const [colorBy, setColorBy] = useState<SelectProps.Option>({
    value: DEFAULT_COLOR_BY,
    label: DEFAULT_COLOR_BY,
  })

  const [loading, setLoading] = useState(false)
  // Use external (lifted) result if provided, fall back to local
  const [localUmapResult, setLocalUmapResult] = useState<UMAPResponse | null>(null)
  const umapResult = externalUmapResult !== undefined ? externalUmapResult : localUmapResult
  const [flashItems, setFlashItems] = useState<FlashbarProps.MessageDefinition[]>([])

  // Build color_by options from columns prop
  const colorByOptions: SelectProps.Option[] = columns.length > 0
    ? columns.map((c) => ({ value: c.name, label: c.label }))
    : [{ value: DEFAULT_COLOR_BY, label: DEFAULT_COLOR_BY }]

  // Render / update Plotly figure whenever result changes
  useEffect(() => {
    if (!umapResult || !plotDivRef.current) return

    let figure: { data: Plotly.Data[]; layout: Partial<Plotly.Layout> }
    try {
      figure = JSON.parse(umapResult.plotlyJson) as {
        data: Plotly.Data[]
        layout: Partial<Plotly.Layout>
      }
    } catch {
      return
    }

    Plotly.react(plotDivRef.current, figure.data, figure.layout, { responsive: true })
  }, [umapResult])

  // Attach click handler after plot renders
  useEffect(() => {
    const div = plotDivRef.current
    if (!div || !umapResult) return

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const handler = (eventData: any) => {
      const point = eventData?.points?.[0]
      if (!point) return
      // Patient ID may be in text or customdata
      const patientId: string =
        typeof point.text === 'string' && point.text
          ? point.text
          : Array.isArray(point.customdata)
          ? String(point.customdata[0])
          : typeof point.customdata === 'string'
          ? point.customdata
          : ''
      if (patientId) {
        onQuerySelect(patientId)
      }
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ;(div as any).on('plotly_click', handler)
    return () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      ;(div as any).removeListener?.('plotly_click', handler)
    }
  }, [umapResult, onQuerySelect])

  // Keep a stable ref to the latest run function so the colorBy effect doesn't go stale
  const handleRunUMAPRef = useRef<() => Promise<void>>(async () => {})

  const handleRunUMAP = useCallback(async (opts?: { querySlideName?: string; neighbourSlides?: string[] }) => {
    setLoading(true)
    setFlashItems([])
    try {
      const result = await runUMAP({
        nNeighbors,
        minDist,
        colorBy: colorBy.value ?? DEFAULT_COLOR_BY,
        filters,
        querySlideName: opts?.querySlideName,
        neighbourSlideNames: opts?.neighbourSlides,
      })
      setLocalUmapResult(result)
      onUmapResult?.(result)
    } catch (err) {
      if (err instanceof ApiError && err.status === 409) {
        onNotLoadedError?.()
        return
      }
      if (err instanceof ApiError && err.status === 400) {
        onFiltersRestrictive?.()
        return
      }
      const msg = err instanceof Error ? err.message : 'Unknown error'
      setFlashItems([
        {
          id: 'umap-error',
          type: 'error',
          dismissible: true,
          header: 'UMAP failed',
          content: msg,
          onDismiss: () => setFlashItems([]),
        },
      ])
    } finally {
      setLoading(false)
    }
  }, [nNeighbors, minDist, colorBy, filters, onNotLoadedError, onFiltersRestrictive])

  // Keep ref in sync
  useEffect(() => {
    handleRunUMAPRef.current = handleRunUMAP
  }, [handleRunUMAP])

  // Auto-rerun when color_by changes (only if a plot already exists)
  const isFirstColorByRender = useRef(true)
  useEffect(() => {
    if (isFirstColorByRender.current) {
      isFirstColorByRender.current = false
      return
    }
    if (umapResult) {
      void handleRunUMAPRef.current()
    }
  }, [colorBy]) // eslint-disable-line react-hooks/exhaustive-deps

  // Auto-rerun with highlights when "Show on UMAP" is clicked (highlightVersion increments each time)
  const isFirstHighlightRender = useRef(true)
  useEffect(() => {
    if (isFirstHighlightRender.current) {
      isFirstHighlightRender.current = false
      return
    }
    if (!highlightSlideName) return

    setLoading(true)
    setFlashItems([])
    runUMAP({
      nNeighbors,
      minDist,
      colorBy: colorBy.value ?? DEFAULT_COLOR_BY,
      filters,
      querySlideName: highlightSlideName,
      neighbourSlideNames,
    }).then((result) => {
      setLocalUmapResult(result)
      onUmapResult?.(result)
    }).catch((err) => {
      const msg = err instanceof Error ? err.message : 'Unknown error'
      setFlashItems([{ id: 'umap-error', type: 'error', dismissible: true, header: 'UMAP failed', content: msg, onDismiss: () => setFlashItems([]) }])
    }).finally(() => setLoading(false))
  }, [highlightVersion]) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <SpaceBetween size="m">
      {flashItems.length > 0 && <Flashbar items={flashItems} />}

      <Container
        header={
          <Header
            variant="h2"
            description="Visualise patient embeddings in 2-D with UMAP"
            actions={
              <Button variant="primary" onClick={() => void handleRunUMAP()} loading={loading}>
                Run UMAP
              </Button>
            }
          >
            UMAP Explorer
          </Header>
        }
      >
        <SpaceBetween size="m">
          <FormField label="n_neighbors" description="Controls local vs. global structure (5 – 50)">
            <Slider
              min={5}
              max={50}
              step={1}
              value={nNeighbors}
              onChange={({ detail }) => setNNeighbors(detail.value)}
            />
          </FormField>

          <FormField label="min_dist" description="Minimum distance between embedded points (0.01 – 0.5)">
            <Slider
              min={0.01}
              max={0.5}
              step={0.01}
              value={minDist}
              onChange={({ detail }) => setMinDist(detail.value)}
            />
          </FormField>

          <FormField label="Color by">
            <Select
              selectedOption={colorBy}
              options={colorByOptions}
              onChange={({ detail }) => setColorBy(detail.selectedOption)}
            />
          </FormField>
        </SpaceBetween>
      </Container>

      {loading && (
        <Box textAlign="center" padding="xl">
          <Spinner size="large" />
          <Box variant="p" color="text-status-inactive" padding={{ top: 's' }}>
            Running UMAP…
          </Box>
        </Box>
      )}

      {!loading && umapResult && (
        <Container
          header={
            <Header
              variant="h2"
              description={`${umapResult.nPoints} points · ${umapResult.nClusters} clusters — click a point to set query patient`}
              actions={
                <Button
                  iconName="download"
                  onClick={() => void exportUmapHtml(umapResult.plotlyJson, umapResult.nPoints, umapResult.nClusters)}
                >
                  Export HTML
                </Button>
              }
            >
              UMAP Plot
            </Header>
          }
        >
          <div ref={plotDivRef} style={{ width: '100%', minHeight: 500 }} />
        </Container>
      )}

      {!loading && !umapResult && (
        <Box textAlign="center" padding="xl" color="text-status-inactive">
          Configure parameters above and click <strong>Run UMAP</strong> to visualise the embedding.
        </Box>
      )}
    </SpaceBetween>
  )
}
