import React, { useRef, useState, useEffect, useCallback } from 'react'
import Plotly from 'plotly.js-dist'
import Button from '@cloudscape-design/components/button'
import Checkbox from '@cloudscape-design/components/checkbox'
import Container from '@cloudscape-design/components/container'
import Header from '@cloudscape-design/components/header'
import SpaceBetween from '@cloudscape-design/components/space-between'
import ColumnLayout from '@cloudscape-design/components/column-layout'
import Flashbar, { FlashbarProps } from '@cloudscape-design/components/flashbar'
import Spinner from '@cloudscape-design/components/spinner'
import Box from '@cloudscape-design/components/box'
import KeyValuePairs from '@cloudscape-design/components/key-value-pairs'
import Table, { TableProps } from '@cloudscape-design/components/table'
import StatusIndicator from '@cloudscape-design/components/status-indicator'
import Tabs from '@cloudscape-design/components/tabs'
import Select, { SelectProps } from '@cloudscape-design/components/select'
import FormField from '@cloudscape-design/components/form-field'
import Slider from '@cloudscape-design/components/slider'
import ExpandableSection from '@cloudscape-design/components/expandable-section'
import { classifyCohorts, analyzeBiomarkers, generateBoxPlot, runOutcomeUMAP, exportBiomarkerCSV, interpretResults, ApiError } from '../api/client'
import type {
  OutcomeCriteria,
  ClassifyResponse,
  BiomarkerResponse,
  AnalyteComparison,
  OutcomeUMAPResponse,
  DeviationCell,
} from '../types'

interface BiomarkerPanelProps {
  onNotLoadedError: () => void
}

const DEFAULT_CRITERIA: OutcomeCriteria = {
  deceased: false,
  tumorCausedDeath: false,
  recurrence: false,
  progression: false,
  metastasis: false,
}

function anyCriteriaSelected(c: OutcomeCriteria): boolean {
  return c.deceased || c.tumorCausedDeath || c.recurrence || c.progression || c.metastasis
}

function formatSexDistribution(dist: Record<string, number>): string {
  const parts = Object.entries(dist).map(([sex, count]) => `${sex}: ${count}`)
  return parts.length > 0 ? parts.join(', ') : '—'
}

function formatPValue(p: number): string {
  if (p < 0.001) return p.toExponential(2)
  return p.toFixed(4)
}

function formatMean(v: number): string {
  return v.toFixed(2)
}

function formatEffectSize(d: number): string {
  return d.toFixed(3)
}

const COLUMN_DEFINITIONS: TableProps.ColumnDefinition<AnalyteComparison>[] = [
  {
    id: 'analyteName',
    header: 'Analyte',
    cell: (item) => item.analyteName,
    sortingField: 'analyteName',
  },
  {
    id: 'nonResponderMean',
    header: 'Non_Responder mean',
    cell: (item) => formatMean(item.nonResponderMean),
    sortingField: 'nonResponderMean',
  },
  {
    id: 'responderMean',
    header: 'Responder mean',
    cell: (item) => formatMean(item.responderMean),
    sortingField: 'responderMean',
  },
  {
    id: 'pValue',
    header: 'p-value',
    cell: (item) => formatPValue(item.pValue),
    sortingField: 'pValue',
  },
  {
    id: 'adjustedPValue',
    header: 'Adjusted p-value',
    cell: (item) => formatPValue(item.adjustedPValue),
    sortingField: 'adjustedPValue',
  },
  {
    id: 'effectSize',
    header: 'Effect Size',
    cell: (item) => formatEffectSize(item.effectSize),
    sortingField: 'effectSize',
  },
  {
    id: 'significant',
    header: 'Significance',
    cell: (item) =>
      item.significant ? (
        <StatusIndicator type="success">Significant</StatusIndicator>
      ) : (
        <StatusIndicator type="stopped">—</StatusIndicator>
      ),
    sortingComparator: (a, b) => Number(b.significant) - Number(a.significant),
  },
]

export default function BiomarkerPanel({ onNotLoadedError }: BiomarkerPanelProps): React.ReactElement {
  const [criteria, setCriteria] = useState<OutcomeCriteria>(DEFAULT_CRITERIA)
  const [classifyResponse, setClassifyResponse] = useState<ClassifyResponse | null>(null)
  const [biomarkerResponse, setBiomarkerResponse] = useState<BiomarkerResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [flashItems, setFlashItems] = useState<FlashbarProps.MessageDefinition[]>([])

  // Comparison table sorting state
  const [sortingColumn, setSortingColumn] = useState<TableProps.SortingColumn<AnalyteComparison>>(
    COLUMN_DEFINITIONS[4] // default sort by adjusted p-value
  )
  const [sortingDescending, setSortingDescending] = useState(false)

  // Box plot state
  const [selectedAnalyte, setSelectedAnalyte] = useState<string | null>(null)
  const [boxPlotJson, setBoxPlotJson] = useState<string | null>(null)
  const [boxPlotLoading, setBoxPlotLoading] = useState(false)
  const boxPlotDivRef = useRef<HTMLDivElement>(null)

  // Visualization tabs state
  const [activeVizTab, setActiveVizTab] = useState('boxplot')

  // Heatmap state
  const heatmapDivRef = useRef<HTMLDivElement>(null)

  // AI Interpretation state
  const [interpretation, setInterpretation] = useState<string | null>(null)
  const [interpretLoading, setInterpretLoading] = useState(false)
  const [umapInterpretation, setUmapInterpretation] = useState<string | null>(null)
  const [umapInterpretLoading, setUmapInterpretLoading] = useState(false)

  // UMAP state
  const [umapModality, setUmapModality] = useState<SelectProps.Option>({ value: 'imaging', label: 'Imaging' })
  const [umapColorBy, setUmapColorBy] = useState<SelectProps.Option>({ value: 'cohort', label: 'cohort' })
  const [umapNNeighbors, setUmapNNeighbors] = useState(15)
  const [umapMinDist, setUmapMinDist] = useState(0.1)
  const [umapResponse, setUmapResponse] = useState<OutcomeUMAPResponse | null>(null)
  const [umapLoading, setUmapLoading] = useState(false)
  const umapDivRef = useRef<HTMLDivElement>(null)

  const MODALITY_OPTIONS: SelectProps.Option[] = [
    { value: 'imaging', label: 'Imaging' },
    { value: 'clinical', label: 'Clinical' },
    { value: 'multimodal', label: 'Multimodal' },
  ]

  const COLOR_BY_OPTIONS: SelectProps.Option[] = [
    { value: 'cohort', label: 'cohort' },
    { value: 'survival_status', label: 'survival_status' },
    { value: 'recurrence', label: 'recurrence' },
    { value: 'smoking_status', label: 'smoking_status' },
    { value: 'sex', label: 'sex' },
  ]

  const hasSelection = anyCriteriaSelected(criteria)

  const handleClassify = useCallback(async () => {
    setLoading(true)
    setFlashItems([])
    setClassifyResponse(null)
    setBiomarkerResponse(null)
    setSelectedAnalyte(null)
    setBoxPlotJson(null)
    setInterpretation(null)
    setUmapInterpretation(null)

    try {
      const classifyResult = await classifyCohorts({ criteria })
      setClassifyResponse(classifyResult)

      // Auto-trigger biomarker analysis
      try {
        const biomarkerResult = await analyzeBiomarkers({ criteria })
        setBiomarkerResponse(biomarkerResult)

        // Auto-trigger AI interpretation
        setInterpretLoading(true)
        interpretResults({
          contextType: 'biomarker_stats',
          contextData: {
            significant_analytes: biomarkerResult.comparisons.filter(c => c.significant).map(c => ({
              analyte_name: c.analyteName,
              adjusted_p_value: c.adjustedPValue,
              effect_size: c.effectSize,
              non_responder_mean: c.nonResponderMean,
              responder_mean: c.responderMean,
            })),
            total_analytes: biomarkerResult.comparisons.length,
          },
        })
          .then(res => setInterpretation(res.interpretation))
          .catch(() => setInterpretation('AI interpretation temporarily unavailable.'))
          .finally(() => setInterpretLoading(false))
      } catch (bioErr) {
        if (bioErr instanceof ApiError && bioErr.status === 409) {
          onNotLoadedError()
          return
        }
        const msg = bioErr instanceof Error ? bioErr.message : 'Unknown error'
        setFlashItems([{
          id: 'biomarker-error',
          type: 'error',
          dismissible: true,
          header: 'Biomarker analysis failed',
          content: msg,
          onDismiss: () => setFlashItems([]),
        }])
      }
    } catch (err) {
      if (err instanceof ApiError && err.status === 409) {
        onNotLoadedError()
        return
      }
      const msg = err instanceof Error ? err.message : 'Unknown error'
      setFlashItems([{
        id: 'classify-error',
        type: 'error',
        dismissible: true,
        header: 'Cohort classification failed',
        content: msg,
        onDismiss: () => setFlashItems([]),
      }])
    } finally {
      setLoading(false)
    }
  }, [criteria, onNotLoadedError])

  // Handle row click — fetch box plot for selected analyte
  const handleRowClick = useCallback(async (item: AnalyteComparison) => {
    setSelectedAnalyte(item.analyteName)
    setBoxPlotLoading(true)
    try {
      const result = await generateBoxPlot({ criteria, analyteName: item.analyteName })
      setBoxPlotJson(result.plotlyJson)
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Unknown error'
      setFlashItems(prev => [
        ...prev.filter(f => f.id !== 'boxplot-error'),
        {
          id: 'boxplot-error',
          type: 'error' as const,
          dismissible: true,
          header: 'Box plot failed',
          content: msg,
          onDismiss: () => setFlashItems(items => items.filter(f => f.id !== 'boxplot-error')),
        },
      ])
      setBoxPlotJson(null)
    } finally {
      setBoxPlotLoading(false)
    }
  }, [criteria])

  // Render Plotly box plot when JSON changes
  useEffect(() => {
    if (!boxPlotJson || !boxPlotDivRef.current) return
    try {
      const figure = JSON.parse(boxPlotJson) as {
        data: Plotly.Data[]
        layout: Partial<Plotly.Layout>
      }
      Plotly.react(boxPlotDivRef.current, figure.data, figure.layout, { responsive: true })
    } catch {
      // ignore malformed JSON
    }
  }, [boxPlotJson])

  // Render Plotly heatmap when deviation scores or active tab changes
  useEffect(() => {
    if (activeVizTab !== 'heatmap' || !biomarkerResponse?.deviationScores?.length || !heatmapDivRef.current) return

    const cells = biomarkerResponse.deviationScores
    // Get unique analyte names (rows) and patient IDs grouped by cohort (columns)
    const analyteNames = [...new Set(cells.map((c: DeviationCell) => c.analyteName))]
    const nonResponderIds = [...new Set(cells.filter((c: DeviationCell) => c.cohort === 'non_responder').map((c: DeviationCell) => c.patientId))]
    const responderIds = [...new Set(cells.filter((c: DeviationCell) => c.cohort === 'responder').map((c: DeviationCell) => c.patientId))]
    const patientIds = [...nonResponderIds, ...responderIds]

    // Build lookup map
    const lookup = new Map<string, number | null>()
    for (const c of cells) {
      lookup.set(`${c.analyteName}::${c.patientId}`, c.deviationScore)
    }

    // Build z-matrix (rows=analytes, columns=patients)
    const z: (number | null)[][] = analyteNames.map(analyte =>
      patientIds.map(pid => lookup.get(`${analyte}::${pid}`) ?? null)
    )

    const data: Plotly.Data[] = [{
      type: 'heatmap' as const,
      z,
      x: patientIds,
      y: analyteNames,
      colorscale: 'RdBu',
      reversescale: true,
      zmid: 0,
      colorbar: { title: 'Deviation Score' },
      hoverongaps: false,
    }]

    const layout: Partial<Plotly.Layout> = {
      title: 'Deviation Score Heatmap',
      xaxis: {
        title: 'Patients (Non_Responder | Responder)',
        tickangle: -45,
        tickfont: { size: 8 },
      },
      yaxis: { title: 'Analytes', automargin: true },
      margin: { l: 150, b: 100, t: 50, r: 50 },
      // Add a vertical line separating cohorts
      shapes: nonResponderIds.length > 0 ? [{
        type: 'line',
        x0: nonResponderIds.length - 0.5,
        x1: nonResponderIds.length - 0.5,
        y0: -0.5,
        y1: analyteNames.length - 0.5,
        line: { color: 'black', width: 2, dash: 'dash' },
      }] : [],
    }

    Plotly.react(heatmapDivRef.current, data, layout, { responsive: true })
  }, [activeVizTab, biomarkerResponse])

  // Render Plotly UMAP scatter when response changes
  useEffect(() => {
    if (!umapResponse || !umapDivRef.current) return
    try {
      const figure = JSON.parse(umapResponse.plotlyJson) as {
        data: Plotly.Data[]
        layout: Partial<Plotly.Layout>
      }
      Plotly.react(umapDivRef.current, figure.data, figure.layout, { responsive: true })
    } catch {
      // ignore malformed JSON
    }
  }, [umapResponse])

  // Handle UMAP run
  const handleRunUMAP = useCallback(async () => {
    setUmapLoading(true)
    setUmapInterpretation(null)
    try {
      const result = await runOutcomeUMAP({
        criteria,
        modality: (umapModality.value as 'imaging' | 'clinical' | 'multimodal') ?? 'imaging',
        nNeighbors: umapNNeighbors,
        minDist: umapMinDist,
        colorBy: umapColorBy.value ?? 'cohort',
      })
      setUmapResponse(result)
    } catch (err) {
      if (err instanceof ApiError && err.status === 409) {
        onNotLoadedError()
        return
      }
      const msg = err instanceof Error ? err.message : 'Unknown error'
      setFlashItems(prev => [
        ...prev.filter(f => f.id !== 'umap-error'),
        {
          id: 'umap-error',
          type: 'error' as const,
          dismissible: true,
          header: 'Outcome UMAP failed',
          content: msg,
          onDismiss: () => setFlashItems(items => items.filter(f => f.id !== 'umap-error')),
        },
      ])
    } finally {
      setUmapLoading(false)
    }
  }, [criteria, umapModality, umapColorBy, umapNNeighbors, umapMinDist, onNotLoadedError])

  // Handle UMAP cluster interpretation
  const handleInterpretClusters = useCallback(async () => {
    if (!umapResponse) return
    setUmapInterpretLoading(true)
    try {
      const res = await interpretResults({
        contextType: 'umap_clusters',
        contextData: {
          silhouette_score: umapResponse.silhouetteScore,
          n_points: umapResponse.nPoints,
          modality: umapModality.value ?? 'imaging',
          color_by: umapColorBy.value ?? 'cohort',
          n_neighbors: umapNNeighbors,
          min_dist: umapMinDist,
        },
      })
      setUmapInterpretation(res.interpretation)
    } catch {
      setUmapInterpretation('AI interpretation temporarily unavailable.')
    } finally {
      setUmapInterpretLoading(false)
    }
  }, [umapResponse, umapModality, umapColorBy, umapNNeighbors, umapMinDist])

  // Handle CSV export
  const handleExport = useCallback(async () => {
    try {
      const blob = await exportBiomarkerCSV({ criteria })
      const url = URL.createObjectURL(blob)
      const date = new Date().toISOString().slice(0, 10)
      const selectedCriteria = Object.entries(criteria)
        .filter(([, v]) => v)
        .map(([k]) => k.replace(/([A-Z])/g, '_$1').toLowerCase())
        .join('_')
      const filename = `biomarker_comparison_${date}_${selectedCriteria}.csv`
      const a = document.createElement('a')
      a.href = url
      a.download = filename
      document.body.appendChild(a)
      a.click()
      a.remove()
      URL.revokeObjectURL(url)
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Unknown error'
      setFlashItems(prev => [
        ...prev.filter(f => f.id !== 'export-error'),
        {
          id: 'export-error',
          type: 'error' as const,
          dismissible: true,
          header: 'CSV export failed',
          content: msg,
          onDismiss: () => setFlashItems(items => items.filter(f => f.id !== 'export-error')),
        },
      ])
    }
  }, [criteria])

  // Sort comparison items
  const sortedComparisons = React.useMemo(() => {
    if (!biomarkerResponse) return []
    const items = [...biomarkerResponse.comparisons]
    const col = sortingColumn
    if (!col) return items

    items.sort((a, b) => {
      // Use sortingComparator if available (significance column)
      if (col.sortingComparator) {
        return col.sortingComparator(a, b)
      }
      // Use sortingField for other columns
      const field = col.sortingField as keyof AnalyteComparison | undefined
      if (!field) return 0
      const va = a[field]
      const vb = b[field]
      if (typeof va === 'string' && typeof vb === 'string') return va.localeCompare(vb)
      if (typeof va === 'number' && typeof vb === 'number') return va - vb
      return 0
    })
    if (sortingDescending) items.reverse()
    return items
  }, [biomarkerResponse, sortingColumn, sortingDescending])

  return (
    <SpaceBetween size="m">
      {flashItems.length > 0 && <Flashbar items={flashItems} />}

      {/* ── Cohort Definition ──────────────────────────────────── */}
      <Container
        header={
          <Header
            variant="h2"
            description="Select one or more unfavorable outcome criteria to classify patients into Non_Responder and Responder cohorts"
            actions={
              <Button
                variant="primary"
                onClick={() => void handleClassify()}
                loading={loading}
                disabled={!hasSelection}
              >
                Classify
              </Button>
            }
          >
            Cohort Definition
          </Header>
        }
      >
        <SpaceBetween size="s">
          <ColumnLayout columns={2}>
            <Checkbox
              checked={criteria.deceased}
              onChange={({ detail }) => setCriteria(prev => ({ ...prev, deceased: detail.checked }))}
            >
              Deceased
            </Checkbox>
            <Checkbox
              checked={criteria.tumorCausedDeath}
              onChange={({ detail }) => setCriteria(prev => ({ ...prev, tumorCausedDeath: detail.checked }))}
            >
              Tumor-caused death
            </Checkbox>
            <Checkbox
              checked={criteria.recurrence}
              onChange={({ detail }) => setCriteria(prev => ({ ...prev, recurrence: detail.checked }))}
            >
              Recurrence
            </Checkbox>
            <Checkbox
              checked={criteria.progression}
              onChange={({ detail }) => setCriteria(prev => ({ ...prev, progression: detail.checked }))}
            >
              Progression
            </Checkbox>
            <Checkbox
              checked={criteria.metastasis}
              onChange={({ detail }) => setCriteria(prev => ({ ...prev, metastasis: detail.checked }))}
            >
              Metastasis
            </Checkbox>
          </ColumnLayout>
          {!hasSelection && (
            <Box variant="small" color="text-status-warning">
              Please select at least one outcome criterion to classify cohorts.
            </Box>
          )}
        </SpaceBetween>
      </Container>

      {/* ── Loading ────────────────────────────────────────────── */}
      {loading && (
        <Box textAlign="center" padding="xl">
          <Spinner size="large" />
          <Box variant="p" color="text-status-inactive" padding={{ top: 's' }}>
            Classifying cohorts and analyzing biomarkers…
          </Box>
        </Box>
      )}

      {/* ── Cohort Summary Cards ───────────────────────────────── */}
      {!loading && classifyResponse && (
        <>
          <ColumnLayout columns={2}>
            <Container header={<Header variant="h2">Non-Responders</Header>}>
              <KeyValuePairs
                columns={1}
                items={[
                  { label: 'Count', value: String(classifyResponse.nonResponder.count) },
                  {
                    label: 'Mean age',
                    value: classifyResponse.nonResponder.meanAge != null
                      ? classifyResponse.nonResponder.meanAge.toFixed(1)
                      : '—',
                  },
                  {
                    label: 'Sex distribution',
                    value: formatSexDistribution(classifyResponse.nonResponder.sexDistribution),
                  },
                ]}
              />
            </Container>
            <Container header={<Header variant="h2">Responders</Header>}>
              <KeyValuePairs
                columns={1}
                items={[
                  { label: 'Count', value: String(classifyResponse.responder.count) },
                  {
                    label: 'Mean age',
                    value: classifyResponse.responder.meanAge != null
                      ? classifyResponse.responder.meanAge.toFixed(1)
                      : '—',
                  },
                  {
                    label: 'Sex distribution',
                    value: formatSexDistribution(classifyResponse.responder.sexDistribution),
                  },
                ]}
              />
            </Container>
          </ColumnLayout>
          {classifyResponse.excludedCount > 0 && (
            <Box variant="small" color="text-status-info" padding={{ top: 'xs' }}>
              <StatusIndicator type="info">
                {classifyResponse.excludedCount} patient{classifyResponse.excludedCount !== 1 ? 's' : ''} excluded due to insufficient analyte data
              </StatusIndicator>
            </Box>
          )}
        </>
      )}

      {/* ── Biomarker Comparison Table ──────────────────────── */}
      {!loading && biomarkerResponse && (
        <Container
          header={
            <Header
              variant="h2"
              description={`${biomarkerResponse.comparisons.length} analytes compared · ${biomarkerResponse.comparisons.filter(c => c.significant).length} significant (adj. p < 0.05) · Click a row to view box plot`}
              actions={
                <Button
                  iconName="download"
                  onClick={() => void handleExport()}
                  disabled={loading || !biomarkerResponse}
                >
                  Export CSV
                </Button>
              }
            >
              Biomarker Comparison
            </Header>
          }
        >
          <Table
            items={sortedComparisons}
            columnDefinitions={COLUMN_DEFINITIONS}
            variant="embedded"
            stripedRows
            resizableColumns
            sortingColumn={sortingColumn}
            sortingDescending={sortingDescending}
            onSortingChange={({ detail }) => {
              setSortingColumn(detail.sortingColumn)
              setSortingDescending(detail.isDescending ?? false)
            }}
            onRowClick={({ detail }) => void handleRowClick(detail.item)}
            empty={
              <Box textAlign="center" color="text-status-inactive">
                No analyte comparisons available.
              </Box>
            }
          />
        </Container>
      )}

      {/* ── AI Interpretation ──────────────────────────────── */}
      {!loading && biomarkerResponse && (
        <ExpandableSection
          variant="container"
          headerText="AI Interpretation"
          defaultExpanded={!!interpretation}
        >
          {interpretLoading ? (
            <Box textAlign="center" padding="l">
              <Spinner size="normal" />
              <Box variant="p" color="text-status-inactive" padding={{ top: 's' }}>
                Generating AI interpretation…
              </Box>
            </Box>
          ) : interpretation ? (
            <Box variant="p" padding="s">
              <div style={{ whiteSpace: 'pre-wrap' }}>{interpretation}</div>
            </Box>
          ) : (
            <Box variant="p" color="text-status-inactive" padding="s">
              AI interpretation temporarily unavailable.
            </Box>
          )}
        </ExpandableSection>
      )}

      {/* ── Visualization Tabs (Box Plot, Heatmap, UMAP) ──── */}
      {!loading && biomarkerResponse && (
        <Tabs
          activeTabId={activeVizTab}
          onChange={({ detail }) => setActiveVizTab(detail.activeTabId)}
          tabs={[
            {
              id: 'boxplot',
              label: 'Box Plot',
              content: selectedAnalyte ? (
                <Container header={<Header variant="h2">Box Plot: {selectedAnalyte}</Header>}>
                  {boxPlotLoading ? (
                    <Box textAlign="center" padding="l">
                      <Spinner size="large" />
                      <Box variant="p" color="text-status-inactive" padding={{ top: 's' }}>
                        Loading box plot…
                      </Box>
                    </Box>
                  ) : boxPlotJson ? (
                    <div ref={boxPlotDivRef} style={{ width: '100%', minHeight: 450 }} />
                  ) : (
                    <Box textAlign="center" color="text-status-inactive" padding="l">
                      Failed to load box plot.
                    </Box>
                  )}
                </Container>
              ) : (
                <Box textAlign="center" color="text-status-inactive" padding="xl">
                  Click a row in the comparison table above to view a box plot.
                </Box>
              ),
            },
            {
              id: 'heatmap',
              label: 'Heatmap',
              content: biomarkerResponse.deviationScores?.length ? (
                <Container header={<Header variant="h2" description="Rows = analytes, columns = patients grouped by cohort (Non_Responder | Responder). Null values shown as gaps.">Deviation Score Heatmap</Header>}>
                  <div ref={heatmapDivRef} style={{ width: '100%', minHeight: 500 }} />
                </Container>
              ) : (
                <Box textAlign="center" color="text-status-inactive" padding="xl">
                  No deviation scores available. Ensure biomarker analysis has completed.
                </Box>
              ),
            },
            {
              id: 'umap',
              label: 'UMAP',
              content: (
                <SpaceBetween size="m">
                  <Container
                    header={
                      <Header
                        variant="h2"
                        description="Project patients into 2-D using imaging, clinical, or multimodal features"
                        actions={
                          <Button
                            variant="primary"
                            onClick={() => void handleRunUMAP()}
                            loading={umapLoading}
                            disabled={!hasSelection}
                          >
                            Run UMAP
                          </Button>
                        }
                      >
                        Outcome UMAP
                      </Header>
                    }
                  >
                    <SpaceBetween size="m">
                      <FormField label="Modality">
                        <Select
                          selectedOption={umapModality}
                          options={MODALITY_OPTIONS}
                          onChange={({ detail }) => setUmapModality(detail.selectedOption)}
                        />
                      </FormField>
                      <FormField label="Color by">
                        <Select
                          selectedOption={umapColorBy}
                          options={COLOR_BY_OPTIONS}
                          onChange={({ detail }) => setUmapColorBy(detail.selectedOption)}
                        />
                      </FormField>
                      <FormField label="n_neighbors" description="Controls local vs. global structure (5 – 50)">
                        <Slider
                          min={5}
                          max={50}
                          step={1}
                          value={umapNNeighbors}
                          onChange={({ detail }) => setUmapNNeighbors(detail.value)}
                        />
                      </FormField>
                      <FormField label="min_dist" description="Minimum distance between embedded points (0.01 – 0.5)">
                        <Slider
                          min={0.01}
                          max={0.5}
                          step={0.01}
                          value={umapMinDist}
                          onChange={({ detail }) => setUmapMinDist(detail.value)}
                        />
                      </FormField>
                    </SpaceBetween>
                  </Container>

                  {umapLoading && (
                    <Box textAlign="center" padding="xl">
                      <Spinner size="large" />
                      <Box variant="p" color="text-status-inactive" padding={{ top: 's' }}>
                        Running outcome UMAP…
                      </Box>
                    </Box>
                  )}

                  {!umapLoading && umapResponse && (
                    <Container
                      header={
                        <Header
                          variant="h2"
                          description={`${umapResponse.nPoints} points · Silhouette score: ${umapResponse.silhouetteScore.toFixed(3)}`}
                          actions={
                            <Button
                              onClick={() => void handleInterpretClusters()}
                              loading={umapInterpretLoading}
                            >
                              Interpret Clusters
                            </Button>
                          }
                        >
                          UMAP Projection
                        </Header>
                      }
                    >
                      <SpaceBetween size="m">
                        <div ref={umapDivRef} style={{ width: '100%', minHeight: 500 }} />
                        {umapInterpretLoading && (
                          <Box textAlign="center" padding="l">
                            <Spinner size="normal" />
                            <Box variant="p" color="text-status-inactive" padding={{ top: 's' }}>
                              Generating cluster interpretation…
                            </Box>
                          </Box>
                        )}
                        {!umapInterpretLoading && umapInterpretation && (
                          <ExpandableSection headerText="AI Cluster Interpretation" defaultExpanded>
                            <Box variant="p" padding="s">
                              <div style={{ whiteSpace: 'pre-wrap' }}>{umapInterpretation}</div>
                            </Box>
                          </ExpandableSection>
                        )}
                      </SpaceBetween>
                    </Container>
                  )}

                  {!umapLoading && !umapResponse && (
                    <Box textAlign="center" padding="xl" color="text-status-inactive">
                      Configure parameters above and click <strong>Run UMAP</strong> to project patients.
                    </Box>
                  )}
                </SpaceBetween>
              ),
            },
          ]}
        />
      )}

      {/* ── Placeholder for subsequent tasks (visualizations, export) ── */}
      {!loading && !classifyResponse && (
        <Box textAlign="center" padding="xl" color="text-status-inactive">
          Select outcome criteria above and click <strong>Classify</strong> to discover biomarker differences between cohorts.
        </Box>
      )}
    </SpaceBetween>
  )
}
