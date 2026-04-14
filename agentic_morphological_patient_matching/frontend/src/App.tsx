import React, { useEffect, useState, useCallback } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import Flashbar, { FlashbarProps } from '@cloudscape-design/components/flashbar'
import Spinner from '@cloudscape-design/components/spinner'
import Box from '@cloudscape-design/components/box'
import { fetchStatus, fetchSummary } from './api/client'
import type { MetadataFilters, DataSummary } from './types'
import DataPanel from './panels/DataPanel'
import RetrievalPanel from './panels/RetrievalPanel'
import UMAPPanel from './panels/UMAPPanel'
import ClusterPanel from './panels/ClusterPanel'
import ChatPanel from './panels/ChatPanel'
import BiomarkerPanel from './panels/BiomarkerPanel'
import ImmersiveShell from './components/ImmersiveShell'
import './immersive.css'

const loadingFallback = (
  <Box textAlign="center" padding="xl">
    <Spinner size="large" />
  </Box>
)

// ─── Filter helpers ───────────────────────────────────────────────────────────

function filtersToParam(filters: MetadataFilters): string {
  if (Object.keys(filters).length === 0) return ''
  return JSON.stringify(filters)
}

function paramToFilters(param: string | null): MetadataFilters {
  if (!param) return {}
  try {
    return JSON.parse(param) as MetadataFilters
  } catch {
    return {}
  }
}

// ─── App ──────────────────────────────────────────────────────────────────────

export default function App(): React.ReactElement {
  const navigate = useNavigate()
  const location = useLocation()
  const queryClient = useQueryClient()

  const searchParams = new URLSearchParams(location.search)
  const [filters, setFilters] = useState<MetadataFilters>(() =>
    paramToFilters(searchParams.get('filters')),
  )

  const [dataSummary, setDataSummary] = useState<DataSummary | null>(null)
  const [flashItems, setFlashItems] = useState<FlashbarProps.MessageDefinition[]>([])
  const [queryPatientId, setQueryPatientId] = useState<string>('')
  const [highlightIds, setHighlightIds] = useState<string[]>([])
  const [queryHighlightId, setQueryHighlightId] = useState<string | undefined>(undefined)
  const [highlightSlideName, setHighlightSlideName] = useState<string | undefined>(undefined)
  const [neighbourSlideNames, setNeighbourSlideNames] = useState<string[]>([])
  const [highlightVersion, setHighlightVersion] = useState(0)
  const [activeSection, setActiveSection] = useState('hero')

  const [umapResult, setUmapResult] = useState<import('./types').UMAPResponse | null>(null)
  const [retrievalResult, setRetrievalResult] = useState<import('./types').RetrievalResponse | null>(null)
  const [clusterResult, setClusterResult] = useState<import('./types').ClusterResponse | null>(null)
  const [weights, setWeights] = useState({ alpha: 0.4, beta: 0.4, gamma: 0.2 })

  // ── GET /api/status on mount ──────────────────────────────────────────────
  const {
    data: status,
    isLoading: statusLoading,
    isError: statusError,
    error: statusErrorObj,
  } = useQuery({
    queryKey: ['status'],
    queryFn: fetchStatus,
    retry: false,
  })

  useEffect(() => {
    if (status?.loaded && dataSummary === null) {
      fetchSummary()
        .then((summary) => setDataSummary(summary))
        .catch(() => {/* non-fatal */})
    }
  }, [status?.loaded]) // eslint-disable-line react-hooks/exhaustive-deps

  const dataLoaded = dataSummary !== null || status?.loaded === true
  const statusResolved = !statusLoading

  useEffect(() => {
    if (statusError) {
      setFlashItems((prev) => [
        ...prev.filter((f) => f.id !== 'status-error'),
        {
          id: 'status-error',
          type: 'error',
          dismissible: true,
          header: 'Could not reach backend',
          content: statusErrorObj instanceof Error ? statusErrorObj.message : 'Unknown error',
          onDismiss: () => setFlashItems((prev) => prev.filter((f) => f.id !== 'status-error')),
        },
      ])
    }
  }, [statusError, statusErrorObj])

  useEffect(() => {
    if (dataLoaded) {
      setFlashItems((prev) => [
        ...prev.filter((f) => f.id !== 'data-loaded'),
        {
          id: 'data-loaded',
          type: 'success',
          dismissible: true,
          header: 'Dataset loaded',
          content: dataSummary
            ? `${dataSummary.nSlides} slides · ${dataSummary.nPatients} patients`
            : `${status?.nSlides ?? ''} slides · ${status?.nPatients ?? ''} patients`,
          onDismiss: () => setFlashItems((prev) => prev.filter((f) => f.id !== 'data-loaded')),
        },
      ])
    }
  }, [dataLoaded]) // eslint-disable-line react-hooks/exhaustive-deps

  // Filter changes are handled directly by panels in scroll mode

  const handleDataLoaded = useCallback((summary: DataSummary) => {
    setDataSummary(summary)
    void queryClient.invalidateQueries({ queryKey: ['status'] })
  }, [queryClient])

  const handleQuerySelect = useCallback((patientId: string) => {
    setQueryPatientId(patientId)
  }, [])

  const handleShowOnUMAP = useCallback(
    (queryId: string, matchIds: string[], querySlideName?: string, neighbourSlides?: string[]) => {
      setQueryHighlightId(queryId)
      setHighlightIds(matchIds)
      setHighlightSlideName(querySlideName)
      setNeighbourSlideNames(neighbourSlides ?? [])
      setHighlightVersion(v => v + 1)
    },
    [],
  )

  const handleNotLoadedError = useCallback(() => {
    setFlashItems((prev) => [
      ...prev.filter((f) => f.id !== 'not-loaded'),
      {
        id: 'not-loaded',
        type: 'warning',
        dismissible: true,
        header: 'Dataset not loaded',
        content: 'Please load a dataset in the Data section first.',
        onDismiss: () => setFlashItems((prev) => prev.filter((f) => f.id !== 'not-loaded')),
      },
    ])
  }, [])

  const handleFiltersRestrictive = useCallback(() => {
    setFlashItems((prev) => [
      ...prev.filter((f) => f.id !== 'filters-restrictive'),
      {
        id: 'filters-restrictive',
        type: 'warning',
        dismissible: true,
        header: 'Filters too restrictive',
        content: 'Fewer than 2 slides match. Relax your filter selections.',
        onDismiss: () => setFlashItems((prev) => prev.filter((f) => f.id !== 'filters-restrictive')),
      },
    ])
  }, [])

  const filterableColumns = dataSummary?.filterableColumns ?? []

  const notifications: FlashbarProps.MessageDefinition[] = [
    ...(statusLoading
      ? [{ id: 'status-loading', type: 'in-progress' as const, loading: true, content: 'Checking backend…' }]
      : []),
    ...flashItems,
  ]

  // ── Build panel elements for each scroll section ──────────────────────────
  const panelElements = [
    // 1. Data
    <DataPanel key="data" onDataLoaded={handleDataLoaded} initialPath={status?.csvPath} summary={dataSummary} />,

    // 2. Explore (UMAP)
    !statusResolved && !dataSummary ? loadingFallback : dataLoaded ? (
      <UMAPPanel
        key="explore"
        filters={filters}
        onQuerySelect={handleQuerySelect}
        highlightIds={highlightIds}
        queryHighlightId={queryHighlightId}
        highlightSlideName={highlightSlideName}
        neighbourSlideNames={neighbourSlideNames}
        highlightVersion={highlightVersion}
        columns={filterableColumns}
        onNotLoadedError={handleNotLoadedError}
        onFiltersRestrictive={handleFiltersRestrictive}
        umapResult={umapResult}
        onUmapResult={setUmapResult}
      />
    ) : <Box key="explore" textAlign="center" color="text-status-inactive" padding="xl">Load data above first.</Box>,

    // 3. Morphology Groups (Cluster)
    !statusResolved && !dataSummary ? loadingFallback : dataLoaded ? (
      <ClusterPanel
        key="cluster"
        filters={filters}
        onNotLoadedError={handleNotLoadedError}
        onFiltersRestrictive={handleFiltersRestrictive}
        onClusterResult={setClusterResult}
      />
    ) : <Box key="cluster" textAlign="center" color="text-status-inactive" padding="xl">Load data above first.</Box>,

    // 4. Patient Matcher (Retrieval)
    !statusResolved && !dataSummary ? loadingFallback : dataLoaded ? (
      <RetrievalPanel
        key="retrieve"
        filters={filters}
        initialPatientId={queryPatientId}
        onShowOnUMAP={handleShowOnUMAP}
        onNotLoadedError={handleNotLoadedError}
        onFiltersRestrictive={handleFiltersRestrictive}
        result={retrievalResult}
        onResult={setRetrievalResult}
        weights={weights}
        onWeightsChange={setWeights}
        nullP95={dataSummary?.nullDistribution?.percentile95}
      />
    ) : <Box key="retrieve" textAlign="center" color="text-status-inactive" padding="xl">Load data above first.</Box>,

    // 5. Chat
    !statusResolved && !dataSummary ? loadingFallback : dataLoaded ? (
      <ChatPanel
        key="chat"
        appContext={{
          nSlides: dataSummary?.nSlides ?? status?.nSlides,
          nPatients: dataSummary?.nPatients ?? status?.nPatients,
          activeFilters: filters,
          queryPatientId: queryPatientId || undefined,
          retrievalResults: retrievalResult?.matches,
          alpha: weights.alpha,
          beta: weights.beta,
          gamma: weights.gamma,
          umapNPoints: umapResult?.nPoints,
          umapNClusters: umapResult?.nClusters,
          bestK: clusterResult?.bestK,
        }}
        onRetrievalResult={(result) => {
          setRetrievalResult(result)
          setQueryPatientId(result.queryPatient.patientId)
        }}
      />
    ) : <Box key="chat" textAlign="center" color="text-status-inactive" padding="xl">Load data above first.</Box>,

    // 6. Biomarker Discovery
    !statusResolved && !dataSummary ? loadingFallback : dataLoaded ? (
      <BiomarkerPanel key="biomarkers" onNotLoadedError={handleNotLoadedError} />
    ) : <Box key="biomarkers" textAlign="center" color="text-status-inactive" padding="xl">Load data above first.</Box>,
  ]

  return (
    <>
      {/* Floating notifications */}
      {notifications.length > 0 && (
        <div style={{ position: 'fixed', top: 16, left: '50%', transform: 'translateX(-50%)', zIndex: 1000, maxWidth: 600, width: '90%' }}>
          <Flashbar items={notifications} stackItems />
        </div>
      )}

      <ImmersiveShell
        activeSection={activeSection}
        onSectionChange={setActiveSection}
      >
        {panelElements}
      </ImmersiveShell>
    </>
  )
}

export { filtersToParam, paramToFilters }
export type { MetadataFilters }
