import React, { useEffect, useState, useCallback } from 'react'
import { useNavigate, useLocation, Routes, Route, Navigate } from 'react-router-dom'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import AppLayout from '@cloudscape-design/components/app-layout'
import SideNavigation, { SideNavigationProps } from '@cloudscape-design/components/side-navigation'
import Flashbar, { FlashbarProps } from '@cloudscape-design/components/flashbar'
import Alert from '@cloudscape-design/components/alert'
import SpaceBetween from '@cloudscape-design/components/space-between'
import Spinner from '@cloudscape-design/components/spinner'
import Box from '@cloudscape-design/components/box'
import { fetchStatus, fetchSummary } from './api/client'
import type { MetadataFilters, DataSummary, UMAPResponse, RetrievalResponse, AppContext } from './types'
import DataPanel from './panels/DataPanel'
import RetrievalPanel from './panels/RetrievalPanel'
import UMAPPanel from './panels/UMAPPanel'
import ClusterPanel from './panels/ClusterPanel'
import ChatPanel from './panels/ChatPanel'
import BiomarkerPanel from './panels/BiomarkerPanel'
import FilterPanel from './components/FilterPanel'

const loadingFallback = (
  <Box textAlign="center" padding="xl">
    <Spinner size="large" />
  </Box>
)

// ─── Filter URL serialisation ─────────────────────────────────────────────────

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

// ─── Nav helpers ──────────────────────────────────────────────────────────────

const NAV_ITEMS = [
  { text: 'Data', href: '/data' },
  { text: 'Explore', href: '/explore' },
  { text: 'Morphology Groups', href: '/cluster' },
  { text: 'Patient Matcher', href: '/retrieve' },
  { text: 'Chat', href: '/chat' },
  { text: 'Biomarker Discovery', href: '/biomarkers' },
]

const DATA_ONLY_PATHS = new Set(['/explore', '/cluster', '/retrieve', '/chat', '/biomarkers'])

// ─── App ──────────────────────────────────────────────────────────────────────

export default function App(): React.ReactElement {
  const navigate = useNavigate()
  const location = useLocation()
  const queryClient = useQueryClient()

  // Restore filters from URL on load
  const searchParams = new URLSearchParams(location.search)
  const [filters, setFilters] = useState<MetadataFilters>(() =>
    paramToFilters(searchParams.get('filters')),
  )

  const [dataSummary, setDataSummary] = useState<DataSummary | null>(null)
  const [flashItems, setFlashItems] = useState<FlashbarProps.MessageDefinition[]>([])
  const [queryPatientId, setQueryPatientId] = useState<string>('')
  const [highlightIds, setHighlightIds] = useState<string[]>([])
  const [queryHighlightId, setQueryHighlightId] = useState<string | undefined>(undefined)
  // Slide-level highlight names for UMAP overlay (set by "Show on UMAP")
  const [highlightSlideName, setHighlightSlideName] = useState<string | undefined>(undefined)
  const [neighbourSlideNames, setNeighbourSlideNames] = useState<string[]>([])
  const [highlightVersion, setHighlightVersion] = useState(0)

  // Lifted panel state — survives tab navigation
  const [umapResult, setUmapResult] = useState<import('./types').UMAPResponse | null>(null)
  const [retrievalResult, setRetrievalResult] = useState<import('./types').RetrievalResponse | null>(null)
  const [clusterResult, setClusterResult] = useState<import('./types').ClusterResponse | null>(null)
  const [weights, setWeights] = useState({ alpha: 0.4, beta: 0.4, gamma: 0.2 })

  // Error state for "too restrictive" filters (HTTP 400)
  const [filtersRestrictive, setFiltersRestrictive] = useState(false)

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

  // Auto-fetch DataSummary when backend already has data loaded (e.g. after page refresh)
  useEffect(() => {
    if (status?.loaded && dataSummary === null) {
      fetchSummary()
        .then((summary) => setDataSummary(summary))
        .catch(() => {/* non-fatal — user can reload manually */})
    }
  }, [status?.loaded]) // eslint-disable-line react-hooks/exhaustive-deps

  const dataLoaded = dataSummary !== null || status?.loaded === true

  // Don't redirect while status is still loading — wait for the result
  const statusResolved = !statusLoading

  // Show error flash if status call fails
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

  // Show success flash when data becomes loaded
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

  // ── Filter persistence ────────────────────────────────────────────────────
  const handleFiltersChange = useCallback(
    (newFilters: MetadataFilters) => {
      setFilters(newFilters)
      // Clear "too restrictive" warning when filters change
      setFiltersRestrictive(false)
      setFlashItems((prev) => prev.filter((f) => f.id !== 'filters-restrictive'))
      const params = new URLSearchParams(location.search)
      const serialised = filtersToParam(newFilters)
      if (serialised) {
        params.set('filters', serialised)
      } else {
        params.delete('filters')
      }
      navigate({ search: params.toString() }, { replace: true })
    },
    [location.search, navigate],
  )

  // ── Data loaded callback (from DataPanel) ─────────────────────────────────
  const handleDataLoaded = useCallback((summary: DataSummary) => {
    setDataSummary(summary)
    // Invalidate status cache so dataLoaded reflects the new state immediately
    void queryClient.invalidateQueries({ queryKey: ['status'] })
  }, [queryClient])

  // ── UMAP click → set query patient and navigate to Retrieve ──────────────
  const handleQuerySelect = useCallback(
    (patientId: string) => {
      setQueryPatientId(patientId)
      navigate('/retrieve')
    },
    [navigate],
  )

  // ── "Show on UMAP" from RetrievalPanel ────────────────────────────────────
  const handleShowOnUMAP = useCallback(
    (queryId: string, matchIds: string[], querySlideName?: string, neighbourSlides?: string[]) => {
      setQueryHighlightId(queryId)
      setHighlightIds(matchIds)
      setHighlightSlideName(querySlideName)
      setNeighbourSlideNames(neighbourSlides ?? [])
      setHighlightVersion(v => v + 1)
      navigate('/explore')
    },
    [navigate],
  )

  // ── HTTP 409 "not loaded" handler — used by panels ────────────────────────
  const handleNotLoadedError = useCallback(() => {
    setFlashItems((prev) => [
      ...prev.filter((f) => f.id !== 'not-loaded'),
      {
        id: 'not-loaded',
        type: 'warning',
        dismissible: true,
        header: 'Dataset not loaded',
        content: 'Please load a dataset on the Data tab before running this operation.',
        action: (
          <Box>
            <a
              href="#"
              onClick={(e) => {
                e.preventDefault()
                navigate('/data')
              }}
              style={{ color: 'inherit', textDecoration: 'underline', cursor: 'pointer' }}
            >
              Go to Data tab
            </a>
          </Box>
        ),
        onDismiss: () => setFlashItems((prev) => prev.filter((f) => f.id !== 'not-loaded')),
      },
    ])
  }, [navigate])

  // ── HTTP 400 "too restrictive" handler — used by panels ───────────────────
  const handleFiltersRestrictive = useCallback(() => {
    setFiltersRestrictive(true)
    setFlashItems((prev) => [
      ...prev.filter((f) => f.id !== 'filters-restrictive'),
      {
        id: 'filters-restrictive',
        type: 'warning',
        dismissible: true,
        header: 'Filters too restrictive',
        content:
          'The active filters match fewer than 2 slides. Try relaxing your filter selections in the Filter panel.',
        onDismiss: () => {
          setFlashItems((prev) => prev.filter((f) => f.id !== 'filters-restrictive'))
          setFiltersRestrictive(false)
        },
      },
    ])
  }, [])

  // ── Navigation ────────────────────────────────────────────────────────────
  const activeHref = '/' + (location.pathname.split('/')[1] || 'data')

  const navItems: SideNavigationProps.Item[] = NAV_ITEMS.map((item) => ({
    type: 'link' as const,
    text: item.text,
    href: item.href,
  }))

  function handleNavFollow(
    event: CustomEvent<SideNavigationProps.FollowDetail>,
  ): void {
    event.preventDefault()
    const href = event.detail.href
    if (!dataLoaded && DATA_ONLY_PATHS.has(href)) {
      setFlashItems((prev) => [
        ...prev.filter((f) => f.id !== 'nav-blocked'),
        {
          id: 'nav-blocked',
          type: 'info',
          dismissible: true,
          header: 'Load data first',
          content: 'Please load a dataset on the Data tab before navigating here.',
          onDismiss: () => setFlashItems((prev) => prev.filter((f) => f.id !== 'nav-blocked')),
        },
      ])
      return
    }
    navigate(href)
  }

  // ── Loading flash while status is being fetched ───────────────────────────
  const notifications: FlashbarProps.MessageDefinition[] = [
    ...(statusLoading
      ? [
          {
            id: 'status-loading',
            type: 'in-progress' as const,
            loading: true,
            content: 'Checking backend status…',
          },
        ]
      : []),
    ...flashItems,
  ]

  // ── FilterPanel sidebar (shown on Explore / Cluster / Retrieve) ───────────
  const filterableColumns = dataSummary?.filterableColumns ?? []
  const showFilterSidebar =
    filterableColumns.length > 0 &&
    (activeHref === '/explore' || activeHref === '/cluster' || activeHref === '/retrieve')

  const filterSidebar = showFilterSidebar ? (
    <SpaceBetween size="m">
      {filtersRestrictive && (
        <Alert
          type="warning"
          header="Filters too restrictive"
          dismissible
          onDismiss={() => {
            setFiltersRestrictive(false)
            setFlashItems((prev) => prev.filter((f) => f.id !== 'filters-restrictive'))
          }}
        >
          Fewer than 2 slides match the current filters. Relax your selections to proceed.
        </Alert>
      )}
      <FilterPanel
        columns={filterableColumns}
        activeFilters={filters}
        onChange={handleFiltersChange}
      />
    </SpaceBetween>
  ) : undefined

  // ── Content routing ───────────────────────────────────────────────────────
  const content = (
    <Routes>
      <Route path="/data" element={<DataPanel onDataLoaded={handleDataLoaded} initialPath={status?.csvPath} summary={dataSummary} />} />
      <Route
        path="/explore"
        element={
          !statusResolved && dataSummary === null ? loadingFallback : dataLoaded ? (
            <UMAPPanel
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
          ) : (
            <Navigate to="/data" replace />
          )
        }
      />
      <Route
        path="/cluster"
        element={
          !statusResolved && dataSummary === null ? loadingFallback : dataLoaded ? (
            <ClusterPanel
              filters={filters}
              onNotLoadedError={handleNotLoadedError}
              onFiltersRestrictive={handleFiltersRestrictive}
              onClusterResult={setClusterResult}
            />
          ) : (
            <Navigate to="/data" replace />
          )
        }
      />
      <Route
        path="/retrieve"
        element={
          !statusResolved && dataSummary === null ? loadingFallback : dataLoaded ? (
            <RetrievalPanel
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
          ) : (
            <Navigate to="/data" replace />
          )
        }
      />
      <Route
        path="/chat"
        element={
          !statusResolved && dataSummary === null ? loadingFallback : dataLoaded ? (
            <ChatPanel
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
          ) : (
            <Navigate to="/data" replace />
          )
        }
      />
      <Route
        path="/biomarkers"
        element={
          !statusResolved && dataSummary === null ? loadingFallback : dataLoaded ? (
            <BiomarkerPanel onNotLoadedError={handleNotLoadedError} />
          ) : (
            <Navigate to="/data" replace />
          )
        }
      />
      <Route path="*" element={<Navigate to="/data" replace />} />
    </Routes>
  )

  return (
    <AppLayout
      navigation={
        <SideNavigation
          header={{ text: 'Patient Similarity', href: '/data' }}
          activeHref={activeHref}
          items={navItems}
          onFollow={handleNavFollow}
        />
      }
      notifications={<Flashbar items={notifications} stackItems />}
      content={content}
      tools={filterSidebar}
      toolsHide={!showFilterSidebar}
      navigationWidth={220}
    />
  )
}

// Export filter helpers for use in child panels
export { filtersToParam, paramToFilters }
export type { MetadataFilters }
