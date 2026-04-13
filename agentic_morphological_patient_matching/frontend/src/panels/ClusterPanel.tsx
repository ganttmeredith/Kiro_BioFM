import React, { useRef, useState, useEffect, useCallback } from 'react'
import Plotly from 'plotly.js-dist'
import Button from '@cloudscape-design/components/button'
import Container from '@cloudscape-design/components/container'
import Header from '@cloudscape-design/components/header'
import SpaceBetween from '@cloudscape-design/components/space-between'
import FormField from '@cloudscape-design/components/form-field'
import Input from '@cloudscape-design/components/input'
import Multiselect, { MultiselectProps } from '@cloudscape-design/components/multiselect'
import Flashbar, { FlashbarProps } from '@cloudscape-design/components/flashbar'
import Spinner from '@cloudscape-design/components/spinner'
import Box from '@cloudscape-design/components/box'
import KeyValuePairs from '@cloudscape-design/components/key-value-pairs'
import { runClustering, ApiError } from '../api/client'
import { exportClusterHtml } from '../utils/exportHtml'
import type { MetadataFilters, ClusterResponse } from '../types'

interface ClusterPanelProps {
  filters: MetadataFilters
  /** Called when the API returns HTTP 409 "dataset not loaded" */
  onNotLoadedError?: () => void
  /** Called when the API returns HTTP 400 "filters too restrictive" */
  onFiltersRestrictive?: () => void
  /** Lifted state callback — called when clustering completes */
  onClusterResult?: (result: ClusterResponse) => void
}

const DEFAULT_K_MIN = 2
const DEFAULT_K_MAX = 10
const DEFAULT_CROSS_TAB_COLS = ['primary_tumor_site', 'histologic_type', 'hpv_association_p16']

const CROSS_TAB_OPTIONS: MultiselectProps.Option[] = [
  { value: 'primary_tumor_site', label: 'Primary Tumor Site' },
  { value: 'histologic_type', label: 'Histologic Type' },
  { value: 'hpv_association_p16', label: 'HPV Association (p16)' },
  { value: 'pt_stage', label: 'pT Stage' },
  { value: 'pn_stage', label: 'pN Stage' },
  { value: 'smoking_status', label: 'Smoking Status' },
]

export default function ClusterPanel({ filters, onNotLoadedError, onFiltersRestrictive, onClusterResult }: ClusterPanelProps): React.ReactElement {
  const plotDivRef = useRef<HTMLDivElement>(null)

  const [kMin, setKMin] = useState(DEFAULT_K_MIN)
  const [kMax, setKMax] = useState(DEFAULT_K_MAX)
  const [crossTabulateCols, setCrossTabulateCols] = useState<MultiselectProps.Options>(
    DEFAULT_CROSS_TAB_COLS.map((v) => {
      const opt = CROSS_TAB_OPTIONS.find((o) => o.value === v)
      return opt ?? { value: v, label: v }
    }),
  )

  const [loading, setLoading] = useState(false)
  const [clusterResult, setClusterResult] = useState<ClusterResponse | null>(null)
  const [flashItems, setFlashItems] = useState<FlashbarProps.MessageDefinition[]>([])

  // Render Plotly figure whenever result changes
  useEffect(() => {
    if (!clusterResult || !plotDivRef.current) return

    let figure: { data: Plotly.Data[]; layout: Partial<Plotly.Layout> }
    try {
      figure = JSON.parse(clusterResult.plotlyJson) as {
        data: Plotly.Data[]
        layout: Partial<Plotly.Layout>
      }
    } catch {
      return
    }

    Plotly.react(plotDivRef.current, figure.data, figure.layout, { responsive: true })
  }, [clusterResult])

  const handleRunClustering = useCallback(async () => {
    if (kMin < 2 || kMax < kMin) {
      setFlashItems([
        {
          id: 'cluster-validation',
          type: 'error',
          dismissible: true,
          header: 'Invalid parameters',
          content: 'k_min must be ≥ 2 and k_max must be ≥ k_min.',
          onDismiss: () => setFlashItems([]),
        },
      ])
      return
    }

    setLoading(true)
    setFlashItems([])
    try {
      const result = await runClustering({
        kMin,
        kMax,
        crossTabulateCols: crossTabulateCols.map((o) => o.value as string),
        filters,
      })
      setClusterResult(result)
      onClusterResult?.(result)
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
          id: 'cluster-error',
          type: 'error',
          dismissible: true,
          header: 'Clustering failed',
          content: msg,
          onDismiss: () => setFlashItems([]),
        },
      ])
    } finally {
      setLoading(false)
    }
  }, [kMin, kMax, crossTabulateCols, filters, onNotLoadedError, onFiltersRestrictive])

  return (
    <SpaceBetween size="m">
      {flashItems.length > 0 && <Flashbar items={flashItems} />}

      {/* ── Objective description ─────────────────────────────────── */}
      <Container
        header={<Header variant="h2">Morphological Grouping Analysis</Header>}
      >
        <SpaceBetween size="s">
          <Box variant="p">
            This analysis discovers natural patient groupings directly in the{' '}
            <strong>1536-dimensional ABMIL slide embedding space</strong> — not in the 2D UMAP
            projection. Each patient is represented by their mean-pooled slide embedding vector,
            and KMeans clustering (with cosine-equivalent L2-normalised Euclidean distance) is
            swept across a range of k values to find the optimal number of morphological groups.
          </Box>
          <Box variant="p">
            <strong>What this tells you:</strong>
          </Box>
          <ul style={{ margin: '0 0 4px 20px', lineHeight: 1.8, fontSize: 14 }}>
            <li>
              <strong>Silhouette score vs k</strong> — identifies the k where patients are most
              cohesively grouped by morphology. Higher silhouette = tighter, better-separated
              clusters in embedding space.
            </li>
            <li>
              <strong>Cross-tabulation heatmaps</strong> — shows how the discovered morphological
              groups align (or don't) with clinical labels like tumor site, histologic type, and
              HPV status. Strong alignment means the embedding space encodes that clinical
              distinction; weak alignment means morphology and that label are decoupled.
            </li>
          </ul>
          <Box variant="p" color="text-status-inactive">
            <strong>Important caveat:</strong> H-optimus-0 embeddings are pretrained on general
            pan-cancer morphology. Silhouette scores near zero are expected — the embedding space
            is continuous rather than discretely clustered by HNSCC subtype. Use the
            cross-tabulation to identify any dominant clinical signal, then apply it as a metadata
            filter in the Explore or Retrieve tabs to narrow your analysis.
          </Box>
        </SpaceBetween>
      </Container>

      <Container
        header={
          <Header
            variant="h2"
            description="Sweep KMeans over k=k_min..k_max on 1536-d patient embeddings and select the best k by silhouette score"
            actions={
              <Button variant="primary" onClick={handleRunClustering} loading={loading}>
                Run Analysis
              </Button>
            }
          >
            Parameters
          </Header>
        }
      >
        <SpaceBetween size="m">
          <FormField label="k_min" description="Minimum number of groups to evaluate (≥ 2)">
            <Input
              type="number"
              value={String(kMin)}
              onChange={({ detail }) => setKMin(Math.max(2, parseInt(detail.value, 10) || 2))}
            />
          </FormField>

          <FormField label="k_max" description="Maximum number of groups to evaluate (≥ k_min)">
            <Input
              type="number"
              value={String(kMax)}
              onChange={({ detail }) => setKMax(Math.max(kMin, parseInt(detail.value, 10) || kMin))}
            />
          </FormField>

          <FormField
            label="Cross-tabulate against"
            description="Clinical labels to compare against the discovered morphological groups"
          >
            <Multiselect
              selectedOptions={crossTabulateCols}
              options={CROSS_TAB_OPTIONS}
              onChange={({ detail }) => setCrossTabulateCols(detail.selectedOptions)}
              placeholder="Select columns…"
            />
          </FormField>
        </SpaceBetween>
      </Container>

      {loading && (
        <Box textAlign="center" padding="xl">
          <Spinner size="large" />
          <Box variant="p" color="text-status-inactive" padding={{ top: 's' }}>
            Running morphological grouping sweep…
          </Box>
        </Box>
      )}

      {!loading && clusterResult && (
        <SpaceBetween size="m">
          <Container
            header={
              <Header
                variant="h2"
                description="Optimal number of morphological groups and separation quality"
              >
                Results
              </Header>
            }
          >
            <KeyValuePairs
              columns={2}
              items={[
                { label: 'Optimal k (best silhouette)', value: String(clusterResult.bestK) },
                {
                  label: 'Silhouette score',
                  value: `${clusterResult.bestSilhouette.toFixed(4)} ${clusterResult.bestSilhouette < 0.05 ? '— weak structure (expected for HNSCC)' : clusterResult.bestSilhouette < 0.15 ? '— moderate structure' : '— clear structure'}`,
                },
              ]}
            />
          </Container>

          <Container
            header={
              <Header
                variant="h2"
                description="Silhouette score per k (top) and cross-tabulation heatmaps of morphological groups vs clinical labels (bottom)"
                actions={
                  <Button
                    iconName="download"
                    onClick={() => void exportClusterHtml(clusterResult.plotlyJson, clusterResult.bestK, clusterResult.bestSilhouette)}
                  >
                    Export HTML
                  </Button>
                }
              >
                Silhouette Sweep &amp; Clinical Cross-tabulation
              </Header>
            }
          >
            <div ref={plotDivRef} style={{ width: '100%', minHeight: 500 }} />
          </Container>
        </SpaceBetween>
      )}

      {!loading && !clusterResult && (
        <Box textAlign="center" padding="xl" color="text-status-inactive">
          Configure parameters above and click <strong>Run Analysis</strong> to discover
          morphological groupings in the embedding space.
        </Box>
      )}
    </SpaceBetween>
  )
}
