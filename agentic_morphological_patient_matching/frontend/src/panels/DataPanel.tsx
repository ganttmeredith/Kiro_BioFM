import React, { useEffect, useRef, useState } from 'react'
import Button from '@cloudscape-design/components/button'
import Container from '@cloudscape-design/components/container'
import FormField from '@cloudscape-design/components/form-field'
import Header from '@cloudscape-design/components/header'
import Input from '@cloudscape-design/components/input'
import Modal from '@cloudscape-design/components/modal'
import SpaceBetween from '@cloudscape-design/components/space-between'
import Tabs from '@cloudscape-design/components/tabs'
import Table from '@cloudscape-design/components/table'
import Alert from '@cloudscape-design/components/alert'
import Box from '@cloudscape-design/components/box'
import StatusIndicator from '@cloudscape-design/components/status-indicator'
import Plotly from 'plotly.js-dist'
import { loadData } from '../api/client'
import type { DataSummary } from '../types'

interface DataPanelProps {
  onDataLoaded: (summary: DataSummary) => void
  /** Pre-filled path from status (last loaded path) */
  initialPath?: string
  /** Lifted summary state — persists across tab navigation */
  summary?: DataSummary | null
}

export default function DataPanel({ onDataLoaded, initialPath, summary: externalSummary }: DataPanelProps): React.ReactElement {
  const [modalOpen, setModalOpen] = useState(false)
  const [localPath, setLocalPath] = useState(initialPath ?? '')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [s3Path, setS3Path] = useState('')
  const [activeTab, setActiveTab] = useState('local')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [localSummary, setLocalSummary] = useState<DataSummary | null>(null)
  const summary = externalSummary !== undefined ? externalSummary : localSummary

  const histogramRef = useRef<HTMLDivElement>(null)

  // Pre-fill path when initialPath arrives (from status on page load)
  useEffect(() => {
    if (initialPath && !localPath) setLocalPath(initialPath)
  }, [initialPath]) // eslint-disable-line react-hooks/exhaustive-deps

  // Render histogram whenever summary changes
  useEffect(() => {
    if (!summary || !histogramRef.current) return
    const { mean, std, percentile5, percentile95 } = summary.nullDistribution
    void Plotly.react(
      histogramRef.current,
      [{ type: 'bar', x: ['p5', 'mean', 'mean+std', 'p95'], y: [percentile5, mean, mean + std, percentile95], marker: { color: '#0972d3' } }],
      { title: { text: 'Null Cosine Similarity Distribution' }, xaxis: { title: { text: 'Statistic' } }, yaxis: { title: { text: 'Cosine Similarity' } }, margin: { t: 40, r: 20, b: 50, l: 60 }, height: 280 },
      { responsive: true, displayModeBar: false },
    )
  }, [summary])

  const activePath = activeTab === 's3' ? s3Path.trim() : localPath.trim()

  async function handleLoad(): Promise<void> {
    if (activeTab === 'local' && selectedFile) {
      // Upload the actual file
      setLoading(true)
      setError(null)
      try {
        const formData = new FormData()
        formData.append('file', selectedFile)
        const res = await fetch('/api/data/upload', { method: 'POST', body: formData })
        if (!res.ok) {
          const detail = await res.json().catch(() => ({ detail: res.statusText }))
          throw new Error(detail?.detail ?? res.statusText)
        }
        const result = await res.json() as DataSummary
        setLocalSummary(result)
        onDataLoaded(result)
        setModalOpen(false)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load data')
      } finally {
        setLoading(false)
      }
      return
    }

    // Path-based load (local path typed manually or S3)
    const path = activeTab === 's3' ? s3Path.trim() : localPath.trim()
    if (!path) return
    setLoading(true)
    setError(null)
    try {
      const result = await loadData({ csvPath: path })
      setLocalSummary(result)
      onDataLoaded(result)
      setModalOpen(false)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data')
    } finally {
      setLoading(false)
    }
  }

  return (
    <Box padding="l">
      <SpaceBetween size="l">
        {/* ── Load button + status ──────────────────────────────────────── */}
        <Container header={<Header variant="h2">Dataset</Header>}>
          <SpaceBetween size="m">
            {summary ? (
              <SpaceBetween size="s">
                <StatusIndicator type="success">
                  Loaded — {summary.nSlides} slides · {summary.nPatients} patients
                </StatusIndicator>
                <Box color="text-status-inactive" fontSize="body-s">
                  {activePath || localPath}
                </Box>
                <Button onClick={() => setModalOpen(true)}>Change dataset</Button>
              </SpaceBetween>
            ) : (
              <SpaceBetween size="s">
                <Box color="text-status-inactive">No dataset loaded.</Box>
                <Button variant="primary" onClick={() => setModalOpen(true)}>
                  Load dataset
                </Button>
              </SpaceBetween>
            )}
          </SpaceBetween>
        </Container>

        {/* ── Load modal ───────────────────────────────────────────────── */}
        <Modal
          visible={modalOpen}
          onDismiss={() => { setModalOpen(false); setError(null) }}
          header="Load dataset"
          footer={
            <Box float="right">
              <SpaceBetween direction="horizontal" size="xs">
                <Button variant="link" onClick={() => { setModalOpen(false); setError(null) }}>
                  Cancel
                </Button>
                <Button
                  variant="primary"
                  onClick={() => void handleLoad()}
                  loading={loading}
                  disabled={activeTab === 'local' ? (!selectedFile && !localPath.trim()) : !s3Path.trim()}
                >
                  Load
                </Button>
              </SpaceBetween>
            </Box>
          }
        >
          <SpaceBetween size="m">
            <Tabs
              activeTabId={activeTab}
              onChange={({ detail }) => { setActiveTab(detail.activeTabId); setError(null) }}
              tabs={[
                {
                  id: 'local',
                  label: 'Local path',
                  content: (
                    <SpaceBetween size="s">
                      <FormField
                        label="File path"
                        description="Relative or absolute path to enriched_slide_embeddings.csv"
                      >
                        <Input
                          value={localPath}
                          onChange={({ detail }) => setLocalPath(detail.value)}
                          placeholder="02_Morphological_Patient_Similarity_Retrieval/enriched_slide_embeddings.csv"
                          disabled={loading}
                          onKeyDown={({ detail }) => { if (detail.key === 'Enter') void handleLoad() }}
                        />
                      </FormField>
                      <Button
                        iconName="upload"
                        onClick={() => {
                          const input = document.createElement('input')
                          input.type = 'file'
                          input.accept = '.csv'
                          input.onchange = (e) => {
                            const file = (e.target as HTMLInputElement).files?.[0]
                            if (file) {
                              setSelectedFile(file)
                              setLocalPath(file.name)
                            }
                          }
                          input.click()
                        }}
                      >
                        Browse…
                      </Button>
                    </SpaceBetween>
                  ),
                },
                {
                  id: 's3',
                  label: 'S3 path',
                  content: (
                    <FormField
                      label="S3 URI"
                      description="Full S3 URI to the CSV file (requires AWS_PROFILE=pathologyworkshop)"
                    >
                      <Input
                        value={s3Path}
                        onChange={({ detail }) => setS3Path(detail.value)}
                        placeholder="s3://my-bucket/path/to/enriched_slide_embeddings.csv"
                        disabled={loading}
                        onKeyDown={({ detail }) => { if (detail.key === 'Enter') void handleLoad() }}
                      />
                    </FormField>
                  ),
                },
              ]}
            />
            {error && (
              <Alert type="error" header="Load failed" dismissible onDismiss={() => setError(null)}>
                {error}
              </Alert>
            )}
          </SpaceBetween>
        </Modal>

        {/* ── Summary + diagnostics ─────────────────────────────────────── */}
        {summary && (
          <>
            <Container header={<Header variant="h2">Embedding Silhouette Scores</Header>}>
              <Table
                items={summary.silhouetteScores}
                columnDefinitions={[
                  { id: 'column', header: 'Metadata Column', cell: (r) => r.column, sortingField: 'column' },
                  { id: 'score', header: 'Silhouette Score', cell: (r) => r.score.toFixed(4), sortingField: 'score' },
                ]}
                variant="embedded"
                empty={<Box textAlign="center" color="inherit">No scores available</Box>}
              />
            </Container>

            <Container header={<Header variant="h2">Null Cosine Distribution</Header>}>
              <div ref={histogramRef} style={{ width: '100%' }} />
            </Container>
          </>
        )}
      </SpaceBetween>
    </Box>
  )
}
