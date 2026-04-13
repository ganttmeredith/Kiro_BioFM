import React, { useEffect, useState, useCallback } from 'react'
import Button from '@cloudscape-design/components/button'
import Container from '@cloudscape-design/components/container'
import ExpandableSection from '@cloudscape-design/components/expandable-section'
import Header from '@cloudscape-design/components/header'
import SpaceBetween from '@cloudscape-design/components/space-between'
import FormField from '@cloudscape-design/components/form-field'
import Slider from '@cloudscape-design/components/slider'
import Toggle from '@cloudscape-design/components/toggle'
import Autosuggest from '@cloudscape-design/components/autosuggest'
import Table from '@cloudscape-design/components/table'
import Box from '@cloudscape-design/components/box'
import Flashbar, { FlashbarProps } from '@cloudscape-design/components/flashbar'
import ProgressBar from '@cloudscape-design/components/progress-bar'
import Spinner from '@cloudscape-design/components/spinner'
import WeightController from '../components/WeightController'
import { fetchPatients, queryRetrieval, buildProfiles, ApiError } from '../api/client'
import { exportRetrievalHtml } from '../utils/exportHtml'
import type { MetadataFilters, PatientMatch, RetrievalResponse } from '../types'

interface RetrievalPanelProps {
  filters: MetadataFilters
  initialPatientId?: string
  onShowOnUMAP?: (queryId: string, matchIds: string[], querySlideName?: string, neighbourSlides?: string[]) => void
  /** Lifted state — persists across tab navigation */
  result?: RetrievalResponse | null
  onResult?: (result: RetrievalResponse) => void
  /** Lifted weights state — persists across tab navigation */
  weights?: { alpha: number; beta: number; gamma: number }
  onWeightsChange?: (w: { alpha: number; beta: number; gamma: number }) => void
  /** p95 of null cosine distribution — used to compute slide_sim_vs_baseline */
  nullP95?: number
  /** Called when the API returns HTTP 409 "dataset not loaded" */
  onNotLoadedError?: () => void
  /** Called when the API returns HTTP 400 "filters too restrictive" */
  onFiltersRestrictive?: () => void
}

const COLUMN_DEFS = [
  { id: 'rank', header: 'Rank', cell: (r: PatientMatch) => r.rank === 0 ? '0 (query)' : r.rank },
  { id: 'patientId', header: 'Patient ID', cell: (r: PatientMatch) => r.patientId },
  { id: 'composite', header: 'Composite', cell: (r: PatientMatch) => r.composite.toFixed(4) },
  { id: 'slideSim', header: 'Slide sim (α)', cell: (r: PatientMatch) => r.slideSim.toFixed(4) },
  { id: 'slideSimVsBaseline', header: 'vs baseline', cell: (r: PatientMatch, baseline?: number) => baseline ? (r.slideSim - baseline).toFixed(4) : '—' },
  { id: 'compSim', header: 'Comp sim (β)', cell: (r: PatientMatch) => r.compSim.toFixed(4) },
  { id: 'metaSim', header: 'Meta sim (γ)', cell: (r: PatientMatch) => r.metaSim.toFixed(4) },
  { id: 'primaryTumorSite', header: 'Tumor site', cell: (r: PatientMatch) => r.primaryTumorSite },
  { id: 'histologicType', header: 'Histologic type', cell: (r: PatientMatch) => r.histologicType },
  { id: 'hpvAssociationP16', header: 'HPV/p16', cell: (r: PatientMatch) => r.hpvAssociationP16 },
  { id: 'pTStage', header: 'pT stage', cell: (r: PatientMatch) => r.ptStage },
  { id: 'pNStage', header: 'pN stage', cell: (r: PatientMatch) => r.pnStage },
  { id: 'smokingStatus', header: 'Smoking', cell: (r: PatientMatch) => r.smokingStatus },
  { id: 'ageAtInitialDiagnosis', header: 'Age at Diagnosis', cell: (r: PatientMatch) => r.ageAtInitialDiagnosis ?? '—' },
  { id: 'yearOfInitialDiagnosis', header: 'Year of Diagnosis', cell: (r: PatientMatch) => r.yearOfInitialDiagnosis ?? '—' },
]

// SSE progress event shape from /api/profiles/build
interface BuildProgressEvent {
  progress?: number
  message?: string
  done?: boolean
  error?: string
}

export default function RetrievalPanel({
  filters,
  initialPatientId = '',
  onShowOnUMAP,
  result: externalResult,
  onResult,
  weights: externalWeights,
  onWeightsChange,
  nullP95,
  onNotLoadedError,
  onFiltersRestrictive,
}: RetrievalPanelProps): React.ReactElement {
  const [patients, setPatients] = useState<string[]>([])
  const [patientId, setPatientId] = useState(initialPatientId)
  const [k, setK] = useState(10)
  const [deduplicate, setDeduplicate] = useState(true)
  const [localWeights, setLocalWeights] = useState({ alpha: 0.4, beta: 0.4, gamma: 0.2 })
  const weights = externalWeights ?? localWeights
  const handleWeightsChange = useCallback(
    (w: { alpha: number; beta: number; gamma: number }) => {
      setLocalWeights(w)
      onWeightsChange?.(w)
    },
    [onWeightsChange],
  )
  const [loading, setLoading] = useState(false)
  const [localResult, setLocalResult] = useState<RetrievalResponse | null>(null)
  // Use external (lifted) result if provided
  const result = externalResult !== undefined ? externalResult : localResult
  const [flashItems, setFlashItems] = useState<FlashbarProps.MessageDefinition[]>([])

  // Profile build state
  const [profilesNotBuilt, setProfilesNotBuilt] = useState(false)
  const [buildInProgress, setBuildInProgress] = useState(false)
  const [buildProgress, setBuildProgress] = useState(0)
  const [buildMessage, setBuildMessage] = useState('')

  // Sync initialPatientId prop changes (e.g. from UMAP click)
  useEffect(() => {
    if (initialPatientId) setPatientId(initialPatientId)
  }, [initialPatientId])

  // Load patient list for autocomplete
  useEffect(() => {
    fetchPatients()
      .then(setPatients)
      .catch(() => {/* non-fatal — autocomplete just won't have suggestions */})
  }, [])

  const autosuggestOptions = patients.map((p) => ({ value: p }))

  const addFlash = useCallback((item: FlashbarProps.MessageDefinition) => {
    setFlashItems((prev) => [...prev.filter((f) => f.id !== item.id), item])
  }, [])

  const dismissFlash = useCallback((id: string) => {
    setFlashItems((prev) => prev.filter((f) => f.id !== id))
  }, [])

  const handleSubmit = useCallback(async () => {
    if (!patientId.trim()) {
      addFlash({
        id: 'no-patient',
        type: 'warning',
        dismissible: true,
        header: 'No patient selected',
        content: 'Please enter or select a patient ID before querying.',
        onDismiss: () => dismissFlash('no-patient'),
      })
      return
    }

    setLoading(true)
    setFlashItems([])
    setProfilesNotBuilt(false)
    try {
      const res = await queryRetrieval({
        patientId: patientId.trim(),
        k,
        alpha: weights.alpha,
        beta: weights.beta,
        gamma: weights.gamma,
        filters,
        deduplicateByPatient: deduplicate,
        slideAggregation: 'max',
      })
      setLocalResult(res)
      onResult?.(res)
    } catch (err) {
      if (err instanceof ApiError && err.status === 409) {
        const msg = err.message.toLowerCase()
        if (msg.includes('profile') || msg.includes('composition')) {
          // Profiles not built
          setProfilesNotBuilt(true)
        } else {
          // Dataset not loaded
          onNotLoadedError?.()
        }
        return
      }
      if (err instanceof ApiError && err.status === 400) {
        onFiltersRestrictive?.()
        return
      }
      const msg = err instanceof Error ? err.message : 'Unknown error'
      addFlash({
        id: 'retrieval-error',
        type: 'error',
        dismissible: true,
        header: 'Retrieval failed',
        content: msg,
        onDismiss: () => dismissFlash('retrieval-error'),
      })
    } finally {
      setLoading(false)
    }
  }, [patientId, k, weights, filters, deduplicate, addFlash, dismissFlash, onNotLoadedError, onFiltersRestrictive])

  // ── Build profiles via SSE ────────────────────────────────────────────────
  const handleBuildProfiles = useCallback(async () => {
    setBuildInProgress(true)
    setBuildProgress(0)
    setBuildMessage('Starting profile build…')
    setFlashItems([])

    try {
      const reader = await buildProfiles()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += value
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''

        for (const line of lines) {
          if (!line.startsWith('data:')) continue
          const json = line.slice('data:'.length).trim()
          if (!json) continue
          try {
            const event = JSON.parse(json) as BuildProgressEvent
            if (event.error) {
              addFlash({
                id: 'build-error',
                type: 'error',
                dismissible: true,
                header: 'Profile build failed',
                content: event.error,
                onDismiss: () => dismissFlash('build-error'),
              })
              setBuildInProgress(false)
              return
            }
            if (event.progress !== undefined) {
              setBuildProgress(Math.round(event.progress * 100))
            }
            if (event.message) {
              setBuildMessage(event.message)
            }
            if (event.done) {
              setBuildProgress(100)
              setBuildMessage('Profiles built successfully.')
              setProfilesNotBuilt(false)
              setBuildInProgress(false)
              addFlash({
                id: 'build-success',
                type: 'success',
                dismissible: true,
                header: 'Profiles ready',
                content: 'Composition profiles have been built. You can now run retrieval queries.',
                onDismiss: () => dismissFlash('build-success'),
              })
              return
            }
          } catch {
            // ignore malformed SSE lines
          }
        }
      }
      // Stream ended without explicit done
      setBuildProgress(100)
      setBuildMessage('Build complete.')
      setProfilesNotBuilt(false)
      setBuildInProgress(false)
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Unknown error'
      addFlash({
        id: 'build-error',
        type: 'error',
        dismissible: true,
        header: 'Profile build failed',
        content: msg,
        onDismiss: () => dismissFlash('build-error'),
      })
      setBuildInProgress(false)
    }
  }, [addFlash, dismissFlash])

  const handleShowOnUMAP = useCallback(() => {
    if (!result || !onShowOnUMAP) return
    const matchIds = result.matches.map((m) => m.patientId)
    onShowOnUMAP(
      result.queryPatient.patientId,
      matchIds,
      result.querySlideName,
      result.neighborSlideNames,
    )
  }, [result, onShowOnUMAP])

  // Rows: query patient (rank 0) + matches
  const tableItems: PatientMatch[] = result
    ? [result.queryPatient, ...result.matches]
    : []

  return (
    <SpaceBetween size="m">
      {flashItems.length > 0 && <Flashbar items={flashItems} />}

      {/* ── Profiles not built banner ──────────────────────────────────── */}
      {profilesNotBuilt && !buildInProgress && (
        <Flashbar
          items={[
            {
              id: 'profiles-not-built',
              type: 'warning',
              header: 'Composition profiles not built',
              content:
                'Composition profiles are required for retrieval. Build them now to enable composite similarity scoring.',
              action: (
                <Button onClick={() => void handleBuildProfiles()} variant="normal">
                  Build Profiles
                </Button>
              ),
            },
          ]}
        />
      )}

      {/* ── SSE build progress bar ─────────────────────────────────────── */}
      {buildInProgress && (
        <Container header={<Header variant="h2">Building Composition Profiles</Header>}>
          <SpaceBetween size="s">
            <ProgressBar
              value={buildProgress}
              label="Build progress"
              description={buildMessage}
              status={buildProgress === 100 ? 'success' : 'in-progress'}
            />
          </SpaceBetween>
        </Container>
      )}

      {/* ── Objective description ──────────────────────────────────────── */}
      <Container header={<Header variant="h2">Patient Matcher</Header>}>
        <SpaceBetween size="s">
          <Box variant="p">
            Given a query patient, this tool ranks every other patient in the cohort by a{' '}
            <strong>composite similarity score</strong> that combines three independent signals:
          </Box>
          <ul style={{ margin: '0 0 4px 20px', lineHeight: 1.9, fontSize: 14 }}>
            <li>
              <strong>Slide morphology (α)</strong> — cosine similarity in the 1536-dimensional
              ABMIL-aggregated H-optimus-0 embedding space. Captures overall tissue architecture
              and cellular morphology. Max-aggregated across all slide pairs between the two patients.
            </li>
            <li>
              <strong>Tissue composition (β)</strong> — cosine similarity of patch-level
              morphological cluster proportion profiles. Captures how much of each tissue type
              (tumour epithelium, stroma, necrosis, etc.) each patient's slides contain —
              complementary signal that mean-pooling discards.
            </li>
            <li>
              <strong>Clinical metadata (γ)</strong> — fraction of exactly-matching fields across
              primary tumor site, histologic type, HPV/p16 status, pT stage, and pN stage.
            </li>
          </ul>
          <Box variant="p">
            <strong>What you will learn:</strong> which patients in the HANCOCK cohort are most
            similar to your patient of interest, how that similarity breaks down across morphology
            vs composition vs clinical features, and whether the slide similarity score is
            meaningfully above the population baseline (p95 ≈ 0.66). Use the weight sliders to
            emphasise the signal most relevant to your question — e.g. increase α to prioritise
            pure morphological matches, or increase γ to enforce clinical concordance.
          </Box>
          <Box variant="p" color="text-status-inactive">
            Results can be pushed to the <strong>Explore</strong> tab to visualise where matched
            patients sit in the embedding space.
          </Box>
        </SpaceBetween>
      </Container>

      {/* ── Weight tuning guidance ─────────────────────────────────── */}
      <Container>
        <ExpandableSection headerText="How to tune the similarity weights (α, β, γ)" variant="default">
          <SpaceBetween size="s">
            <Box variant="p">
              The three weights control what "similar" means. They must sum to 1.0. Here's when to adjust each:
            </Box>

            <Box variant="p"><strong>Increase α (slide morphology) when:</strong></Box>
            <ul style={{ margin: '0 0 4px 20px', lineHeight: 1.8, fontSize: 14 }}>
              <li>You want purely morphological matches — patients whose H&E slides look the same regardless of clinical labels</li>
              <li>Clinical metadata is incomplete or unreliable for your cohort</li>
              <li>You're exploring whether morphology predicts an outcome independent of staging</li>
            </ul>

            <Box variant="p"><strong>Increase β (tissue composition) when:</strong></Box>
            <ul style={{ margin: '0 0 4px 20px', lineHeight: 1.8, fontSize: 14 }}>
              <li>You care about tumour microenvironment — proportion of stroma, necrosis, immune infiltrate</li>
              <li>Slide-level embeddings are giving surprising results (check: does comp_sim agree with slide_sim?)</li>
              <li>You've validated that composition profiles separate your label of interest (run "validate composition profiles" in Chat)</li>
            </ul>

            <Box variant="p"><strong>Increase γ (clinical metadata) when:</strong></Box>
            <ul style={{ margin: '0 0 4px 20px', lineHeight: 1.8, fontSize: 14 }}>
              <li>You need matches that share specific clinical features — same HPV status, same pT stage, same histologic type</li>
              <li>You're building a trial-eligible cohort and clinical criteria are hard constraints</li>
              <li>Morphological similarity alone is returning clinically heterogeneous results</li>
            </ul>

            <Box variant="p"><strong>Diagnosing unexpected results:</strong></Box>
            <ul style={{ margin: '0 0 4px 20px', lineHeight: 1.8, fontSize: 14 }}>
              <li>Top matches have high composite but different histology → α and β are dominating; increase γ</li>
              <li>Top matches share clinical features but low slide_sim → γ is dominating; the morphology doesn't agree with the metadata match</li>
              <li>comp_sim is consistently near 1.0 for all matches → composition profiles may not be discriminative; reduce β and validate profiles in Chat</li>
              <li>slide_sim ≤ 0.66 for all matches → scores are within population noise (p95 baseline ≈ 0.66); no patient is genuinely morphologically similar</li>
            </ul>

            <Box variant="p" color="text-status-inactive">
              The "vs baseline" column in results shows slide_sim minus the population mean (~0.44). Positive values indicate genuine morphological similarity above chance.
            </Box>
          </SpaceBetween>
        </ExpandableSection>
      </Container>

      <Container
        header={
          <Header
            variant="h2"
            description="Composite score = α × slide morphology + β × tissue composition + γ × clinical metadata"
            actions={
              <SpaceBetween size="s" direction="horizontal">
                {result && onShowOnUMAP && (
                  <Button onClick={handleShowOnUMAP}>Show on UMAP</Button>
                )}
                <Button
                  variant="primary"
                  onClick={() => void handleSubmit()}
                  loading={loading}
                  disabled={buildInProgress}
                >
                  Find Similar Patients
                </Button>
              </SpaceBetween>
            }
          >
            Query
          </Header>
        }
      >
        <SpaceBetween size="m">
          <FormField label="Query patient ID">
            <Autosuggest
              value={patientId}
              options={autosuggestOptions}
              onChange={({ detail }) => setPatientId(detail.value)}
              onSelect={({ detail }) => setPatientId(detail.value)}
              placeholder="Type or select a patient ID…"
              enteredTextLabel={(v) => `Use "${v}"`}
            />
          </FormField>

          <FormField label="Top-k results" description={`Return up to ${k} matches`}>
            <Slider min={1} max={50} step={1} value={k} onChange={({ detail }) => setK(detail.value)} />
          </FormField>

          <FormField label="Deduplicate by patient">
            <Toggle
              checked={deduplicate}
              onChange={({ detail }) => setDeduplicate(detail.checked)}
            >
              {deduplicate ? 'One result per patient' : 'All slides included'}
            </Toggle>
          </FormField>

          <FormField label="Similarity weights" description="α + β + γ must equal 1.0">
            <WeightController
              alpha={weights.alpha}
              beta={weights.beta}
              gamma={weights.gamma}
              onChange={handleWeightsChange}
            />
          </FormField>
        </SpaceBetween>
      </Container>

      {loading && (
        <Box textAlign="center" padding="xl">
          <Spinner size="large" />
          <Box variant="p" color="text-status-inactive" padding={{ top: 's' }}>
            Running retrieval…
          </Box>
        </Box>
      )}

      {!loading && result && (
        <Container
          header={
            <Header
              variant="h2"
              description={`Query: patient ${result.queryPatient.patientId} · ${result.matches.length} similar patients ranked by composite score`}
              actions={
                <Button iconName="download" onClick={() => exportRetrievalHtml(result)}>
                  Export HTML
                </Button>
              }
            >
              Similar Patients
            </Header>
          }
        >
          <Table
            items={tableItems}
            columnDefinitions={COLUMN_DEFS.map(col => ({
              ...col,
              cell: col.id === 'slideSimVsBaseline'
                ? (r: PatientMatch) => col.cell(r, nullP95)
                : col.cell,
            }))}
            variant="embedded"
            stripedRows
            resizableColumns
            getRowId={(item: PatientMatch) => `${item.rank}-${item.patientId}`}
            empty={
              <Box textAlign="center" color="text-status-inactive">
                No matches found.
              </Box>
            }
          />
        </Container>
      )}

      {!loading && !result && !profilesNotBuilt && !buildInProgress && (
        <Box textAlign="center" padding="xl" color="text-status-inactive">
          Select a patient and click <strong>Query</strong> to retrieve similar patients.
        </Box>
      )}
    </SpaceBetween>
  )
}
