import React, { useState, useRef, useEffect, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import Alert from '@cloudscape-design/components/alert'
import Box from '@cloudscape-design/components/box'
import Button from '@cloudscape-design/components/button'
import Header from '@cloudscape-design/components/header'
import Input from '@cloudscape-design/components/input'
import Modal from '@cloudscape-design/components/modal'
import SpaceBetween from '@cloudscape-design/components/space-between'
import Spinner from '@cloudscape-design/components/spinner'
import Table from '@cloudscape-design/components/table'
import type { AppContext, ChatMessage } from '../types'

// ─── Artifact types ───────────────────────────────────────────────────────────

interface RetrievalRow {
  rank: number
  patientId: string
  composite: number
  slideSim: number
  compSim: number
  metaSim: number
  primaryTumorSite?: string
  histologicType?: string
  hpvAssociationP16?: string
  ptStage?: string
  pnStage?: string
  smokingStatus?: string
  ageAtInitialDiagnosis?: string | number
}

interface RetrievalArtifact {
  type: 'retrieval'
  queryPatient: RetrievalRow
  matches: RetrievalRow[]
}

interface PlotlyArtifact {
  type: 'plotly'
  plotlyJson: string
  title: string
  pngBase64?: string  // captured after render for report inclusion
}

interface SearchPatientRow {
  patient_id: string
  age_at_initial_diagnosis: string | number
  year_of_initial_diagnosis: string | number
  primary_tumor_site: string
  histologic_type: string
  hpv_association_p16: string
  pt_stage: string
  pn_stage: string
  smoking_status: string
  sex: string
  survival_status: string
  recurrence: string
}

interface SearchArtifact {
  type: 'search'
  total: number
  patients: SearchPatientRow[]
  query?: string
}

type Artifact = RetrievalArtifact | PlotlyArtifact | SearchArtifact

interface ChatEntry {
  role: 'user' | 'assistant'
  content: string
  artifacts?: Artifact[]
  usedEmbeddings?: boolean  // true if any e_ embedding tool was called (run_retrieval, run_umap, etc.)
  usedTools?: boolean       // true if any tool was called (including search, plot, etc.)
}

// ─── Markdown renderer (no external deps) ────────────────────────────────────

function renderInline(text: string): React.ReactNode[] {
  // Split on **bold**, *italic*, `code`
  const parts = text.split(/(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g)
  return parts.map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**'))
      return <strong key={i}>{part.slice(2, -2)}</strong>
    if (part.startsWith('*') && part.endsWith('*'))
      return <em key={i}>{part.slice(1, -1)}</em>
    if (part.startsWith('`') && part.endsWith('`'))
      return (
        <code key={i} style={{
          background: '#e9ebed', borderRadius: 3, padding: '1px 5px',
          fontFamily: 'monospace', fontSize: '0.88em',
        }}>
          {part.slice(1, -1)}
        </code>
      )
    return part
  })
}

function MarkdownText({ text }: { text: string }): React.ReactElement {
  const lines = text.split('\n')
  const nodes: React.ReactNode[] = []
  let i = 0

  while (i < lines.length) {
    const line = lines[i]

    // Fenced code block
    if (line.startsWith('```')) {
      const lang = line.slice(3).trim()
      const codeLines: string[] = []
      i++
      while (i < lines.length && !lines[i].startsWith('```')) {
        codeLines.push(lines[i])
        i++
      }
      nodes.push(
        <pre key={i} style={{
          background: '#1a1a2e', color: '#e2e8f0', borderRadius: 6,
          padding: '10px 14px', overflowX: 'auto', fontSize: 13,
          fontFamily: 'monospace', margin: '6px 0', lineHeight: 1.5,
        }}>
          {lang && <div style={{ color: '#94a3b8', fontSize: 11, marginBottom: 4 }}>{lang}</div>}
          {codeLines.join('\n')}
        </pre>
      )
      i++
      continue
    }

    // Heading
    const hMatch = line.match(/^(#{1,3})\s+(.+)/)
    if (hMatch) {
      const level = hMatch[1].length
      const sizes = ['1.1em', '1em', '0.95em']
      nodes.push(
        <div key={i} style={{
          fontWeight: 700, fontSize: sizes[level - 1],
          marginTop: level === 1 ? 10 : 6, marginBottom: 2,
          borderBottom: level === 1 ? '1px solid #d1d5db' : undefined,
          paddingBottom: level === 1 ? 3 : undefined,
        }}>
          {renderInline(hMatch[2])}
        </div>
      )
      i++
      continue
    }

    // Unordered list item
    const ulMatch = line.match(/^[-*]\s+(.+)/)
    if (ulMatch) {
      const items: string[] = []
      while (i < lines.length && lines[i].match(/^[-*]\s+/)) {
        items.push(lines[i].replace(/^[-*]\s+/, ''))
        i++
      }
      nodes.push(
        <ul key={i} style={{ margin: '4px 0', paddingLeft: 20 }}>
          {items.map((item, j) => (
            <li key={j} style={{ marginBottom: 2 }}>{renderInline(item)}</li>
          ))}
        </ul>
      )
      continue
    }

    // Ordered list item
    const olMatch = line.match(/^\d+\.\s+(.+)/)
    if (olMatch) {
      const items: string[] = []
      while (i < lines.length && lines[i].match(/^\d+\.\s+/)) {
        items.push(lines[i].replace(/^\d+\.\s+/, ''))
        i++
      }
      nodes.push(
        <ol key={i} style={{ margin: '4px 0', paddingLeft: 20 }}>
          {items.map((item, j) => (
            <li key={j} style={{ marginBottom: 2 }}>{renderInline(item)}</li>
          ))}
        </ol>
      )
      continue
    }

    // Markdown table — starts with a pipe character
    if (line.trimStart().startsWith('|')) {
      const tableLines: string[] = []
      while (i < lines.length && lines[i].trimStart().startsWith('|')) {
        tableLines.push(lines[i])
        i++
      }
      // Parse rows: split on |, trim, drop empty first/last cells from leading/trailing pipes
      const parseRow = (l: string) =>
        l.split('|').slice(1, -1).map(c => c.trim())
      const isSeparator = (l: string) => /^[\s|:\-]+$/.test(l)

      const headers = parseRow(tableLines[0])
      const dataRows = tableLines
        .slice(1)
        .filter(l => !isSeparator(l))
        .map(parseRow)

      nodes.push(
        <div key={i} style={{ overflowX: 'auto', margin: '6px 0' }}>
          <table style={{
            borderCollapse: 'collapse', fontSize: 13, width: '100%',
            border: '1px solid #d1d5db',
          }}>
            <thead>
              <tr style={{ background: '#f3f4f6' }}>
                {headers.map((h, j) => (
                  <th key={j} style={{
                    padding: '5px 10px', textAlign: 'left', fontWeight: 600,
                    borderBottom: '2px solid #d1d5db', borderRight: '1px solid #e5e7eb',
                    whiteSpace: 'nowrap',
                  }}>
                    {renderInline(h)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {dataRows.map((row, ri) => (
                <tr key={ri} style={{ background: ri % 2 === 0 ? '#fff' : '#f9fafb' }}>
                  {row.map((cell, ci) => (
                    <td key={ci} style={{
                      padding: '4px 10px', borderBottom: '1px solid #e5e7eb',
                      borderRight: '1px solid #e5e7eb',
                    }}>
                      {renderInline(cell)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )
      continue
    }

    // Horizontal rule
    if (line.match(/^---+$/)) {
      nodes.push(<hr key={i} style={{ border: 'none', borderTop: '1px solid #d1d5db', margin: '6px 0' }} />)
      i++
      continue
    }

    // Blank line → small gap
    if (line.trim() === '') {
      nodes.push(<div key={i} style={{ height: 6 }} />)
      i++
      continue
    }

    // Normal paragraph line
    nodes.push(
      <div key={i} style={{ lineHeight: 1.6 }}>
        {renderInline(line)}
      </div>
    )
    i++
  }

  return <div style={{ fontSize: 14 }}>{nodes}</div>
}

// ─── Inline retrieval table ───────────────────────────────────────────────────

const RETRIEVAL_COLS = [
  { id: 'rank', header: 'Rank', cell: (r: RetrievalRow) => r.rank === 0 ? '0 (query)' : r.rank, width: 70, sortingField: 'rank' },
  { id: 'patientId', header: 'Patient', cell: (r: RetrievalRow) => r.patientId, width: 80, sortingField: 'patientId' },
  { id: 'composite', header: 'Composite', cell: (r: RetrievalRow) => r.composite.toFixed(4), width: 90, sortingField: 'composite' },
  { id: 'slideSim', header: 'Slide (α)', cell: (r: RetrievalRow) => r.slideSim.toFixed(4), width: 85, sortingField: 'slideSim' },
  { id: 'compSim', header: 'Comp (β)', cell: (r: RetrievalRow) => r.compSim.toFixed(4), width: 85, sortingField: 'compSim' },
  { id: 'metaSim', header: 'Meta (γ)', cell: (r: RetrievalRow) => r.metaSim.toFixed(4), width: 85, sortingField: 'metaSim' },
  { id: 'primaryTumorSite', header: 'Tumor site', cell: (r: RetrievalRow) => r.primaryTumorSite ?? '—', sortingField: 'primaryTumorSite' },
  { id: 'histologicType', header: 'Histology', cell: (r: RetrievalRow) => r.histologicType ?? '—', sortingField: 'histologicType' },
  { id: 'hpv', header: 'HPV/p16', cell: (r: RetrievalRow) => r.hpvAssociationP16 ?? '—', width: 80, sortingField: 'hpvAssociationP16' },
  { id: 'ptStage', header: 'pT', cell: (r: RetrievalRow) => r.ptStage ?? '—', width: 55, sortingField: 'ptStage' },
  { id: 'pnStage', header: 'pN', cell: (r: RetrievalRow) => r.pnStage ?? '—', width: 55, sortingField: 'pnStage' },
  { id: 'smoking', header: 'Smoking', cell: (r: RetrievalRow) => r.smokingStatus ?? '—', sortingField: 'smokingStatus' },
  { id: 'age', header: 'Age', cell: (r: RetrievalRow) => r.ageAtInitialDiagnosis ?? '—', width: 60, sortingField: 'ageAtInitialDiagnosis' },
]

const PAGE_SIZE = 5

function PaginationBar({ page, totalPages, onPrev, onNext }: {
  page: number; totalPages: number; onPrev: () => void; onNext: () => void
}) {
  if (totalPages <= 1) return null
  return (
    <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
      <button
        onClick={onPrev} disabled={page === 0}
        style={{ background: 'none', border: '1px solid #aab7c4', borderRadius: 4, padding: '1px 7px', cursor: page === 0 ? 'default' : 'pointer', opacity: page === 0 ? 0.4 : 1 }}
        aria-label="Previous page"
      >‹</button>
      <span>{page + 1} / {totalPages}</span>
      <button
        onClick={onNext} disabled={page === totalPages - 1}
        style={{ background: 'none', border: '1px solid #aab7c4', borderRadius: 4, padding: '1px 7px', cursor: page === totalPages - 1 ? 'default' : 'pointer', opacity: page === totalPages - 1 ? 0.4 : 1 }}
        aria-label="Next page"
      >›</button>
    </span>
  )
}

function InlineRetrievalTable({ artifact }: { artifact: RetrievalArtifact }) {
  const rows = [artifact.queryPatient, ...artifact.matches]
  const [sortCol, setSortCol] = useState<string>('rank')
  const [sortDesc, setSortDesc] = useState(false)
  const [page, setPage] = useState(0)

  const sorted = [...rows].sort((a, b) => {
    const col = RETRIEVAL_COLS.find(c => c.id === sortCol)
    if (!col) return 0
    const av = (a as Record<string, unknown>)[col.sortingField ?? col.id]
    const bv = (b as Record<string, unknown>)[col.sortingField ?? col.id]
    if (av == null) return 1
    if (bv == null) return -1
    const cmp = av < bv ? -1 : av > bv ? 1 : 0
    return sortDesc ? -cmp : cmp
  })

  const totalPages = Math.ceil(sorted.length / PAGE_SIZE)
  const pageItems = sorted.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)

  return (
    <div style={{ marginTop: 8, overflowX: 'auto' }}>
      <div style={{ fontSize: 12, color: '#5f6b7a', marginBottom: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontWeight: 600, color: '#0972d3' }}>🔬 Morphological similarity</span>
          <span style={{
            background: '#f0f4ff', border: '1px solid #c5d5f5', borderRadius: 4,
            padding: '1px 7px', fontFamily: 'monospace', fontSize: 11, color: '#1a4a8a',
          }}>
            query=patient {artifact.queryPatient.patientId}
          </span>
          <span>{artifact.matches.length} similar patient{artifact.matches.length !== 1 ? 's' : ''} — showing {page * PAGE_SIZE + 1}–{Math.min((page + 1) * PAGE_SIZE, sorted.length)}</span>
        </span>
        <PaginationBar page={page} totalPages={totalPages} onPrev={() => setPage(p => p - 1)} onNext={() => setPage(p => p + 1)} />
      </div>
      <Table
        items={pageItems}
        columnDefinitions={RETRIEVAL_COLS}
        variant="embedded"
        stripedRows
        resizableColumns
        sortingColumn={{ sortingField: sortCol }}
        sortingDescending={sortDesc}
        onSortingChange={({ detail }) => {
          setSortCol(detail.sortingColumn.sortingField ?? 'rank')
          setSortDesc(detail.isDescending ?? false)
          setPage(0)
        }}
        getRowId={(r: RetrievalRow) => `${r.rank}-${r.patientId}`}
        empty={<Box textAlign="center">No results</Box>}
      />
    </div>
  )
}

// ─── Inline search results table ─────────────────────────────────────────────

const SEARCH_COLS = [
  { id: 'patient_id', header: 'Patient ID', cell: (r: SearchPatientRow) => r.patient_id, width: 90, sortingField: 'patient_id' },
  { id: 'age', header: 'Age', cell: (r: SearchPatientRow) => r.age_at_initial_diagnosis, width: 60, sortingField: 'age_at_initial_diagnosis' },
  { id: 'year', header: 'Year', cell: (r: SearchPatientRow) => r.year_of_initial_diagnosis, width: 65, sortingField: 'year_of_initial_diagnosis' },
  { id: 'site', header: 'Tumor site', cell: (r: SearchPatientRow) => r.primary_tumor_site, sortingField: 'primary_tumor_site' },
  { id: 'histology', header: 'Histology', cell: (r: SearchPatientRow) => r.histologic_type, sortingField: 'histologic_type' },
  { id: 'hpv', header: 'HPV/p16', cell: (r: SearchPatientRow) => r.hpv_association_p16, width: 80, sortingField: 'hpv_association_p16' },
  { id: 'pt', header: 'pT', cell: (r: SearchPatientRow) => r.pt_stage, width: 55, sortingField: 'pt_stage' },
  { id: 'pn', header: 'pN', cell: (r: SearchPatientRow) => r.pn_stage, width: 55, sortingField: 'pn_stage' },
  { id: 'smoking', header: 'Smoking', cell: (r: SearchPatientRow) => r.smoking_status, sortingField: 'smoking_status' },
  { id: 'sex', header: 'Sex', cell: (r: SearchPatientRow) => r.sex, width: 60, sortingField: 'sex' },
  { id: 'survival', header: 'Survival', cell: (r: SearchPatientRow) => r.survival_status, sortingField: 'survival_status' },
  { id: 'recurrence', header: 'Recurrence', cell: (r: SearchPatientRow) => r.recurrence, width: 90, sortingField: 'recurrence' },
]

function InlineSearchTable({ artifact }: { artifact: SearchArtifact }) {
  const [sortCol, setSortCol] = useState<string>('patient_id')
  const [sortDesc, setSortDesc] = useState(false)
  const [page, setPage] = useState(0)

  const sorted = [...artifact.patients].sort((a, b) => {
    const col = SEARCH_COLS.find(c => c.id === sortCol)
    if (!col) return 0
    const av = (a as Record<string, unknown>)[col.sortingField ?? col.id]
    const bv = (b as Record<string, unknown>)[col.sortingField ?? col.id]
    if (av == null) return 1
    if (bv == null) return -1
    const cmp = av < bv ? -1 : av > bv ? 1 : 0
    return sortDesc ? -cmp : cmp
  })

  const totalPages = Math.ceil(sorted.length / PAGE_SIZE)
  const pageItems = sorted.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)

  return (
    <div style={{ marginTop: 8, overflowX: 'auto' }}>
      <div style={{ fontSize: 12, color: '#5f6b7a', marginBottom: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
          <span style={{ fontWeight: 600, color: '#0972d3' }}>🔍 Patient search</span>
          {artifact.query && (
            <span style={{
              background: '#f0f4ff', border: '1px solid #c5d5f5', borderRadius: 4,
              padding: '1px 7px', fontFamily: 'monospace', fontSize: 11, color: '#1a4a8a',
            }}>
              {artifact.query}
            </span>
          )}
          <span>{artifact.total} patient{artifact.total !== 1 ? 's' : ''} matched
            {sorted.length > 0 ? ` — showing ${page * PAGE_SIZE + 1}–${Math.min((page + 1) * PAGE_SIZE, sorted.length)} of ${sorted.length}` : ''}
          </span>
        </span>
        <PaginationBar page={page} totalPages={totalPages} onPrev={() => setPage(p => p - 1)} onNext={() => setPage(p => p + 1)} />
      </div>
      <Table
        items={pageItems}
        columnDefinitions={SEARCH_COLS}
        variant="embedded"
        stripedRows
        resizableColumns
        sortingColumn={{ sortingField: sortCol }}
        sortingDescending={sortDesc}
        onSortingChange={({ detail }) => {
          setSortCol(detail.sortingColumn.sortingField ?? 'patient_id')
          setSortDesc(detail.isDescending ?? false)
          setPage(0)
        }}
        getRowId={(r: SearchPatientRow) => r.patient_id}
        empty={<Box textAlign="center">No patients matched</Box>}
      />
    </div>
  )
}

// ─── Inline Plotly chart ──────────────────────────────────────────────────────

function InlinePlotlyChart({ artifact, onPngCaptured, height = 380, wide = false }: {
  artifact: PlotlyArtifact
  onPngCaptured?: (png: string) => void
  height?: number
  wide?: boolean
}) {
  const divRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!divRef.current) return
    import('plotly.js-dist').then((Plotly) => {
      try {
        const fig = JSON.parse(artifact.plotlyJson) as { data: unknown[]; layout: unknown }
        const origLayout = fig.layout as Record<string, unknown>
        const layout: Record<string, unknown> = {
          ...origLayout,
          margin: { t: 40, r: 10, b: 40, l: 50 },
          height,
        }
        if (wide) {
          // Wide charts (UMAP, clustering) fill their container
          layout.autosize = true
          layout.width = undefined
        } else {
          // Narrow charts keep their natural width from the backend layout
          layout.autosize = false
        }
        void Plotly.react(divRef.current!, fig.data as Plotly.Data[], layout as Plotly.Layout, { responsive: wide })
          .then(() => {
            if (!onPngCaptured || artifact.pngBase64) return
            return Plotly.toImage(divRef.current!, { format: 'png', width: wide ? 900 : 400, height })
          })
          .then((url) => {
            if (url && onPngCaptured) {
              onPngCaptured(url.replace(/^data:image\/png;base64,/, ''))
            }
          })
      } catch { /* ignore */ }
    })
  }, [artifact.plotlyJson]) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div style={{ marginTop: 8 }}>
      <div style={{ fontSize: 12, color: '#5f6b7a', marginBottom: 4 }}>{artifact.title}</div>
      <div ref={divRef} style={{ width: wide ? '100%' : 'fit-content' }} />
    </div>
  )
}

function ArtifactBlock({ artifact, onPngCaptured, chartHeight, wide }: {
  artifact: Artifact
  onPngCaptured?: (png: string) => void
  chartHeight?: number
  wide?: boolean
}) {
  if (artifact.type === 'retrieval') return <InlineRetrievalTable artifact={artifact} />
  if (artifact.type === 'search') return <InlineSearchTable artifact={artifact} />
  if (artifact.type === 'plotly') return <InlinePlotlyChart artifact={artifact} onPngCaptured={onPngCaptured} height={chartHeight} wide={wide} />
  return null
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function toSnakeCaseContext(ctx: AppContext): Record<string, unknown> {
  return {
    n_slides: ctx.nSlides,
    n_patients: ctx.nPatients,
    active_filters: ctx.activeFilters,
    query_patient_id: ctx.queryPatientId,
    retrieval_results: ctx.retrievalResults,
    alpha: ctx.alpha,
    beta: ctx.beta,
    gamma: ctx.gamma,
    umap_n_points: ctx.umapNPoints,
    umap_n_clusters: ctx.umapNClusters,
    best_k: ctx.bestK,
  }
}

const TOOL_LABELS: Record<string, string> = {
  run_retrieval: 'Running patient similarity retrieval…',
  run_umap: 'Running UMAP projection…',
  run_clustering: 'Running cluster analysis…',
  get_dataset_info: 'Fetching dataset info…',
  search_patients: 'Searching patients…',
}

// ─── ChatPanel ────────────────────────────────────────────────────────────────

interface ChatPanelProps {
  appContext: AppContext
  onRetrievalResult?: (result: import('../types').RetrievalResponse) => void
}

export default function ChatPanel({ appContext }: ChatPanelProps): React.ReactElement {
  const [entries, setEntries] = useState<ChatEntry[]>([])
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [activeToolUse, setActiveToolUse] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [reportOpen, setReportOpen] = useState(false)
  const [reportContent, setReportContent] = useState<string | null>(null)
  const [reportLoading, setReportLoading] = useState(false)
  const [reportError, setReportError] = useState<string | null>(null)
  const [reportCharts, setReportCharts] = useState<{ title: string; pngBase64: string }[]>([])
  const bottomRef = useRef<HTMLDivElement>(null)
  const abortRef = useRef<AbortController | null>(null)

  const handleGenerateReport = useCallback(async () => {
    setReportOpen(true)
    setReportContent(null)
    setReportError(null)
    setReportLoading(true)
    try {
      // ── 1. Capture Plotly PNGs on-demand from stored JSON ──────────────
      const Plotly = await import('plotly.js-dist')
      const charts: { title: string; pngBase64: string }[] = []
      for (const entry of entries) {
        for (const artifact of entry.artifacts ?? []) {
          if (artifact.type === 'plotly') {
            try {
              const fig = JSON.parse(artifact.plotlyJson) as { data: unknown[]; layout: unknown }
              const url = await Plotly.toImage(
                { data: fig.data as Plotly.Data[], layout: fig.layout as Plotly.Layout },
                { format: 'png', width: 900, height: 420 }
              )
              charts.push({ title: artifact.title, pngBase64: url.replace(/^data:image\/png;base64,/, '') })
            } catch { /* skip if render fails */ }
          }
        }
      }

      // ── 2. Serialise tables as markdown ───────────────────────────────
      const transcriptLines: string[] = []
      let chartIdx = 0
      for (const entry of entries) {
        const role = entry.role === 'user' ? 'Researcher' : 'Assistant'
        if (entry.content) transcriptLines.push(`**${role}:** ${entry.content}`)
        for (const artifact of entry.artifacts ?? []) {
          if (artifact.type === 'plotly') {
            transcriptLines.push(`[Figure ${++chartIdx}: ${artifact.title}]`)
          } else if (artifact.type === 'retrieval') {
            const rows = [artifact.queryPatient, ...artifact.matches]
            const header = '| Rank | Patient | Composite | Slide sim | Comp sim | Meta sim | Tumor site | Histology | HPV/p16 | pT | pN | Age |'
            const sep   = '|------|---------|-----------|-----------|----------|----------|------------|-----------|---------|----|----|-----|'
            const body = rows.map(r =>
              `| ${r.rank === 0 ? '0 (query)' : r.rank} | ${r.patientId} | ${r.composite.toFixed(3)} | ${r.slideSim.toFixed(3)} | ${r.compSim.toFixed(3)} | ${r.metaSim.toFixed(3)} | ${r.primaryTumorSite ?? '—'} | ${r.histologicType ?? '—'} | ${r.hpvAssociationP16 ?? '—'} | ${r.ptStage ?? '—'} | ${r.pnStage ?? '—'} | ${r.ageAtInitialDiagnosis ?? '—'} |`
            ).join('\n')
            transcriptLines.push(`**Morphological similarity results (query: patient ${artifact.queryPatient.patientId})**\n\n${header}\n${sep}\n${body}`)
          } else if (artifact.type === 'search') {
            const header = '| Patient | Age | Tumor site | Histology | HPV/p16 | pT | pN | Smoking | Survival |'
            const sep   = '|---------|-----|------------|-----------|---------|----|----|---------|----------|'
            const body = artifact.patients.slice(0, 50).map(r =>
              `| ${r.patient_id} | ${r.age_at_initial_diagnosis} | ${r.primary_tumor_site} | ${r.histologic_type} | ${r.hpv_association_p16} | ${r.pt_stage} | ${r.pn_stage} | ${r.smoking_status} | ${r.survival_status} |`
            ).join('\n')
            const note = artifact.patients.length > 50 ? `\n*(showing first 50 of ${artifact.total} matched)*` : ''
            transcriptLines.push(`**Patient search results (${artifact.query ?? 'filtered'}, ${artifact.total} matched)**\n\n${header}\n${sep}\n${body}${note}`)
          }
        }
      }

      const userContent =
        `Based on the following conversation, data tables, and ${charts.length} chart image(s), generate a formal scientific report.\n\n` +
        `---\n${transcriptLines.join('\n\n')}\n---`

      // ── 3. Build Bedrock messages with vision content ─────────────────
      const messages: object[] = charts.length > 0
        ? [{
            role: 'user',
            content: [
              { type: 'text', text: userContent },
              ...charts.map(c => ({
                type: 'image',
                source: { type: 'base64', media_type: 'image/png', data: c.pngBase64 },
              })),
            ],
          }]
        : [{ role: 'user', content: userContent }]

      const res = await fetch('/api/chat/report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: toMessages(entries),
          app_context: appContext,
          chart_messages: messages,
        }),
      })
      if (!res.ok) {
        const data = await res.json() as { detail?: string }
        throw new Error(data.detail ?? 'Report generation failed')
      }
      if (!res.body) throw new Error('No response body')
      const reader = res.body.pipeThrough(new TextDecoderStream()).getReader()
      let buffer = ''
      let accumulated = ''
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += value
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''
        for (const line of lines) {
          if (!line.startsWith('data:')) continue
          const jsonStr = line.slice(5).trim()
          if (!jsonStr) continue
          try {
            const parsed = JSON.parse(jsonStr) as { delta?: string; done?: boolean; error?: string }
            if (parsed.error) throw new Error(parsed.error)
            if (parsed.delta) { accumulated += parsed.delta; setReportContent(accumulated) }
            if (parsed.done) setReportLoading(false)
          } catch (e) { if (e instanceof SyntaxError) continue; throw e }
        }
      }
      setReportCharts(charts)
    } catch (err) {
      setReportError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setReportLoading(false)
    }
  }, [entries, appContext])

  const printRef = useRef<HTMLDivElement>(null)

  const handlePrintPDF = useCallback(async () => {
    if (!reportContent) return

    // Capture charts on-demand if not already done
    let charts = reportCharts
    if (charts.length === 0) {
      const Plotly = await import('plotly.js-dist')
      const captured: { title: string; pngBase64: string }[] = []
      for (const entry of entries) {
        for (const artifact of entry.artifacts ?? []) {
          if (artifact.type === 'plotly') {
            try {
              // Create a temporary off-screen div for rendering
              const tmpDiv = document.createElement('div')
              tmpDiv.style.cssText = 'position:fixed;left:-9999px;top:0;width:900px;height:420px;'
              document.body.appendChild(tmpDiv)
              const fig = JSON.parse(artifact.plotlyJson) as { data: unknown[]; layout: unknown }
              await Plotly.newPlot(tmpDiv, fig.data as Plotly.Data[], {
                ...(fig.layout as object), width: 900, height: 420,
              })
              const url = await Plotly.toImage(tmpDiv, { format: 'png', width: 900, height: 420 })
              document.body.removeChild(tmpDiv)
              captured.push({ title: artifact.title, pngBase64: url.replace(/^data:image\/png;base64,/, '') })
            } catch { /* skip */ }
          }
        }
      }
      charts = captured
      setReportCharts(captured)
    }

    // Build table HTML from artifacts
    const buildTableHtml = (): string => {
      let html = ''
      for (const entry of entries) {
        for (const artifact of entry.artifacts ?? []) {
          if (artifact.type === 'retrieval') {
            const rows = [artifact.queryPatient, ...artifact.matches]
            html += `<h3>Morphological Similarity Results — Query: Patient ${artifact.queryPatient.patientId}</h3>`
            html += '<table><thead><tr><th>Rank</th><th>Patient</th><th>Composite</th><th>Slide sim</th><th>Comp sim</th><th>Meta sim</th><th>Tumor site</th><th>Histology</th><th>HPV/p16</th><th>pT</th><th>pN</th><th>Age</th></tr></thead><tbody>'
            rows.forEach(r => {
              html += `<tr><td>${r.rank === 0 ? '0 (query)' : r.rank}</td><td>${r.patientId}</td><td>${r.composite.toFixed(3)}</td><td>${r.slideSim.toFixed(3)}</td><td>${r.compSim.toFixed(3)}</td><td>${r.metaSim.toFixed(3)}</td><td>${r.primaryTumorSite ?? '—'}</td><td>${r.histologicType ?? '—'}</td><td>${r.hpvAssociationP16 ?? '—'}</td><td>${r.ptStage ?? '—'}</td><td>${r.pnStage ?? '—'}</td><td>${r.ageAtInitialDiagnosis ?? '—'}</td></tr>`
            })
            html += '</tbody></table>'
          } else if (artifact.type === 'search' && artifact.patients.length > 0) {
            html += `<h3>Patient Search Results — ${artifact.query ?? ''} (${artifact.total} matched)</h3>`
            html += '<table><thead><tr><th>Patient</th><th>Age</th><th>Tumor site</th><th>Histology</th><th>HPV/p16</th><th>pT</th><th>pN</th><th>Smoking</th><th>Survival</th></tr></thead><tbody>'
            artifact.patients.slice(0, 100).forEach(r => {
              html += `<tr><td>${r.patient_id}</td><td>${r.age_at_initial_diagnosis}</td><td>${r.primary_tumor_site}</td><td>${r.histologic_type}</td><td>${r.hpv_association_p16}</td><td>${r.pt_stage}</td><td>${r.pn_stage}</td><td>${r.smoking_status}</td><td>${r.survival_status}</td></tr>`
            })
            if (artifact.patients.length > 100) html += `<tr><td colspan="9" style="text-align:center;font-style:italic">Showing first 100 of ${artifact.total} patients</td></tr>`
            html += '</tbody></table>'
          }
        }
      }
      return html
    }

    // Convert markdown to basic HTML (headings, bold, lists, tables, paragraphs)
    const mdToHtml = (md: string): string => {
      const lines = md.split('\n')
      const out: string[] = []
      let i = 0
      while (i < lines.length) {
        const line = lines[i]
        // Markdown table: line starts with |
        if (line.trimStart().startsWith('|')) {
          const tableLines: string[] = []
          while (i < lines.length && lines[i].trimStart().startsWith('|')) {
            tableLines.push(lines[i])
            i++
          }
          const isSep = (l: string) => /^[\s|:\-]+$/.test(l)
          const parseRow = (l: string) => l.split('|').slice(1, -1).map(c => c.trim())
          const headers = parseRow(tableLines[0])
          const dataRows = tableLines.slice(1).filter(l => !isSep(l)).map(parseRow)
          const thead = '<thead><tr>' + headers.map(h => `<th>${h}</th>`).join('') + '</tr></thead>'
          const tbody = '<tbody>' + dataRows.map(row =>
            '<tr>' + row.map(c => `<td>${c}</td>`).join('') + '</tr>'
          ).join('') + '</tbody>'
          out.push(`<table>${thead}${tbody}</table>`)
          continue
        }
        // Headings
        const hm = line.match(/^(#{1,3})\s+(.+)/)
        if (hm) { out.push(`<h${hm[1].length}>${hm[2]}</h${hm[1].length}>`); i++; continue }
        // HR
        if (/^---+$/.test(line.trim())) { out.push('<hr>'); i++; continue }
        // List items — collect run
        if (/^[-*]\s/.test(line) || /^\d+\.\s/.test(line)) {
          const items: string[] = []
          const ordered = /^\d+\./.test(line)
          while (i < lines.length && (/^[-*]\s/.test(lines[i]) || /^\d+\.\s/.test(lines[i]))) {
            items.push(lines[i].replace(/^[-*]\s+/, '').replace(/^\d+\.\s+/, ''))
            i++
          }
          const tag = ordered ? 'ol' : 'ul'
          out.push(`<${tag}>${items.map(it => `<li>${inlineMd(it)}</li>`).join('')}</${tag}>`)
          continue
        }
        // Blank line
        if (line.trim() === '') { out.push(''); i++; continue }
        // Paragraph
        out.push(`<p>${inlineMd(line)}</p>`)
        i++
      }
      return out.join('\n')
    }

    const inlineMd = (s: string) =>
      s.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
       .replace(/\*(.+?)\*/g, '<em>$1</em>')
       .replace(/`(.+?)`/g, '<code>$1</code>')

    // Build a lookup: figure number → base64 PNG
    const chartByIndex: Record<number, string> = {}
    charts.forEach((c, idx) => { chartByIndex[idx + 1] = c.pngBase64 })

    // Replace [Figure N: title] markers inline in the rendered HTML
    const injectFigures = (html: string): string =>
      html.replace(/\[Figure (\d+): ([^\]]+)\]/g, (_, num, title) => {
        const png = chartByIndex[parseInt(num)]
        if (!png) return `<em>[Figure ${num}: ${title}]</em>`
        return `<div class="figure"><img src="data:image/png;base64,${png}" alt="${title}"><p class="caption">Figure ${num}: ${title}</p></div>`
      })

    const bodyHtml = injectFigures(mdToHtml(reportContent))

    const fullHtml = `<!DOCTYPE html><html><head><meta charset="utf-8">
<title>HNSCC Patient Similarity Analysis Report</title>
<style>
  body { font-family: Georgia, serif; font-size: 11pt; line-height: 1.7; color: #000; max-width: 170mm; margin: 0 auto; padding: 20mm 0; }
  h1 { font-size: 18pt; border-bottom: 2px solid #333; padding-bottom: 4pt; margin-bottom: 8pt; }
  h2 { font-size: 14pt; border-bottom: 1px solid #aaa; padding-bottom: 2pt; margin-top: 16pt; }
  h3 { font-size: 12pt; margin-top: 12pt; }
  p { margin: 0 0 6pt; }
  ul, ol { margin: 4pt 0 6pt 18pt; }
  li { margin-bottom: 2pt; }
  table { width: 100%; border-collapse: collapse; font-size: 8.5pt; margin: 10pt 0; page-break-inside: avoid; }
  th { background: #f0f0f0; border: 1px solid #999; padding: 3pt 5pt; text-align: left; font-weight: bold; }
  td { border: 1px solid #ccc; padding: 2pt 5pt; }
  tr:nth-child(even) td { background: #fafafa; }
  .figure { page-break-inside: avoid; margin: 14pt 0; text-align: center; }
  .figure img { max-width: 100%; height: auto; border: 1px solid #e0e0e0; }
  .caption { font-size: 9pt; color: #555; font-style: italic; margin-top: 4pt; }
  .report-header { text-align: center; margin-bottom: 20pt; padding-bottom: 10pt; border-bottom: 2px solid #333; }
  .report-footer { margin-top: 24pt; padding-top: 6pt; border-top: 1px solid #ccc; font-size: 8pt; color: #777; text-align: center; }
  hr { border: none; border-top: 1px solid #ccc; margin: 12pt 0; }
  strong { font-weight: bold; }
  em { font-style: italic; }
  @page { margin: 20mm 15mm; size: A4; }
</style>
</head><body>
<div class="report-header">
  <div style="font-size:9pt;color:#777">HANCOCK HNSCC Patient Similarity Analysis</div>
  <div style="font-size:8pt;color:#999">${new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}</div>
</div>
${bodyHtml}
${buildTableHtml()}
<div class="report-footer">Generated by HANCOCK Patient Similarity Analysis System &nbsp;·&nbsp; Embeddings: H-optimus-0 (Bioptimus) &nbsp;·&nbsp; Aggregation: GatedAttentionABMIL</div>
<script>window.onload = function(){ window.print(); }</script>
</body></html>`

    const win = window.open('', '_blank')
    if (win) {
      win.document.write(fullHtml)
      win.document.close()
    }
  }, [reportContent, reportCharts, entries])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [entries, streaming])

  const toMessages = (es: ChatEntry[]): ChatMessage[] =>
    es.map((e) => ({
      role: e.role,
      content: e.content.trim() || (e.role === 'assistant' ? '[results shown above]' : '…'),
    }))

  const handleSend = useCallback(async () => {
    const text = input.trim()
    if (!text || streaming) return

    const userEntry: ChatEntry = { role: 'user', content: text }
    const nextEntries = [...entries, userEntry]
    setEntries(nextEntries)
    setInput('')
    setError(null)
    setStreaming(true)
    setActiveToolUse(null)

    const controller = new AbortController()
    abortRef.current = controller

    try {
      const res = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: toMessages(nextEntries),
          app_context: toSnakeCaseContext(appContext),
        }),
        signal: controller.signal,
      })

      if (!res.ok) {
        const detail = await res.json().catch(() => ({ detail: res.statusText }))
        throw new Error((detail as { detail?: string })?.detail ?? res.statusText)
      }
      if (!res.body) throw new Error('No response body')

      setEntries((prev) => [...prev, { role: 'assistant', content: '' }])

      const reader = res.body.pipeThrough(new TextDecoderStream()).getReader()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += value
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''

        for (const line of lines) {
          if (!line.startsWith('data:')) continue
          const jsonStr = line.slice('data:'.length).trim()
          if (!jsonStr) continue
          try {
            const parsed = JSON.parse(jsonStr) as {
              delta?: string; done?: boolean; error?: string
              tool_use?: string; artifact?: Artifact
            }
            if (parsed.error) { setError(parsed.error); setStreaming(false); setActiveToolUse(null); return }
            if (parsed.done) { setStreaming(false); setActiveToolUse(null); return }
            if (parsed.tool_use) {
              setActiveToolUse(parsed.tool_use)
              // Mark usedTools for any tool call; usedEmbeddings only for e_-column tools
              const EMBEDDING_TOOLS = new Set([
                'run_retrieval', 'run_umap', 'run_clustering',
                'recompute_slide_embeddings', 'validate_composition_profiles',
              ])
              setEntries((prev) => {
                const updated = [...prev]
                const last = updated[updated.length - 1]
                if (last?.role === 'assistant') {
                  updated[updated.length - 1] = {
                    ...last,
                    usedTools: true,
                    usedEmbeddings: last.usedEmbeddings || EMBEDDING_TOOLS.has(parsed.tool_use!),
                  }
                }
                return updated
              })
            }
            if (parsed.artifact) {
              setEntries((prev) => {
                const updated = [...prev]
                const last = updated[updated.length - 1]
                if (last?.role === 'assistant') {
                  const existing = last.artifacts ?? []
                  updated[updated.length - 1] = { ...last, artifacts: [...existing, parsed.artifact!] }
                }
                return updated
              })
            }
            if (parsed.delta !== undefined) {
              setActiveToolUse(null)
              setEntries((prev) => {
                const updated = [...prev]
                const last = updated[updated.length - 1]
                if (last?.role === 'assistant')
                  updated[updated.length - 1] = { ...last, content: last.content + parsed.delta }
                return updated
              })
            }
          } catch { /* ignore malformed SSE */ }
        }
      }
    } catch (err) {
      if ((err as Error).name === 'AbortError') return
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setStreaming(false)
      setActiveToolUse(null)
    }
  }, [input, entries, streaming, appContext])

  const handleClear = useCallback(() => {
    abortRef.current?.abort()
    setEntries([])
    setInput('')
    setError(null)
    setStreaming(false)
    setActiveToolUse(null)
  }, [])

  const handleKeyDown = useCallback(
    (e: CustomEvent<{ key: string }>) => { if (e.detail.key === 'Enter') void handleSend() },
    [handleSend],
  )

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: 'calc(100vh - 120px)',
      minHeight: 400,
    }}>
      {/* ── Header ──────────────────────────────────────────────────── */}
      <div style={{ padding: '12px 20px 8px', borderBottom: '1px solid #e9ebed', flexShrink: 0 }}>
        <Header
          variant="h2"
          actions={
            <SpaceBetween direction="horizontal" size="xs">
              <Button
                variant="normal"
                iconName="file"
                onClick={() => void handleGenerateReport()}
                disabled={entries.length === 0 || streaming}
              >
                Generate report
              </Button>
              <Button variant="normal" onClick={handleClear} disabled={entries.length === 0 && !streaming}>
                Clear chat
              </Button>
            </SpaceBetween>
          }
        >
          Chat
        </Header>
      </div>

      {/* ── Report modal ─────────────────────────────────────────────── */}
      <Modal
        visible={reportOpen}
        onDismiss={() => setReportOpen(false)}
        size="large"
        header="Scientific Analysis Report"
      >
        {reportLoading && !reportContent && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '24px 0' }}>
            <Spinner size="normal" />
            <span style={{ color: '#5f6b7a', fontSize: 14 }}>Generating scientific report…</span>
          </div>
        )}
        {reportError && <Alert type="error">{reportError}</Alert>}
        {reportContent && (
          <>
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginBottom: 12, alignItems: 'center' }}>
              {reportLoading && (
                <span style={{ fontSize: 12, color: '#5f6b7a', display: 'flex', alignItems: 'center', gap: 6, marginRight: 'auto' }}>
                  <Spinner size="normal" /> Writing report…
                </span>
              )}
              {!reportLoading && (
                <Button variant="primary" iconName="file" onClick={handlePrintPDF}>
                  Save as PDF
                </Button>
              )}
              <Button variant="normal" onClick={() => setReportOpen(false)}>Close</Button>
            </div>
            <div style={{
              maxHeight: '65vh', overflowY: 'auto', padding: '4px 8px',
              fontFamily: 'Georgia, serif', fontSize: 14, lineHeight: 1.7,
              borderTop: '1px solid #e9ebed', paddingTop: 12,
            }}>
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{reportContent}</ReactMarkdown>
            </div>
          </>
        )}
      </Modal>

      {/* ── Message history ─────────────────────────────────────────── */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        padding: '16px 20px',
        display: 'flex',
        flexDirection: 'column',
        gap: 12,
      }}>
        {entries.length === 0 && !streaming && (
          <div style={{
            flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
            color: '#5f6b7a', fontSize: 14,
          }}>
            Ask a question about the loaded dataset or a specific patient.
          </div>
        )}

        {entries.map((entry, i) => (
          <div key={i} style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: entry.role === 'user' ? 'flex-end' : 'flex-start',
            gap: 6,
            width: (entry.role === 'assistant' && entry.artifacts?.some(a => a.type === 'plotly')) ? '100%' : undefined,
          }}>
            {/* Role label */}
            <div style={{ fontSize: 11, color: '#5f6b7a', fontWeight: 600, letterSpacing: '0.04em', display: 'flex', alignItems: 'center', gap: 6 }}>
              {entry.role === 'user' ? 'You' : 'Assistant'}
              {entry.role === 'assistant' && entry.usedEmbeddings && (
                <span
                  title="Histopathology slide embeddings (ABMIL) were used to compute this result"
                  style={{ display: 'inline-flex', alignItems: 'center', gap: 3, cursor: 'default', userSelect: 'none' }}
                >
                  {/* Microscope */}
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#7c3aed" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <rect x="9" y="1" width="6" height="3" rx="1" />
                    <line x1="12" y1="4" x2="12" y2="8" />
                    <rect x="9.5" y="8" width="5" height="4" rx="0.8" />
                    <line x1="12" y1="12" x2="12" y2="14" />
                    <rect x="7" y="14" width="10" height="2" rx="0.5" />
                    <path d="M9 4 Q6 4 6 8 L6 14" />
                    <path d="M5 21 L19 21" />
                    <path d="M6 21 L6 19 Q6 18 8 18 L16 18 Q18 18 18 19 L18 21" />
                    <circle cx="5.5" cy="11" r="1.2" />
                  </svg>
                  {/* Histology slide */}
                  <svg width="18" height="10" viewBox="0 0 36 16" fill="none" aria-hidden="true">
                    <rect x="1" y="4" width="34" height="8" rx="1.5" fill="#f3e8ff" stroke="#7c3aed" strokeWidth="1.4" />
                    <rect x="8" y="5.5" width="20" height="5" rx="1" fill="#ede9fe" stroke="#7c3aed" strokeWidth="1" />
                    <circle cx="13" cy="8" r="1.2" fill="#be185d" />
                    <circle cx="17" cy="7.2" r="0.9" fill="#7c3aed" />
                    <circle cx="20" cy="8.5" r="1.1" fill="#be185d" />
                    <circle cx="24" cy="7.5" r="0.8" fill="#7c3aed" />
                    <circle cx="16" cy="9.2" r="0.7" fill="#be185d" />
                    <circle cx="22" cy="9" r="0.9" fill="#7c3aed" />
                  </svg>
                </span>
              )}
              {entry.role === 'assistant' && entry.usedTools && !entry.usedEmbeddings && (
                <span
                  title="Data tools were used to compute this result"
                  style={{ display: 'inline-flex', alignItems: 'center', cursor: 'default', userSelect: 'none' }}
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#5f6b7a" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <circle cx="11" cy="11" r="7" />
                    <line x1="16.5" y1="16.5" x2="22" y2="22" />
                  </svg>
                </span>
              )}
            </div>

            {/* Text bubble */}
            {entry.content && (
              <div style={{
                maxWidth: entry.artifacts?.length ? '100%' : '78%',
                padding: '10px 14px',
                borderRadius: entry.role === 'user' ? '12px 12px 2px 12px' : '12px 12px 12px 2px',
                background: entry.role === 'user' ? '#0972d3' : '#ffffff',
                color: entry.role === 'user' ? '#ffffff' : '#000716',
                border: entry.role === 'assistant' ? '1px solid #e9ebed' : 'none',
                boxShadow: '0 1px 3px rgba(0,0,0,0.06)',
                wordBreak: 'break-word',
              }}>
                {entry.role === 'user'
                  ? <div style={{ fontSize: 14, lineHeight: 1.5, whiteSpace: 'pre-wrap' }}>{entry.content}</div>
                  : <MarkdownText text={entry.content} />
                }
              </div>
            )}

            {/* Artifacts — plotly charts tile in a 4-unit grid.
                UMAP + clustering: if both present → paired on one row, each 2fr, 2× height.
                Otherwise wide charts (umap/cluster/silhouette) span 2 units at 2× height;
                narrow bar/cross-tab charts span 1 unit at normal height. */}
            {(() => {
              const artifacts = entry.artifacts ?? []
              const plotlyArtifacts = artifacts.filter(a => a.type === 'plotly') as PlotlyArtifact[]

              const isUmap       = (a: PlotlyArtifact) => /umap/i.test(a.title)
              const isClustering = (a: PlotlyArtifact) => /cluster|silhouette/i.test(a.title)
              const isWide       = (a: PlotlyArtifact) => isUmap(a) || isClustering(a)

              const hasUmap       = plotlyArtifacts.some(isUmap)
              const hasClustering = plotlyArtifacts.some(isClustering)
              const pairWide      = hasUmap && hasClustering  // put them together on one row

              const NORMAL_H = 300
              const WIDE_H   = NORMAL_H * 2   // 600px

              const GRID_COLS = 4
              type GridItem = { artifact: PlotlyArtifact; ai: number; span: number; height: number }
              const rows: GridItem[][] = []
              let currentRow: GridItem[] = []
              let usedCols = 0

              // When pairing, collect wide charts separately and emit them as one row at the end
              const wideQueue: GridItem[] = []

              plotlyArtifacts.forEach((artifact) => {
                const ai = artifacts.indexOf(artifact)
                const wide = isWide(artifact)

                if (pairWide && wide) {
                  // defer to paired row
                  wideQueue.push({ artifact, ai, span: 2, height: WIDE_H })
                  return
                }

                const span   = wide ? 2 : 1
                const height = wide ? WIDE_H : NORMAL_H

                if (usedCols + span > GRID_COLS && currentRow.length > 0) {
                  rows.push(currentRow)
                  currentRow = []
                  usedCols = 0
                }
                currentRow.push({ artifact, ai, span, height })
                usedCols += span
              })
              if (currentRow.length > 0) rows.push(currentRow)

              // Emit paired wide row (UMAP + clustering side-by-side, each 2fr)
              if (wideQueue.length > 0) rows.push(wideQueue)

              const result: React.ReactNode[] = []

              rows.forEach((row, rowIdx) => {
                const templateCols = row.map(item => `${item.span}fr`).join(' ')
                // Row height = max of individual chart heights
                const rowH = Math.max(...row.map(item => item.height))

                result.push(
                  <div key={`row-${rowIdx}`} style={{ width: '100%' }}>
                    <div style={{ display: 'grid', gridTemplateColumns: templateCols, gap: 8 }}>
                      {row.map(({ artifact, ai, span }, ci) => (
                        <div key={ci} style={{
                          background: '#ffffff', border: '1px solid #e9ebed',
                          borderRadius: 8, padding: '8px',
                          boxShadow: '0 1px 3px rgba(0,0,0,0.06)', minWidth: 0,
                        }}>
                          <ArtifactBlock
                            artifact={artifact}
                            chartHeight={rowH}
                            wide={span > 1}
                            onPngCaptured={(png) => {
                              setEntries(prev => prev.map((e, ei) => {
                                if (ei !== i) return e
                                return { ...e, artifacts: e.artifacts?.map((a, ai2) =>
                                  ai2 === ai && a.type === 'plotly' ? { ...a, pngBase64: png } : a
                                )}
                              }))
                            }}
                          />
                        </div>
                      ))}
                    </div>
                  </div>
                )
              })

              // Non-plotly artifacts (tables etc.) rendered full-width
              artifacts.forEach((artifact, ai) => {
                if (artifact.type !== 'plotly') {
                  result.push(
                    <div key={`npa-${ai}`} style={{
                      width: '100%', background: '#ffffff', border: '1px solid #e9ebed',
                      borderRadius: 8, padding: '8px 12px', boxShadow: '0 1px 3px rgba(0,0,0,0.06)',
                    }}>
                      <ArtifactBlock artifact={artifact} onPngCaptured={undefined} />
                    </div>
                  )
                }
              })

              return result
            })()}
          </div>
        ))}

        {/* Typing / tool-use indicator */}
        {streaming && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <Spinner size="normal" />
            <span style={{ fontSize: 13, color: '#5f6b7a' }}>
              {activeToolUse ? (TOOL_LABELS[activeToolUse] ?? `Using ${activeToolUse}…`) : 'Thinking…'}
            </span>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* ── Error ───────────────────────────────────────────────────── */}
      {error && (
        <div style={{ padding: '0 20px', flexShrink: 0 }}>
          <Alert type="error" dismissible onDismiss={() => setError(null)} header="Chat error">
            {error}
          </Alert>
        </div>
      )}

      {/* ── Input row ────────────────────────────────────────────────── */}
      <div style={{
        padding: '12px 20px',
        borderTop: '1px solid #e9ebed',
        display: 'flex',
        gap: 8,
        alignItems: 'flex-end',
        flexShrink: 0,
        background: '#fafafa',
      }}>
        <div style={{ flex: 1 }}>
          <Input
            value={input}
            onChange={({ detail }) => setInput(detail.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type a message…"
            disabled={streaming}
          />
        </div>
        <Button
          variant="primary"
          onClick={() => void handleSend()}
          disabled={!input.trim() || streaming}
          loading={streaming}
        >
          Send
        </Button>
      </div>
    </div>
  )
}
