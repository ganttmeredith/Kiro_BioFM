import type {
  AppStatus,
  LoadRequest,
  DataSummary,
  UMAPRequest,
  UMAPResponse,
  ClusterRequest,
  ClusterResponse,
  RetrievalRequest,
  RetrievalResponse,
  ChatRequest,
  ClassifyRequest,
  ClassifyResponse,
  BiomarkerRequest,
  BiomarkerResponse,
  OutcomeUMAPRequest,
  OutcomeUMAPResponse,
  BoxPlotRequest,
  BoxPlotResponse,
  InterpretRequest,
  InterpretResponse,
} from '../types'

// ─── Base helpers ─────────────────────────────────────────────────────────────

async function post<TBody, TResponse>(path: string, body: TBody): Promise<TResponse> {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }))
    throw new ApiError(res.status, detail?.detail ?? res.statusText)
  }
  return res.json() as Promise<TResponse>
}

async function get<TResponse>(path: string): Promise<TResponse> {
  const res = await fetch(path)
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }))
    throw new ApiError(res.status, detail?.detail ?? res.statusText)
  }
  return res.json() as Promise<TResponse>
}

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

// ─── Endpoints ───────────────────────────────────────────────────────────────

/** GET /api/status */
export function fetchStatus(): Promise<AppStatus> {
  return get<AppStatus>('/api/status')
}

/** GET /api/data/summary — returns cached DataSummary if already loaded */
export function fetchSummary(): Promise<DataSummary> {
  return get<DataSummary>('/api/data/summary')
}

/** POST /api/data/load */
export function loadData(req: LoadRequest): Promise<DataSummary> {
  return post<LoadRequest, DataSummary>('/api/data/load', req)
}

/** POST /api/umap/run */
export function runUMAP(req: UMAPRequest): Promise<UMAPResponse> {
  return post<object, UMAPResponse>('/api/umap/run', {
    n_neighbors: req.nNeighbors,
    min_dist: req.minDist,
    color_by: req.colorBy,
    filters: req.filters,
    query_slide: req.querySlideName ?? null,
    neighbor_slides: req.neighbourSlideNames ?? [],
  })
}

/** POST /api/cluster/run */
export function runClustering(req: ClusterRequest): Promise<ClusterResponse> {
  return post<ClusterRequest, ClusterResponse>('/api/cluster/run', req)
}

/** POST /api/retrieval/query */
export function queryRetrieval(req: RetrievalRequest): Promise<RetrievalResponse> {
  return post<RetrievalRequest, RetrievalResponse>('/api/retrieval/query', req)
}

/** GET /api/patients */
export function fetchPatients(): Promise<string[]> {
  return get<string[]>('/api/patients')
}

// ─── SSE helpers ─────────────────────────────────────────────────────────────

/**
 * POST /api/profiles/build — streams Server-Sent Events.
 * Returns a ReadableStream of raw SSE lines; caller parses `data:` fields.
 */
export async function buildProfiles(): Promise<ReadableStreamDefaultReader<string>> {
  const res = await fetch('/api/profiles/build', { method: 'POST' })
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }))
    throw new ApiError(res.status, detail?.detail ?? res.statusText)
  }
  if (!res.body) throw new Error('No response body for SSE stream')
  return res.body.pipeThrough(new TextDecoderStream()).getReader()
}

/**
 * POST /api/chat/stream — streams Server-Sent Events.
 * Yields parsed delta strings; throws ApiError on non-2xx.
 *
 * Usage:
 *   for await (const token of streamChat(req)) { ... }
 */
export async function* streamChat(req: ChatRequest): AsyncGenerator<string> {
  const res = await fetch('/api/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }))
    throw new ApiError(res.status, detail?.detail ?? res.statusText)
  }
  if (!res.body) throw new Error('No response body for chat stream')

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
      const json = line.slice('data:'.length).trim()
      if (!json) continue
      const parsed = JSON.parse(json) as { delta?: string; done?: boolean; error?: string }
      if (parsed.error) throw new Error(parsed.error)
      if (parsed.done) return
      if (parsed.delta !== undefined) yield parsed.delta
    }
  }
}

// ─── Outcome endpoints ───────────────────────────────────────────────────────

/** POST /api/outcome/classify */
export function classifyCohorts(req: ClassifyRequest): Promise<ClassifyResponse> {
  return post<ClassifyRequest, ClassifyResponse>('/api/outcome/classify', req)
}

/** POST /api/outcome/biomarkers */
export function analyzeBiomarkers(req: BiomarkerRequest): Promise<BiomarkerResponse> {
  return post<BiomarkerRequest, BiomarkerResponse>('/api/outcome/biomarkers', req)
}

/** POST /api/outcome/umap */
export function runOutcomeUMAP(req: OutcomeUMAPRequest): Promise<OutcomeUMAPResponse> {
  return post<OutcomeUMAPRequest, OutcomeUMAPResponse>('/api/outcome/umap', req)
}

/** POST /api/outcome/export — returns CSV as Blob */
export async function exportBiomarkerCSV(req: BiomarkerRequest): Promise<Blob> {
  const res = await fetch('/api/outcome/export', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }))
    throw new ApiError(res.status, detail?.detail ?? res.statusText)
  }
  return res.blob()
}

/** POST /api/outcome/boxplot */
export function generateBoxPlot(req: BoxPlotRequest): Promise<BoxPlotResponse> {
  return post<BoxPlotRequest, BoxPlotResponse>('/api/outcome/boxplot', req)
}

/** POST /api/outcome/interpret */
export function interpretResults(req: InterpretRequest): Promise<InterpretResponse> {
  return post<InterpretRequest, InterpretResponse>('/api/outcome/interpret', req)
}
