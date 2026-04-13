// ─── Shared ──────────────────────────────────────────────────────────────────

export type MetadataFilters = Record<string, string[]>

export interface ColumnMeta {
  name: string
  label: string
  values: string[]
}

export interface NullDistStats {
  mean: number
  std: number
  percentile5: number
  percentile95: number
}

export interface SilhouetteRow {
  column: string
  score: number
}

// ─── GET /api/status ─────────────────────────────────────────────────────────

export interface AppStatus {
  loaded: boolean
  nSlides?: number
  nPatients?: number
  csvPath?: string
}

// ─── POST /api/data/load ─────────────────────────────────────────────────────

export interface LoadRequest {
  csvPath: string
}

export interface DataSummary {
  nSlides: number
  nPatients: number
  filterableColumns: ColumnMeta[]
  nullDistribution: NullDistStats
  silhouetteScores: SilhouetteRow[]
}

// ─── POST /api/umap/run ──────────────────────────────────────────────────────

export interface UMAPRequest {
  nNeighbors: number
  minDist: number
  colorBy: string
  filters: MetadataFilters
  querySlideName?: string
  neighbourSlideNames?: string[]
}

export interface UMAPResponse {
  plotlyJson: string
  nPoints: number
  nClusters: number
}

// ─── POST /api/cluster/run ───────────────────────────────────────────────────

export interface ClusterRequest {
  kMin: number
  kMax: number
  crossTabulateCols: string[]
  filters: MetadataFilters
}

export interface ClusterResponse {
  plotlyJson: string
  bestK: number
  bestSilhouette: number
}

// ─── POST /api/retrieval/query ───────────────────────────────────────────────

export interface RetrievalRequest {
  patientId: string
  k: number
  alpha: number
  beta: number
  gamma: number
  filters: MetadataFilters
  deduplicateByPatient: boolean
  slideAggregation: 'max' | 'mean'
}

export interface PatientMatch {
  rank: number
  patientId: string
  composite: number
  slideSim: number
  compSim: number
  metaSim: number
  primaryTumorSite: string
  histologicType: string
  hpvAssociationP16: string
  ptStage: string
  pnStage: string
  smokingStatus: string
  ageAtInitialDiagnosis?: number
  yearOfInitialDiagnosis?: number
}

export interface RetrievalResponse {
  matches: PatientMatch[]
  queryPatient: PatientMatch
  querySlideName?: string
  neighborSlideNames?: string[]
}

// ─── Chat ────────────────────────────────────────────────────────────────────

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface AppContext {
  nSlides?: number
  nPatients?: number
  activeFilters: MetadataFilters
  queryPatientId?: string
  retrievalResults?: PatientMatch[]
  alpha: number
  beta: number
  gamma: number
  umapNPoints?: number
  umapNClusters?: number
  bestK?: number
}

export interface ChatRequest {
  messages: ChatMessage[]
  appContext: AppContext
}
