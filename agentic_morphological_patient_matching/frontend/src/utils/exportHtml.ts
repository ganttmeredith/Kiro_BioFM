/**
 * Utilities for exporting panel content as self-contained HTML reports.
 */

import type { PatientMatch, RetrievalResponse } from '../types'

/** Trigger a browser download of an HTML string. */
function downloadHtml(filename: string, html: string): void {
  const blob = new Blob([html], { type: 'text/html;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

/** Wrap content in a minimal self-contained HTML page. */
function wrapPage(title: string, body: string): string {
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>${title}</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 2rem; color: #1a1a1a; }
    h1 { font-size: 1.5rem; margin-bottom: 0.25rem; }
    .meta { color: #666; font-size: 0.85rem; margin-bottom: 1.5rem; }
    table { border-collapse: collapse; width: 100%; font-size: 0.875rem; }
    th { background: #f3f3f3; text-align: left; padding: 8px 12px; border-bottom: 2px solid #ddd; }
    td { padding: 7px 12px; border-bottom: 1px solid #eee; }
    tr:nth-child(even) td { background: #fafafa; }
    .plot-container { margin-top: 1.5rem; }
  </style>
</head>
<body>
${body}
</body>
</html>`
}

// ─── UMAP export ──────────────────────────────────────────────────────────────

/**
 * Export the current UMAP Plotly figure as a self-contained HTML file.
 * Embeds the full plotly.js bundle so the file is viewable offline.
 */
export async function exportUmapHtml(plotlyJson: string, nPoints: number, nClusters: number): Promise<void> {
  const figure = JSON.parse(plotlyJson) as { data: unknown; layout: unknown }
  const plotlyScript = await fetchPlotlyBundle()

  const body = `
  <h1>UMAP Projection Report</h1>
  <p class="meta">Generated: ${new Date().toLocaleString()} &nbsp;|&nbsp; ${nPoints} points &nbsp;|&nbsp; ${nClusters} clusters</p>
  <div id="plot" class="plot-container"></div>
  <script>${plotlyScript}</script>
  <script>
    Plotly.newPlot('plot', ${JSON.stringify(figure.data)}, ${JSON.stringify(figure.layout)}, { responsive: true });
  </script>`

  downloadHtml(`umap_report_${timestamp()}.html`, wrapPage('UMAP Report', body))
}

// ─── Cluster export ───────────────────────────────────────────────────────────

export async function exportClusterHtml(
  plotlyJson: string,
  bestK: number,
  bestSilhouette: number,
): Promise<void> {
  const figure = JSON.parse(plotlyJson) as { data: unknown; layout: unknown }
  const plotlyScript = await fetchPlotlyBundle()

  const body = `
  <h1>Morphological Grouping Analysis Report</h1>
  <p class="meta">Generated: ${new Date().toLocaleString()} &nbsp;|&nbsp; Optimal k: ${bestK} &nbsp;|&nbsp; Silhouette: ${bestSilhouette.toFixed(4)}</p>
  <div id="plot" class="plot-container"></div>
  <script>${plotlyScript}</script>
  <script>
    Plotly.newPlot('plot', ${JSON.stringify(figure.data)}, ${JSON.stringify(figure.layout)}, { responsive: true });
  </script>`

  downloadHtml(`cluster_report_${timestamp()}.html`, wrapPage('Cluster Report', body))
}

// ─── Retrieval export ─────────────────────────────────────────────────────────

const RETRIEVAL_COLS: { key: keyof PatientMatch; label: string }[] = [
  { key: 'rank', label: 'Rank' },
  { key: 'patientId', label: 'Patient ID' },
  { key: 'composite', label: 'Composite' },
  { key: 'slideSim', label: 'Slide sim (α)' },
  { key: 'compSim', label: 'Comp sim (β)' },
  { key: 'metaSim', label: 'Meta sim (γ)' },
  { key: 'primaryTumorSite', label: 'Tumor site' },
  { key: 'histologicType', label: 'Histologic type' },
  { key: 'hpvAssociationP16', label: 'HPV/p16' },
  { key: 'ptStage', label: 'pT stage' },
  { key: 'pnStage', label: 'pN stage' },
  { key: 'smokingStatus', label: 'Smoking' },
  { key: 'ageAtInitialDiagnosis', label: 'Age at Diagnosis' },
  { key: 'yearOfInitialDiagnosis', label: 'Year of Diagnosis' },
]

export function exportRetrievalHtml(result: RetrievalResponse): void {
  const rows = [result.queryPatient, ...result.matches]

  const thead = `<tr>${RETRIEVAL_COLS.map((c) => `<th>${c.label}</th>`).join('')}</tr>`
  const tbody = rows
    .map((r) => {
      const cells = RETRIEVAL_COLS.map(({ key }) => {
        const val = r[key]
        if (val === null || val === undefined) return '<td>—</td>'
        if (typeof val === 'number' && !Number.isInteger(val)) return `<td>${val.toFixed(4)}</td>`
        if (key === 'rank' && val === 0) return '<td>0 (query)</td>'
        return `<td>${String(val)}</td>`
      })
      return `<tr>${cells.join('')}</tr>`
    })
    .join('\n')

  const body = `
  <h1>Patient Similarity Retrieval Report</h1>
  <p class="meta">Generated: ${new Date().toLocaleString()} &nbsp;|&nbsp; Query patient: ${result.queryPatient.patientId} &nbsp;|&nbsp; ${result.matches.length} matches</p>
  <table>
    <thead>${thead}</thead>
    <tbody>${tbody}</tbody>
  </table>`

  downloadHtml(`retrieval_report_${timestamp()}.html`, wrapPage('Retrieval Report', body))
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function timestamp(): string {
  return new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
}

let _plotlyBundle: string | null = null

/** Fetch the plotly.js bundle once and cache it in memory. */
async function fetchPlotlyBundle(): Promise<string> {
  if (_plotlyBundle) return _plotlyBundle
  // Use the CDN version so the export works without a local copy
  const res = await fetch('https://cdn.plot.ly/plotly-2.35.2.min.js')
  _plotlyBundle = await res.text()
  return _plotlyBundle
}
