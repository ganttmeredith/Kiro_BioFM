declare module 'plotly.js-dist' {
  export interface Layout {
    [key: string]: unknown
  }

  export interface ScatterMarker {
    color?: string | string[]
    size?: number | number[]
    symbol?: string | string[]
    line?: { color?: string; width?: number }
    [key: string]: unknown
  }

  export interface ScatterData {
    type?: string
    mode?: string
    x?: number[] | string[]
    y?: number[] | string[]
    text?: string | string[]
    customdata?: unknown
    name?: string
    marker?: ScatterMarker
    hovertemplate?: string
    [key: string]: unknown
  }

  // Alias so callers can use Plotly.Data
  export type Data = ScatterData

  export function react(
    root: HTMLElement,
    data: Data[],
    layout?: Partial<Layout>,
    config?: Record<string, unknown>,
  ): Promise<void>

  export function newPlot(
    root: HTMLElement,
    data: Data[],
    layout?: Partial<Layout>,
    config?: Record<string, unknown>,
  ): Promise<void>

  export function purge(root: HTMLElement): void
}
