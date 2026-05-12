// API client helpers for the EMBLASE Dashboard

const BASE = '/api'

export async function fetchJSON<T>(path: string): Promise<T> {
  const resp = await fetch(`${BASE}${path}`)
  if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${resp.statusText}`)
  return resp.json() as Promise<T>
}

export async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const resp = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!resp.ok) {
    const text = await resp.text()
    throw new Error(`HTTP ${resp.status}: ${text}`)
  }
  return resp.json() as Promise<T>
}

export async function deleteRequest(path: string): Promise<void> {
  const resp = await fetch(`${BASE}${path}`, { method: 'DELETE' })
  if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${resp.statusText}`)
}

export function createSSE(path: string, onMessage: (data: string) => void, onEvent?: (event: string, data: string) => void): () => void {
  const es = new EventSource(`${BASE}${path}`)
  es.onmessage = (e) => onMessage(e.data)
  if (onEvent) {
    es.addEventListener('state', (e: Event) => onEvent('state', (e as MessageEvent).data))
    es.addEventListener('job_done', (e: Event) => onEvent('job_done', (e as MessageEvent).data))
  }
  return () => es.close()
}
