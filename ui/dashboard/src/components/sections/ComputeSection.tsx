import React, { useEffect, useState, useRef, useCallback } from 'react'
import { Cpu, Activity, Copy, ChevronDown, ChevronUp, Clock,
         CheckCircle2, XCircle, Radio, Play, Square, Loader2 } from 'lucide-react'
import { fetchJSON, postJSON, createSSE } from '../../api'
import type { OrionStatus, NERSCStatus, AppConfig, JobRecord, WatcherStatus } from '../../types'
import { StatusBadge } from '../StatusBadge'
import { Card, CardHeader, CardBody } from '../Card'

interface ComputeSectionProps {
  config: AppConfig | null
  refreshTick: number
  jobHistory: JobRecord[]
  watcherStatus: WatcherStatus | null
  onSubmitJob: (backend: 'nersc' | 'orion') => void
  onLiveWatch: (backend: 'nersc' | 'orion') => void
}

type ComputeTab = 'nersc' | 'orion' | 'history'

// Slurm/HPC terminal states — no need to keep polling once reached
const TERMINAL_STATES = new Set([
  'completed', 'failed', 'cancelled', 'canceled', 'timeout',
  'node_fail', 'out_of_memory', 'done', 'finish',
])

/** Normalise any backend state string to lowercase for consistent comparison and display. */
function normaliseState(s: string): string {
  return s.trim().toLowerCase()
}

/** True when this state will never change again — safe to stop polling. */
function isTerminal(state: string): boolean {
  const n = normaliseState(state)
  for (const t of TERMINAL_STATES) if (n.includes(t)) return true
  return false
}

function stateIcon(jobState: string) {
  const s = normaliseState(jobState)
  if (!s) return <Activity className="h-3.5 w-3.5 text-muted" />
  if (s === 'scheduled')
    return <Clock className="h-3.5 w-3.5 text-sky-400" />
  if (s === 'stopping' || s.includes('cancel'))
    return <XCircle className="h-3.5 w-3.5 text-orange-400" />
  if (s.includes('running') || s.includes('active'))
    return <Radio className="h-3.5 w-3.5 text-violet-500 animate-pulse" />
  if (s.includes('pend') || s.includes('queue') || s.includes('configur'))
    return <Clock className="h-3.5 w-3.5 text-yellow-500" />
  if (s.includes('done') || s.includes('complet') || s.includes('finish'))
    return <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />
  if (s.includes('fail') || s.includes('timeout') || s.includes('node_fail') || s.includes('memory'))
    return <XCircle className="h-3.5 w-3.5 text-red-500" />
  return <Activity className="h-3.5 w-3.5 text-muted" />
}

function ts(): string {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

// ── Section ───────────────────────────────────────────────────────────────────

export function ComputeSection({
  config, refreshTick, jobHistory,
  watcherStatus, onSubmitJob, onLiveWatch,
}: ComputeSectionProps) {
  const [tab, setTab] = useState<ComputeTab>('nersc')

  // Auto-switch to Jobs tab when a new job is submitted (not on initial load)
  const prevCount = useRef(jobHistory.length)
  useEffect(() => {
    if (jobHistory.length > prevCount.current) {
      prevCount.current = jobHistory.length
      setTab('history')
    } else {
      prevCount.current = jobHistory.length
    }
  }, [jobHistory.length])

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-violet-500/10 border border-violet-500/30">
            <Cpu className="h-4 w-4 text-violet-500" />
          </div>
          <div>
            <h3 className="font-semibold text-primary">Compute</h3>
            <p className="text-xs text-secondary">HPC job backends</p>
          </div>
        </div>
        <div className="flex rounded-lg border border-theme overflow-hidden text-xs">
          {(['nersc', 'orion', 'history'] as const).map(t => (
            <button key={t} onClick={() => setTab(t)}
              className={`px-3 py-1.5 font-medium transition-colors ${
                tab === t ? 'bg-violet-600 dark:bg-violet-900 text-white dark:text-violet-200'
                          : 'bg-card text-secondary hover:text-primary'
              }`}>
              {t === 'history' ? 'Jobs' : t.toUpperCase()}
              {t === 'history' && jobHistory.length > 0 && (
                <span className="ml-1 rounded-full bg-indigo-500 text-white text-[10px] px-1 leading-tight">
                  {jobHistory.length}
                </span>
              )}
            </button>
          ))}
        </div>
      </CardHeader>
      <CardBody>
        {tab === 'nersc' && (
          <NERSCPanel config={config} refreshTick={refreshTick}
            watcherStatus={watcherStatus}
            onSubmitJob={() => onSubmitJob('nersc')}
            onLiveWatch={() => onLiveWatch('nersc')} />
        )}
        {tab === 'orion' && (
          <OrionPanel config={config} refreshTick={refreshTick}
            watcherStatus={watcherStatus}
            onSubmitJob={() => onSubmitJob('orion')}
            onLiveWatch={() => onLiveWatch('orion')} />
        )}
        {tab === 'history' && <JobHistoryPanel jobs={jobHistory} />}
      </CardBody>
    </Card>
  )
}

// ── Job history panel ─────────────────────────────────────────────────────────

function JobHistoryPanel({ jobs }: { jobs: JobRecord[] }) {
  const [expandedId, setExpandedId] = useState<string | null>(null)

  if (jobs.length === 0) {
    return (
      <p className="text-xs text-muted italic py-4 text-center">
        No jobs in this session. Use "Submit Job" in the NERSC or Orion panel.
      </p>
    )
  }

  return (
    <div className="h-48 overflow-y-auto scrollbar-thin space-y-2 pr-0.5">
      {jobs.map(job => (
        <JobHistoryItem key={job.job_id} job={job}
          expanded={expandedId === job.job_id}
          onToggle={() => setExpandedId(prev => prev === job.job_id ? null : job.job_id)} />
      ))}
    </div>
  )
}

// ── Terminal-job cache ────────────────────────────────────────────────────────
// Persists final state + log lines for jobs that have reached a terminal state,
// so re-mounting the component (e.g. switching tabs) never re-fetches.
interface CachedJob { state: string; lines: string[] }
const terminalCache = new Map<string, CachedJob>()

// Known-state cache — stores the last-known state for any job, including
// non-terminal ones. Initialised from job.state (from localStorage) so the
// correct icon is shown instantly on mount without waiting for a poll.
const knownStateCache = new Map<string, string>()

// Poll intervals
const NERSC_STATE_POLL_MS = 15_000  // NERSC background fallback; SSE is primary
const ORION_STATE_POLL_MS = 8_000

// ── Dispatcher ────────────────────────────────────────────────────────────────

function JobHistoryItem({ job, expanded, onToggle }: {
  job: JobRecord; expanded: boolean; onToggle: () => void
}) {
  if (job.backend === 'orion') {
    return <OrionJobItem job={job} expanded={expanded} onToggle={onToggle} />
  }
  return <NERSCJobItem job={job} expanded={expanded} onToggle={onToggle} />
}

// ── Shared header row ─────────────────────────────────────────────────────────

function JobItemHeader({ job, jobState, polling, expanded, onToggle, onCancel }: {
  job: JobRecord; jobState: string; polling: boolean; expanded: boolean; onToggle: () => void
  onCancel?: () => void
}) {
  const copyId = () => navigator.clipboard.writeText(job.job_id).catch(() => {})
  const submittedTs = new Date(job.submitted_at * 1000)
    .toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })

  const isStopping = normaliseState(jobState) === 'stopping'
  const isScheduled = normaliseState(jobState) === 'scheduled'
  // show cancel button when: state is known, not terminal, not already stopping, not a placeholder scheduled id, not fetching
  const showCancel = onCancel
    && !isTerminal(jobState)
    && !isStopping
    && jobState !== ''
    && !polling
    && !job.job_id.startsWith('scheduled-')

  return (
    <button onClick={onToggle}
      className="w-full flex items-center gap-2 px-3 py-2 bg-card hover:bg-row-hover transition-colors text-left">
      {polling
        ? <Loader2 className="h-3.5 w-3.5 text-violet-400 animate-spin flex-shrink-0" />
        : stateIcon(jobState)}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-1.5 flex-wrap">
          <code className="text-xs font-mono text-primary">{job.job_id}</code>
          {!isScheduled && (
            <button onClick={e => { e.stopPropagation(); copyId() }}
              className="text-muted hover:text-secondary transition-colors">
              <Copy className="h-3 w-3" />
            </button>
          )}
          {jobState && (
            <span className={`text-[10px] font-mono bg-page border border-theme rounded px-1 ${
              isStopping ? 'text-orange-400' : isScheduled ? 'text-sky-400' : 'text-muted'
            }`}>
              {isStopping ? 'stopping…' : jobState}
            </span>
          )}
        </div>
        <p className="text-xs text-muted truncate">
          {job.mode} · {job.backend.toUpperCase()} · {job.model_name}
        </p>
      </div>
      <div className="flex items-center gap-1.5 flex-shrink-0">
        <span className="text-xs text-muted">{submittedTs}</span>
        {showCancel && (
          <button
            onClick={e => { e.stopPropagation(); onCancel!() }}
            title="Cancel job"
            className="text-muted hover:text-red-500 transition-colors">
            <Square className="h-3.5 w-3.5" />
          </button>
        )}
        {expanded
          ? <ChevronUp className="h-3.5 w-3.5 text-muted" />
          : <ChevronDown className="h-3.5 w-3.5 text-muted" />}
      </div>
    </button>
  )
}

// ── Shared log panel ──────────────────────────────────────────────────────────

function LogPanel({ lines, placeholder }: { lines: string[]; placeholder: string }) {
  const containerRef = useRef<HTMLDivElement>(null)
  // Auto-scroll the log panel itself (not the page) when new lines arrive
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    el.scrollTop = el.scrollHeight
  }, [lines.length])

  return (
    <div className="border-t border-theme bg-page">
      <div ref={containerRef} className="h-56 overflow-y-auto scrollbar-thin p-3 font-mono text-xs leading-relaxed">
        {lines.length === 0
          ? <span className="text-muted italic">{placeholder}</span>
          : lines.map((line, i) => (
            <div key={i} className="whitespace-pre-wrap text-primary">{line}</div>
          ))}
      </div>
    </div>
  )
}

// ── Orion job item ────────────────────────────────────────────────────────────
// Polls /compute/orion/jobs/{id}/status every ORION_STATE_POLL_MS.
// Skips polling entirely if already terminal (terminalCache). Uses knownStateCache
// to show the correct icon immediately on re-mount without a blocking fetch.

function OrionJobItem({ job, expanded, onToggle }: {
  job: JobRecord; expanded: boolean; onToggle: () => void
}) {
  const cached = terminalCache.get(job.job_id)

  // Seed knownStateCache from job record on first encounter
  const initialState = (() => {
    if (cached) return cached.state
    const known = knownStateCache.get(job.job_id)
    if (known) return known
    const s = normaliseState(job.state || '')
    if (s) knownStateCache.set(job.job_id, s)
    return s
  })()

  const [jobState, setJobState] = useState(initialState)
  const [lines, setLines] = useState<string[]>(() => cached?.lines ?? [])
  // fetching = true only during an in-flight HTTP request (not between polls)
  const [fetching, setFetching] = useState(false)
  const done = !!cached || isTerminal(initialState)

  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const pollErrorsRef = useRef(0)
  const jobStateRef = useRef(jobState)
  useEffect(() => { jobStateRef.current = jobState }, [jobState])

  const appendLine = useCallback((text: string) => {
    setLines(prev => [...prev, `[${ts()}] ${text}`])
  }, [])

  const stopPolling = useCallback(() => {
    if (pollTimerRef.current) { clearInterval(pollTimerRef.current); pollTimerRef.current = null }
  }, [])

  const poll = useCallback(() => {
    if (isTerminal(jobStateRef.current)) { stopPolling(); return }
    setFetching(true)
    fetchJSON<{ state?: string; node?: string | null; not_found?: boolean; error?: string }>(
      `/compute/orion/jobs/${job.job_id}/status`
    ).then(d => {
      setFetching(false)
      if (!d.state) {
        pollErrorsRef.current++
        if (pollErrorsRef.current >= 3) stopPolling()
        return
      }
      pollErrorsRef.current = 0
      const ns = normaliseState(d.state)
      setJobState(ns)
      jobStateRef.current = ns
      knownStateCache.set(job.job_id, ns)

      const node = d.node ? `  node=${d.node}` : ''
      const note = d.not_found ? '  (purged from API)' : ''
      appendLine(`state=${ns}${node}${note}`)

      if (isTerminal(ns)) {
        stopPolling()
        setLines(prev => {
          terminalCache.set(job.job_id, { state: ns, lines: prev })
          return prev
        })
      }
    }).catch(() => {
      setFetching(false)
      pollErrorsRef.current++
      if (pollErrorsRef.current >= 3) stopPolling()
    })
  }, [job.job_id, appendLine, stopPolling])

  useEffect(() => {
    if (done) return
    poll()
    pollTimerRef.current = setInterval(poll, ORION_STATE_POLL_MS)
    return stopPolling
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="rounded-lg border border-theme overflow-hidden">
      <JobItemHeader job={job} jobState={jobState} polling={fetching} expanded={expanded} onToggle={onToggle}
        onCancel={() => {
          const prevState = jobStateRef.current
          setJobState('stopping')
          jobStateRef.current = 'stopping'
          fetch(`/compute/orion/jobs/${job.job_id}`, { method: 'DELETE' })
            .then(r => {
              if (!r.ok) throw new Error(`HTTP ${r.status}`)
              // keep polling until confirmed terminal
              if (!pollTimerRef.current) {
                pollTimerRef.current = setInterval(poll, ORION_STATE_POLL_MS)
              }
            })
            .catch(() => {
              // revert — cancel request failed
              setJobState(prevState)
              jobStateRef.current = prevState
            })
        }} />
      {expanded && (
        <LogPanel
          lines={lines}
          placeholder={jobState ? `state=${jobState}` : 'Fetching status…'}
        />
      )}
    </div>
  )
}

// ── NERSC job item ────────────────────────────────────────────────────────────
// Background status poll until SSE connects or terminal. SSE streams job.out
// live when expanded. Terminal state + lines frozen in terminalCache.
// knownStateCache ensures correct icon is shown on re-mount without blocking.

function NERSCJobItem({ job, expanded, onToggle }: {
  job: JobRecord; expanded: boolean; onToggle: () => void
}) {
  const cached = terminalCache.get(job.job_id)

  const initialState = (() => {
    if (cached) return cached.state
    const known = knownStateCache.get(job.job_id)
    if (known) return known
    const s = normaliseState(job.state || '')
    if (s) knownStateCache.set(job.job_id, s)
    return s
  })()

  const [jobState, setJobState] = useState(initialState)
  const [lines, setLines] = useState<string[]>(() => cached?.lines ?? [])
  const [sseStarted, setSseStarted] = useState(false)
  // fetching = true only while an HTTP request or SSE stream is in-flight
  const [fetching, setFetching] = useState(false)
  const done = !!cached || isTerminal(initialState)

  const stopSseRef = useRef<(() => void) | null>(null)
  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const sseActiveRef = useRef(false)
  const pollErrorsRef = useRef(0)
  const jobStateRef = useRef(jobState)
  useEffect(() => { jobStateRef.current = jobState }, [jobState])

  const stopPolling = useCallback(() => {
    if (pollTimerRef.current) { clearInterval(pollTimerRef.current); pollTimerRef.current = null }
  }, [])

  const pollState = useCallback(() => {
    if (sseActiveRef.current || isTerminal(jobStateRef.current)) { stopPolling(); return }
    setFetching(true)
    fetchJSON<{ state?: string }>(`/compute/nersc/jobs/${job.job_id}/status`)
      .then(d => {
        setFetching(false)
        if (!d.state) { pollErrorsRef.current++; if (pollErrorsRef.current >= 3) stopPolling(); return }
        pollErrorsRef.current = 0
        const ns = normaliseState(d.state)
        setJobState(ns)
        jobStateRef.current = ns
        knownStateCache.set(job.job_id, ns)
        if (isTerminal(ns)) stopPolling()
      }).catch(() => {
        setFetching(false)
        pollErrorsRef.current++
        if (pollErrorsRef.current >= 3) stopPolling()
      })
  }, [job.job_id, stopPolling])

  useEffect(() => {
    if (done) return
    pollState()
    pollTimerRef.current = setInterval(pollState, NERSC_STATE_POLL_MS)
    return () => {
      stopPolling()
      stopSseRef.current?.()
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!expanded || sseStarted || done || job.backend !== 'nersc') return
    setSseStarted(true)

    const startSSE = (path: string) => {
      sseActiveRef.current = true
      setFetching(true)
      stopPolling()

      const params = path ? `?log_path=${encodeURIComponent(path)}` : ''
      stopSseRef.current = createSSE(
        `/compute/nersc/jobs/${job.job_id}/logs${params}`,
        (data) => {
          setLines(prev => [...prev, data])
        },
        (event, data) => {
          if (event === 'state' || event === 'job_done') {
            const ns = normaliseState(data)
            setJobState(ns)
            jobStateRef.current = ns
            knownStateCache.set(job.job_id, ns)
            if (event === 'job_done') {
              sseActiveRef.current = false
              setFetching(false)
              stopSseRef.current?.()
              setLines(prev => {
                terminalCache.set(job.job_id, { state: ns, lines: prev })
                return prev
              })
            }
          }
        }
      )
    }

    if (job.log_path) {
      startSSE(job.log_path)
    } else {
      fetchJSON<{ state?: string; log_path?: string }>(
        `/compute/nersc/jobs/${job.job_id}/status`
      ).then(d => {
        if (d.state) { setJobState(normaliseState(d.state)); knownStateCache.set(job.job_id, normaliseState(d.state)) }
        startSSE(d.log_path || '')
      }).catch(() => startSSE(''))
    }
  }, [expanded]) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="rounded-lg border border-theme overflow-hidden">
      <JobItemHeader job={job} jobState={jobState} polling={fetching} expanded={expanded} onToggle={onToggle}
        onCancel={() => {
          const prevState = jobStateRef.current
          setJobState('stopping')
          jobStateRef.current = 'stopping'
          fetch(`/compute/nersc/jobs/${job.job_id}`, { method: 'DELETE' })
            .then(r => {
              if (!r.ok) throw new Error(`HTTP ${r.status}`)
              // resume background poll to confirm cancellation
              if (!pollTimerRef.current && !sseActiveRef.current) {
                pollTimerRef.current = setInterval(pollState, NERSC_STATE_POLL_MS)
              }
            })
            .catch(() => {
              // revert — cancel request failed
              setJobState(prevState)
              jobStateRef.current = prevState
            })
        }} />
      {expanded && (
        <LogPanel
          lines={lines}
          placeholder={sseStarted ? 'Connecting to log stream…' : (jobState ? `state=${jobState}` : 'Waiting…')}
        />
      )}
    </div>
  )
}

// ── Shared action buttons ─────────────────────────────────────────────────────

interface BackendActionsProps {
  backend: 'nersc' | 'orion'
  watcherStatus: WatcherStatus | null
  onSubmitJob: () => void
  onLiveWatch: () => void
}

function BackendActions({ backend, watcherStatus, onSubmitJob, onLiveWatch }: BackendActionsProps) {
  const watcherRunning = watcherStatus?.status === 'running'
  const thisWatcherActive = watcherRunning && watcherStatus?.backend === backend

  const handleStopWatcher = async () => {
    try { await postJSON('/watcher/stop', {}) } catch {}
  }

  return (
    <div className="flex gap-2 pt-2 border-t border-theme mt-3">
      <button onClick={onSubmitJob}
        disabled={watcherRunning}
        title={watcherRunning ? 'Live Watch is active — stop it before submitting a manual job' : undefined}
        className="flex-1 flex items-center justify-center gap-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:bg-indigo-900 disabled:text-indigo-400 text-white text-xs font-medium py-2 transition-colors">
        <Play className="h-3.5 w-3.5" /> Submit Job
      </button>
      {thisWatcherActive ? (
        <button onClick={handleStopWatcher}
          className="flex-1 flex items-center justify-center gap-1.5 rounded-lg bg-red-600 hover:bg-red-500 text-white text-xs font-medium py-2 transition-colors">
          <Square className="h-3.5 w-3.5" /> Stop Watch
        </button>
      ) : (
        <button onClick={onLiveWatch}
          disabled={watcherRunning}
          title={watcherRunning ? `Watcher already running on ${watcherStatus?.backend?.toUpperCase()}` : undefined}
          className="flex-1 flex items-center justify-center gap-1.5 rounded-lg bg-violet-600 hover:bg-violet-500 disabled:bg-violet-900 disabled:text-violet-400 text-white text-xs font-medium py-2 transition-colors">
          <Radio className="h-3.5 w-3.5" /> Live Watch
        </button>
      )}
    </div>
  )
}

// ── NERSC status panel ────────────────────────────────────────────────────────

function NERSCPanel({ config, refreshTick, watcherStatus, onSubmitJob, onLiveWatch }: {
  config: AppConfig | null; refreshTick: number
  watcherStatus: WatcherStatus | null
  onSubmitJob: () => void; onLiveWatch: () => void
}) {
  const [status, setStatus] = useState<NERSCStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [resolvedStatus, setResolvedStatus] = useState<NERSCStatus['status']>('loading')

  useEffect(() => {
    setLoading(true)
    fetchJSON<NERSCStatus>('/compute/nersc/status')
      .then(s => { setStatus(s); setResolvedStatus(s.status) })
      .catch(() => { setStatus({ status: 'error', error: 'Could not reach NERSC API' }); setResolvedStatus('error') })
      .finally(() => setLoading(false))
  }, [refreshTick])

  const svcStatus = loading && resolvedStatus === 'loading' ? 'loading' : resolvedStatus
  const watcherActive = watcherStatus?.status === 'running' && watcherStatus?.backend === 'nersc'

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <StatusBadge status={svcStatus} size="sm" />
        {status?.api_uri && <span className="text-xs text-muted font-mono truncate">{status.api_uri}</span>}
      </div>
      {status?.status === 'online' && (
        <div className="grid grid-cols-2 gap-2">
          <InfoItem label="Resource"   value={status.resource_id} />
          <InfoItem label="Queue"      value={status.queue || config?.nersc_queue} />
          <InfoItem label="Account"    value={status.account || config?.nersc_account} />
          <InfoItem label="Time Limit" value={status.time_limit || config?.nersc_time_limit} />
        </div>
      )}
      {status?.error && (
        <p className="text-xs text-red-500 bg-red-500/10 border border-red-500/20 rounded px-2 py-1.5 font-mono">
          {status.error}
        </p>
      )}
      {status?.message && <p className="text-xs text-secondary">{status.message}</p>}
      {watcherActive && (
        <div className="flex items-center gap-2 rounded bg-violet-500/10 border border-violet-500/20 px-2 py-1.5">
          <Radio className="h-3.5 w-3.5 text-violet-500 animate-pulse flex-shrink-0" />
          <p className="text-xs text-violet-400">
            Live Watch active — {watcherStatus!.jobs_submitted} job{watcherStatus!.jobs_submitted !== 1 ? 's' : ''} submitted
          </p>
        </div>
      )}
      <BackendActions backend="nersc" watcherStatus={watcherStatus}
        onSubmitJob={onSubmitJob} onLiveWatch={onLiveWatch} />
    </div>
  )
}

// ── Orion status panel ────────────────────────────────────────────────────────

function OrionPanel({ config, refreshTick, watcherStatus, onSubmitJob, onLiveWatch }: {
  config: AppConfig | null; refreshTick: number
  watcherStatus: WatcherStatus | null
  onSubmitJob: () => void; onLiveWatch: () => void
}) {
  const [status, setStatus] = useState<OrionStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [resolvedStatus, setResolvedStatus] = useState<OrionStatus['status']>('loading')

  useEffect(() => {
    setLoading(true)
    fetchJSON<OrionStatus>('/compute/orion/status')
      .then(s => { setStatus(s); setResolvedStatus(s.status) })
      .catch(() => { setStatus({ status: 'error', error: 'Could not reach Orion API' }); setResolvedStatus('error') })
      .finally(() => setLoading(false))
  }, [refreshTick])

  const svcStatus = loading && resolvedStatus === 'loading' ? 'loading' : resolvedStatus
  const watcherActive = watcherStatus?.status === 'running' && watcherStatus?.backend === 'orion'

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <StatusBadge status={svcStatus} size="sm" />
        {status?.api_url && <span className="text-xs text-muted font-mono truncate">{status.api_url}</span>}
      </div>
      {status?.status === 'online' && (
        <div className="grid grid-cols-2 gap-2">
          <InfoItem label="Cluster"     value={status.cluster} />
          <InfoItem label="Account"     value={status.account} />
          <InfoItem label="Active Jobs" value={String(status.active_jobs ?? 0)} />
        </div>
      )}
      {status?.error && (
        <p className="text-xs text-red-500 bg-red-500/10 border border-red-500/20 rounded px-2 py-1.5 font-mono">
          {status.error}
        </p>
      )}
      {watcherActive && (
        <div className="flex items-center gap-2 rounded bg-violet-500/10 border border-violet-500/20 px-2 py-1.5">
          <Radio className="h-3.5 w-3.5 text-violet-500 animate-pulse flex-shrink-0" />
          <p className="text-xs text-violet-400">
            Live Watch active — {watcherStatus!.jobs_submitted} job{watcherStatus!.jobs_submitted !== 1 ? 's' : ''} submitted
          </p>
        </div>
      )}
      <BackendActions backend="orion" watcherStatus={watcherStatus}
        onSubmitJob={onSubmitJob} onLiveWatch={onLiveWatch} />
    </div>
  )
}

function InfoItem({ label, value }: { label: string; value?: string }) {
  return (
    <div className="rounded bg-page border border-theme px-2.5 py-2">
      <p className="text-xs text-muted">{label}</p>
      <p className="text-xs text-primary font-mono truncate">{value || '—'}</p>
    </div>
  )
}
