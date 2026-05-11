import React, { useEffect, useState, useRef, useCallback } from 'react'
import { Cpu, Activity, Copy, ChevronDown, ChevronUp, Clock,
         CheckCircle2, XCircle, Radio, Play, Square } from 'lucide-react'
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
  if (s.includes('running') || s.includes('active'))
    return <Radio className="h-3.5 w-3.5 text-violet-500 animate-pulse" />
  if (s.includes('pend') || s.includes('queue') || s.includes('configur'))
    return <Clock className="h-3.5 w-3.5 text-yellow-500" />
  if (s.includes('done') || s.includes('complet') || s.includes('finish'))
    return <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />
  if (s.includes('fail') || s.includes('cancel') || s.includes('timeout') || s.includes('node_fail') || s.includes('memory'))
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

  // Auto-switch to Jobs tab when a new job is submitted
  const prevCount = useRef(0)
  useEffect(() => {
    if (jobHistory.length > prevCount.current) {
      prevCount.current = jobHistory.length
      setTab('history')
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
    <div className="space-y-2">
      {jobs.map(job => (
        <JobHistoryItem key={job.job_id} job={job}
          expanded={expandedId === job.job_id}
          onToggle={() => setExpandedId(prev => prev === job.job_id ? null : job.job_id)} />
      ))}
    </div>
  )
}

// Poll intervals
const NERSC_STATE_POLL_MS = 10_000
const ORION_STATE_POLL_MS = 6_000

function JobHistoryItem({ job, expanded, onToggle }: {
  job: JobRecord; expanded: boolean; onToggle: () => void
}) {
  // jobState: always normalised to lowercase
  const [jobState, setJobState] = useState(() => normaliseState(job.state || ''))
  const [logLines, setLogLines] = useState<string[]>([])
  const [logStarted, setLogStarted] = useState(false)
  const stopLogRef = useRef<(() => void) | null>(null)
  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Refs so interval callbacks always see fresh values without needing re-creation
  const expandedRef = useRef(expanded)
  const jobStateRef = useRef(jobState)
  useEffect(() => { expandedRef.current = expanded }, [expanded])
  useEffect(() => { jobStateRef.current = jobState }, [jobState])

  const appendLog = useCallback((line: string) => {
    setLogLines(prev => [...prev, `[${ts()}] ${line}`])
  }, [])

  const pollMs = job.backend === 'nersc' ? NERSC_STATE_POLL_MS : ORION_STATE_POLL_MS

  // ── Background state polling ───────────────────────────────────────────────
  // Only poll while state is non-terminal. Stops itself once done/failed/etc.
  const fetchState = useCallback(() => {
    // Stop polling if we already know this job is done
    if (isTerminal(jobStateRef.current)) {
      if (pollTimerRef.current) {
        clearInterval(pollTimerRef.current)
        pollTimerRef.current = null
      }
      return
    }

    if (job.backend === 'nersc') {
      fetchJSON<{ state?: string; log_path?: string }>(
        `/compute/nersc/jobs/${job.job_id}/status`
      ).then(d => {
        if (d.state) {
          const ns = normaliseState(d.state)
          setJobState(ns)
          // Stop interval if now terminal
          if (isTerminal(ns) && pollTimerRef.current) {
            clearInterval(pollTimerRef.current)
            pollTimerRef.current = null
          }
        }
      }).catch(() => {})
    } else {
      fetchJSON<{ job_id?: number; state?: string; node?: string | null; error?: string }>(
        `/compute/orion/jobs/${job.job_id}/status`
      ).then(d => {
        if (!d.state) return  // ignore empty/error responses — don't clobber known state
        const ns = normaliseState(d.state)
        setJobState(ns)
        if (isTerminal(ns) && pollTimerRef.current) {
          clearInterval(pollTimerRef.current)
          pollTimerRef.current = null
        }
        // Append a timestamped status line only when expanded
        if (expandedRef.current) {
          const node = d.node || '(queued)'
          appendLog(`state=${ns}  node=${node}`)
        }
      }).catch(() => {})
    }
  }, [job.backend, job.job_id, appendLog])

  // Start polling on mount; stop on unmount
  useEffect(() => {
    // Don't even start if already terminal from persisted state
    if (!isTerminal(jobState)) {
      fetchState()
      pollTimerRef.current = setInterval(fetchState, pollMs)
    }
    return () => {
      if (pollTimerRef.current) clearInterval(pollTimerRef.current)
      stopLogRef.current?.()
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // ── Orion: fetch and display status immediately on first expand ───────────
  useEffect(() => {
    if (!expanded || job.backend !== 'orion') return
    if (logLines.length > 0) return  // already seeded

    fetchJSON<{ state?: string; node?: string | null }>(
      `/compute/orion/jobs/${job.job_id}/status`
    ).then(d => {
      if (d.state) {
        const ns = normaliseState(d.state)
        setJobState(ns)
        const node = d.node || '(queued)'
        appendLog(`state=${ns}  node=${node}`)
      } else if (jobState) {
        appendLog(`state=${jobState}  (no update)`)
      }
    }).catch(() => {
      if (jobState) appendLog(`state=${jobState}  (poll failed)`)
    })
  }, [expanded]) // eslint-disable-line react-hooks/exhaustive-deps

  // ── NERSC: start SSE log stream once on first expand ─────────────────────
  useEffect(() => {
    if (!expanded || logStarted || job.backend !== 'nersc') return
    setLogStarted(true)

    const start = (path: string) => {
      const params = path ? `?log_path=${encodeURIComponent(path)}` : ''
      stopLogRef.current = createSSE(
        `/compute/nersc/jobs/${job.job_id}/logs${params}`,
        (data) => {
          // Skip internal discovery messages (not meaningful to the user)
          if (data.startsWith('[log: ')) return
          setLogLines(prev => [...prev, data])
        },
        (event, data) => {
          if (event === 'state' || event === 'job_done') setJobState(normaliseState(data))
        }
      )
    }

    if (job.log_path) {
      start(job.log_path)
    } else {
      fetchJSON<{ state?: string; log_path?: string }>(
        `/compute/nersc/jobs/${job.job_id}/status`
      ).then(d => {
        if (d.state) setJobState(normaliseState(d.state))
        start(d.log_path || '')
      }).catch(() => start(''))
    }
  }, [expanded]) // eslint-disable-line react-hooks/exhaustive-deps

  const copyId = () => navigator.clipboard.writeText(job.job_id).catch(() => {})
  const submittedTs = new Date(job.submitted_at * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })

  return (
    <div className="rounded-lg border border-theme overflow-hidden">
      <button onClick={onToggle}
        className="w-full flex items-center gap-2 px-3 py-2 bg-card hover:bg-row-hover transition-colors text-left">
        {stateIcon(jobState)}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5 flex-wrap">
            <code className="text-xs font-mono text-primary">{job.job_id}</code>
            <button onClick={e => { e.stopPropagation(); copyId() }}
              className="text-muted hover:text-secondary transition-colors">
              <Copy className="h-3 w-3" />
            </button>
            {jobState && (
              <span className="text-[10px] text-muted font-mono bg-page border border-theme rounded px-1">
                {jobState}
              </span>
            )}
          </div>
          <p className="text-xs text-muted truncate">
            {job.mode} · {job.backend.toUpperCase()} · {job.model_name}
          </p>
        </div>
        <div className="flex items-center gap-1.5 flex-shrink-0">
          <span className="text-xs text-muted">{submittedTs}</span>
          {expanded ? <ChevronUp className="h-3.5 w-3.5 text-muted" /> : <ChevronDown className="h-3.5 w-3.5 text-muted" />}
        </div>
      </button>

      {expanded && (
        <div className="border-t border-theme bg-page">
          <div className="h-56 overflow-y-auto scrollbar-thin p-3 font-mono text-xs leading-relaxed">
            {logLines.length === 0
              ? <span className="text-muted italic">
                  {job.backend === 'nersc' ? 'Connecting to log stream…' : 'Fetching status…'}
                </span>
              : logLines.map((line, i) => (
                <div key={i} className="whitespace-pre-wrap text-primary">{line}</div>
              ))}
          </div>
        </div>
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
