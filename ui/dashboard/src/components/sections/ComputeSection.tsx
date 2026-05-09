import React, { useEffect, useState } from 'react'
import { Cpu, ChevronDown, ChevronUp, Clock, Hash, Activity } from 'lucide-react'
import { fetchJSON, createSSE } from '../../api'
import type { OrionStatus, NERSCStatus, AppConfig } from '../../types'
import { StatusBadge } from '../StatusBadge'
import { Card, CardHeader, CardBody } from '../Card'
import { LogViewer } from '../LogViewer'

interface ComputeSectionProps {
  config: AppConfig | null
  refreshTick: number
}

type ComputeTab = 'orion' | 'nersc'

export function ComputeSection({ config, refreshTick }: ComputeSectionProps) {
  const [tab, setTab] = useState<ComputeTab>('nersc')

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-violet-950 border border-violet-800">
            <Cpu className="h-4 w-4 text-violet-400" />
          </div>
          <div>
            <h3 className="font-semibold text-gray-100">Compute</h3>
            <p className="text-xs text-gray-500">HPC job backends</p>
          </div>
        </div>
        {/* Tab switcher */}
        <div className="flex rounded-lg border border-gray-800 overflow-hidden text-xs">
          {(['nersc', 'orion'] as const).map(t => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`px-3 py-1.5 font-medium transition-colors ${
                tab === t
                  ? 'bg-violet-900 text-violet-200'
                  : 'bg-gray-900 text-gray-500 hover:text-gray-300'
              }`}
            >
              {t === 'nersc' ? 'NERSC' : 'Orion'}
            </button>
          ))}
        </div>
      </CardHeader>
      <CardBody>
        {tab === 'nersc' ? (
          <NERSCPanel config={config} refreshTick={refreshTick} />
        ) : (
          <OrionPanel config={config} refreshTick={refreshTick} />
        )}
      </CardBody>
    </Card>
  )
}

// ── NERSC panel ───────────────────────────────────────────────────────────────

function NERSCPanel({ config, refreshTick }: { config: AppConfig | null; refreshTick: number }) {
  const [status, setStatus] = useState<NERSCStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [showLogs, setShowLogs] = useState(false)
  const [activeJobId, setActiveJobId] = useState('')
  const [logPath, setLogPath] = useState('')
  const [logLines, setLogLines] = useState<string[]>([])
  const [jobState, setJobState] = useState<string>('')

  const fetchStatus = async () => {
    setLoading(true)
    try {
      const data = await fetchJSON<NERSCStatus>('/compute/nersc/status')
      setStatus(data)
    } catch {
      setStatus({ status: 'error', error: 'Could not reach NERSC API via dashboard backend' })
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchStatus() }, [refreshTick])

  const startLogStream = () => {
    if (!activeJobId) return
    setLogLines([])
    const params = logPath ? `?log_path=${encodeURIComponent(logPath)}` : ''
    const stop = createSSE(
      `/compute/nersc/jobs/${activeJobId}/logs${params}`,
      (data) => setLogLines(prev => [...prev, data]),
      (event, data) => {
        if (event === 'state') setJobState(data)
        if (event === 'job_done') setJobState(`Done: ${data}`)
      }
    )
    return stop
  }

  const svcStatus = loading ? 'loading' : (status?.status ?? 'error')

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <StatusBadge status={svcStatus} size="sm" />
        {status?.api_uri && (
          <span className="text-xs text-gray-600 font-mono truncate">{status.api_uri}</span>
        )}
      </div>

      {status && status.status === 'online' && (
        <div className="grid grid-cols-2 gap-2">
          <InfoItem label="Resource" value={status.resource_id} />
          <InfoItem label="Queue" value={status.queue || config?.nersc_queue} />
          <InfoItem label="Account" value={status.account || config?.nersc_account} />
          <InfoItem label="Time Limit" value={status.time_limit || config?.nersc_time_limit} />
        </div>
      )}

      {status?.error && (
        <p className="text-xs text-red-400 bg-red-950/50 border border-red-900 rounded px-2 py-1.5 font-mono">
          {status.error}
        </p>
      )}
      {status?.message && (
        <p className="text-xs text-gray-500">{status.message}</p>
      )}

      {/* Log viewer for a known job */}
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <input
            type="text"
            placeholder="Job ID (e.g. 52700081)"
            value={activeJobId}
            onChange={e => setActiveJobId(e.target.value)}
            className="flex-1 bg-gray-950 border border-gray-800 rounded px-2 py-1.5 text-xs text-gray-300 font-mono placeholder-gray-700 focus:outline-none focus:border-violet-700"
          />
          <button
            onClick={() => { setShowLogs(true); startLogStream() }}
            disabled={!activeJobId}
            className="px-3 py-1.5 text-xs rounded bg-violet-900 text-violet-200 hover:bg-violet-800 disabled:opacity-40 transition-colors"
          >
            Stream Logs
          </button>
        </div>
        <input
          type="text"
          placeholder="Log path on /pscratch (optional)"
          value={logPath}
          onChange={e => setLogPath(e.target.value)}
          className="w-full bg-gray-950 border border-gray-800 rounded px-2 py-1.5 text-xs text-gray-300 font-mono placeholder-gray-700 focus:outline-none focus:border-violet-700"
        />
        {jobState && (
          <div className="flex items-center gap-1.5 text-xs">
            <Activity className="h-3 w-3 text-violet-400" />
            <span className="text-gray-400">State: </span>
            <span className="text-violet-300 font-mono">{jobState}</span>
          </div>
        )}
      </div>

      {showLogs && <LogViewer lines={logLines} title={`NERSC job ${activeJobId}`} />}
    </div>
  )
}

// ── Orion panel ───────────────────────────────────────────────────────────────

function OrionPanel({ config, refreshTick }: { config: AppConfig | null; refreshTick: number }) {
  const [status, setStatus] = useState<OrionStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [showJob, setShowJob] = useState(false)
  const [activeJobId, setActiveJobId] = useState('')
  const [stateEvents, setStateEvents] = useState<string[]>([])

  const fetchStatus = async () => {
    setLoading(true)
    try {
      const data = await fetchJSON<OrionStatus>('/compute/orion/status')
      setStatus(data)
    } catch {
      setStatus({ status: 'error', error: 'Could not reach Orion API via dashboard backend' })
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchStatus() }, [refreshTick])

  const startPolling = () => {
    if (!activeJobId) return
    setStateEvents([])
    setShowJob(true)
    const stop = createSSE(
      `/jobs/orion/${activeJobId}/logs`,
      (data) => setStateEvents(prev => [...prev, data]),
      (event, data) => {
        if (event === 'state') setStateEvents(prev => [...prev, `State: ${data}`])
        if (event === 'job_done') setStateEvents(prev => [...prev, `✓ Done: ${data}`])
      }
    )
  }

  const svcStatus = loading ? 'loading' : (status?.status ?? 'error')

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <StatusBadge status={svcStatus} size="sm" />
        {status?.api_url && (
          <span className="text-xs text-gray-600 font-mono truncate">{status.api_url}</span>
        )}
      </div>

      {status && status.status === 'online' && (
        <div className="grid grid-cols-2 gap-2">
          <InfoItem label="Cluster" value={status.cluster} />
          <InfoItem label="Account" value={status.account} />
          <InfoItem label="Active Jobs" value={String(status.active_jobs ?? 0)} />
        </div>
      )}

      {status?.error && (
        <p className="text-xs text-red-400 bg-red-950/50 border border-red-900 rounded px-2 py-1.5 font-mono">
          {status.error}
        </p>
      )}

      {status?.latest_job && (
        <div className="rounded-lg bg-gray-950 border border-gray-800 px-3 py-2">
          <p className="text-xs text-gray-500 mb-1">Latest active job</p>
          <pre className="text-xs text-gray-400 font-mono overflow-auto">
            {JSON.stringify(status.latest_job, null, 2)}
          </pre>
        </div>
      )}

      {/* Job state polling */}
      <div className="flex items-center gap-2">
        <input
          type="text"
          placeholder="Slurm Job ID"
          value={activeJobId}
          onChange={e => setActiveJobId(e.target.value)}
          className="flex-1 bg-gray-950 border border-gray-800 rounded px-2 py-1.5 text-xs text-gray-300 font-mono placeholder-gray-700 focus:outline-none focus:border-violet-700"
        />
        <button
          onClick={startPolling}
          disabled={!activeJobId}
          className="px-3 py-1.5 text-xs rounded bg-violet-900 text-violet-200 hover:bg-violet-800 disabled:opacity-40 transition-colors"
        >
          Poll Status
        </button>
      </div>

      {showJob && <LogViewer lines={stateEvents} title={`Orion job ${activeJobId} — status events`} />}
    </div>
  )
}

function InfoItem({ label, value }: { label: string; value?: string }) {
  return (
    <div className="rounded bg-gray-950 border border-gray-800 px-2.5 py-2">
      <p className="text-xs text-gray-600">{label}</p>
      <p className="text-xs text-gray-300 font-mono truncate">{value || '—'}</p>
    </div>
  )
}
