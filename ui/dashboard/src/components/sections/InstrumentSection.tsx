import React, { useEffect, useState, useRef, useCallback } from 'react'
import { Microscope, Play, Square, AlertCircle, Loader2 } from 'lucide-react'
import { fetchJSON, postJSON } from '../../api'
import type { InstrumentStatus, AppConfig } from '../../types'
import { Card, CardHeader, CardBody } from '../Card'

interface InstrumentSectionProps {
  config: AppConfig | null
  refreshTick: number
}

export function InstrumentSection({ config, refreshTick }: InstrumentSectionProps) {
  const [status, setStatus] = useState<InstrumentStatus | null>(null)
  const [loading, setLoading] = useState(true)

  // Configurable fields
  const [src, setSrc] = useState('/smi/sandbox/confab26_demo/sources/run_1086139')
  const [imageKey, setImageKey] = useState('primary/pil900KW_image')
  const [batchDelay, setBatchDelay] = useState(0.1)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const fetchStatus = useCallback(async () => {
    try {
      const s = await fetchJSON<InstrumentStatus>('/instrument/status')
      setStatus(s)
      setLoading(false)
      return s
    } catch {
      setLoading(false)
      return null
    }
  }, [])

  useEffect(() => { fetchStatus() }, [refreshTick])

  // Fast-poll while running
  useEffect(() => {
    if (pollRef.current) clearInterval(pollRef.current)
    if (status?.status === 'running') {
      pollRef.current = setInterval(fetchStatus, 2000)
    }
    return () => { if (pollRef.current) clearInterval(pollRef.current) }
  }, [status?.status])

  const handleStart = async () => {
    if (!src.trim()) { setError('Source path is required'); return }
    setSubmitting(true)
    setError(null)
    try {
      await postJSON('/instrument/start', { src: src.trim(), image_key: imageKey, batch_delay: batchDelay })
      await fetchStatus()
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setSubmitting(false)
    }
  }

  const handleStop = async () => {
    try {
      await postJSON('/instrument/stop', {})
      await fetchStatus()
    } catch {}
  }

  const isRunning = status?.status === 'running'
  const isDone    = status?.status === 'done'
  const isError   = status?.status === 'error'

  const pct = status && status.frames_total > 0
    ? Math.round(status.frames_written / status.frames_total * 100)
    : 0

  const rate = status && status.elapsed > 0 && status.frames_written > 0
    ? (status.frames_written / status.elapsed).toFixed(1)
    : null

  const eta = status && rate && status.frames_total > status.frames_written
    ? Math.round((status.frames_total - status.frames_written) / parseFloat(rate))
    : null

  // Only show "loading" badge on very first load
  const badgeCfg = (loading && status === null)
    ? { dot: 'bg-slate-400 animate-pulse', cls: 'bg-slate-100 dark:bg-slate-900 border-slate-300 dark:border-slate-700 text-slate-500 dark:text-slate-400', label: 'Checking…' }
    : isRunning
    ? { dot: 'bg-blue-400 animate-pulse',    cls: 'bg-blue-50  dark:bg-blue-950  border-blue-300  dark:border-blue-800  text-blue-700  dark:text-blue-400',   label: 'Running'  }
    : isError
    ? { dot: 'bg-red-500',                   cls: 'bg-red-50   dark:bg-red-950   border-red-300   dark:border-red-800   text-red-700   dark:text-red-400',    label: 'Error'    }
    : isDone
    ? { dot: 'bg-emerald-400',               cls: 'bg-emerald-50 dark:bg-emerald-950 border-emerald-300 dark:border-emerald-800 text-emerald-700 dark:text-emerald-400', label: 'Done' }
    : { dot: 'bg-slate-400',                 cls: 'bg-slate-100 dark:bg-slate-900 border-slate-300 dark:border-slate-700 text-slate-600 dark:text-slate-400', label: 'Idle'     }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-500/10 border border-blue-500/30">
            <Microscope className="h-4 w-4 text-blue-500" />
          </div>
          <div>
            <h3 className="font-semibold text-primary">Instrument</h3>
            <p className="text-xs text-secondary">Replay data to simulate acquisition</p>
          </div>
        </div>
        <span className={`inline-flex items-center gap-1.5 rounded-full border font-medium px-2 py-0.5 text-xs ${badgeCfg.cls}`}>
            <span className={`inline-block h-2 w-2 rounded-full ${badgeCfg.dot}`} />
            {badgeCfg.label}
          </span>
      </CardHeader>
      <CardBody>
        <div className="space-y-3">

          {/* Source path */}
          <div>
            <label className="text-xs text-secondary block mb-1">Source run path</label>
            <input
              type="text"
              value={src}
              onChange={e => setSrc(e.target.value)}
              disabled={isRunning}
              placeholder="smi/sandbox/demo/inputs/run_1086139"
              className="w-full input-base placeholder:text-muted disabled:opacity-50"
            />
          </div>

          {/* Image key + delay */}
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="text-xs text-secondary block mb-1">Image key</label>
              <input
                type="text"
                value={imageKey}
                onChange={e => setImageKey(e.target.value)}
                disabled={isRunning}
                className="w-full input-base disabled:opacity-50"
              />
            </div>
            <div>
              <label className="text-xs text-secondary block mb-1">Frame delay (s)</label>
              <input
                type="number"
                min={0} max={60} step={0.05}
                value={batchDelay}
                onChange={e => setBatchDelay(Number(e.target.value))}
                disabled={isRunning}
                className="w-full input-base disabled:opacity-50"
              />
            </div>
          </div>

          {/* Destination (read-only) */}
          <div className="text-xs text-muted">
            Destination:{' '}
            <code className="font-mono">{config?.tiled_input_container || 'EMBLASE_TILED_INPUT_CONTAINER'}</code>
          </div>

          {/* Progress bar */}
          {isRunning && status && (
            <div className="space-y-1">
              <div className="flex items-center justify-between text-xs text-secondary">
                <span>{status.frames_written} / {status.frames_total || '?'} frames</span>
                <span className="text-muted">
                  {rate && <>{rate} fr/s{eta != null ? ` · ETA ${eta}s` : ''}</>}
                </span>
              </div>
              <div className="h-1.5 rounded-full bg-page border border-theme overflow-hidden">
                <div
                  className="h-full bg-blue-500 transition-all duration-500"
                  style={{ width: `${pct}%` }}
                />
              </div>
              <p className="text-xs text-muted font-mono truncate">→ {status.rename || '…'}</p>
            </div>
          )}

          {isDone && status && (
            <p className="text-xs text-emerald-500">
              Done in {status.elapsed}s — {status.frames_written} frames copied to{' '}
              <code className="font-mono">{status.rename}</code>
            </p>
          )}

          {isError && status?.error && (
            <div className="flex items-start gap-2 rounded bg-red-500/10 border border-red-500/20 px-2 py-1.5">
              <AlertCircle className="h-3.5 w-3.5 text-red-500 mt-0.5 flex-shrink-0" />
              <p className="text-xs text-red-500 font-mono">{status.error}</p>
            </div>
          )}

          {error && (
            <p className="text-xs text-red-500 font-mono bg-red-500/10 border border-red-500/20 rounded px-2 py-1.5">
              {error}
            </p>
          )}

          {/* Action button */}
          {isRunning ? (
            <button onClick={handleStop}
              className="w-full flex items-center justify-center gap-2 rounded-lg bg-red-600 hover:bg-red-500 text-white font-medium py-2 text-xs transition-colors">
              <Square className="h-3.5 w-3.5" /> Stop Replay
            </button>
          ) : (
            <button onClick={handleStart} disabled={submitting || !src.trim()}
              className="w-full flex items-center justify-center gap-2 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:bg-blue-900 disabled:text-blue-400 text-white font-medium py-2 text-xs transition-colors">
              {submitting
                ? <><Loader2 className="h-3.5 w-3.5 animate-spin" /> Starting…</>
                : <><Play className="h-3.5 w-3.5" /> Start Replay</>}
            </button>
          )}
        </div>
      </CardBody>
    </Card>
  )
}
