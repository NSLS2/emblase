import React, { useEffect, useState, useCallback, useRef } from 'react'
import { RefreshCw, Activity, ExternalLink, Sun, Moon } from 'lucide-react'
import { fetchJSON } from './api'
import type { AppConfig, MLflowModel, JobRecord, WatcherStatus } from './types'
import { TiledSection } from './components/sections/TiledSection'
import { ComputeSection } from './components/sections/ComputeSection'
import { InstrumentSection } from './components/sections/InstrumentSection'
import { ModelsSection } from './components/sections/ModelsSection'
import { ChatbotSection } from './components/sections/ChatbotSection'
import { OpenMetadataSection } from './components/sections/OpenMetadataSection'
import { JobSubmitPanel } from './components/panels/JobSubmitPanel'
import { WatcherPanel } from './components/panels/WatcherPanel'
import { useTheme } from './context/ThemeContext'

const REFRESH_INTERVAL_MS = 30_000
const WATCHER_POLL_MS = 5_000
const JOB_HISTORY_KEY = 'emblase-job-history'

type Backend = 'nersc' | 'orion'

// ── Job history persistence ───────────────────────────────────────────────────

function loadJobHistory(): JobRecord[] {
  try {
    const raw = localStorage.getItem(JOB_HISTORY_KEY)
    if (raw) return JSON.parse(raw) as JobRecord[]
  } catch {}
  return []
}

function saveJobHistory(jobs: JobRecord[]) {
  try { localStorage.setItem(JOB_HISTORY_KEY, JSON.stringify(jobs)) } catch {}
}

// ── App ───────────────────────────────────────────────────────────────────────

export default function App() {
  const { theme, toggle: toggleTheme } = useTheme()
  const [config, setConfig] = useState<AppConfig | null>(null)
  const [refreshTick, setRefreshTick] = useState(0)
  const [manualRefreshTick, setManualRefreshTick] = useState(0)
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date())
  const [models, setModels] = useState<MLflowModel[]>([])

  // Job history — persisted to localStorage; also merged with watcher-submitted jobs
  const [jobHistory, setJobHistory] = useState<JobRecord[]>(() => loadJobHistory())

  // Panel state
  const [jobPanelBackend, setJobPanelBackend] = useState<Backend | null>(null)
  const [watcherPanelBackend, setWatcherPanelBackend] = useState<Backend | null>(null)

  // Watcher status — polled independently
  const [watcherStatus, setWatcherStatus] = useState<WatcherStatus | null>(null)
  const watcherPollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    fetchJSON<AppConfig>('/config').then(setConfig).catch(console.error)
  }, [])

  // Global auto-refresh tick
  useEffect(() => {
    const id = setInterval(() => {
      setRefreshTick(t => t + 1)
      setLastRefresh(new Date())
    }, REFRESH_INTERVAL_MS)
    return () => clearInterval(id)
  }, [])

  // ── Watcher status + jobs poll ────────────────────────────────────────────
  // Polls /watcher/status and /watcher/jobs on the same interval.
  // Any new job_ids found in /watcher/jobs are merged into jobHistory.
  const knownWatcherJobIds = useRef<Set<string>>(
    new Set(loadJobHistory().filter(j => j.mode === 'stream').map(j => j.job_id))
  )

  const pollWatcher = useCallback(() => {
    fetchJSON<WatcherStatus>('/watcher/status').then(setWatcherStatus).catch(() => {})

    fetchJSON<{ jobs: JobRecord[] }>('/watcher/jobs').then(({ jobs }) => {
      const newJobs = jobs.filter(j => !knownWatcherJobIds.current.has(j.job_id))
      if (newJobs.length === 0) return
      newJobs.forEach(j => knownWatcherJobIds.current.add(j.job_id))
      setJobHistory(prev => {
        const merged = [...newJobs, ...prev]
        saveJobHistory(merged)
        return merged
      })
    }).catch(() => {})
  }, [])

  useEffect(() => {
    pollWatcher()
    watcherPollRef.current = setInterval(pollWatcher, WATCHER_POLL_MS)
    return () => { if (watcherPollRef.current) clearInterval(watcherPollRef.current) }
  }, [pollWatcher])

  const handleRefresh = useCallback(() => {
    setRefreshTick(t => t + 1)
    setManualRefreshTick(t => t + 1)
    setLastRefresh(new Date())
  }, [])

  const handleJobSubmitted = useCallback((job: JobRecord) => {
    setJobHistory(prev => {
      const next = [job, ...prev]
      saveJobHistory(next)
      return next
    })
    setJobPanelBackend(null)
  }, [])

  const handleWatcherStarted = useCallback(() => {
    fetchJSON<WatcherStatus>('/watcher/status').then(setWatcherStatus).catch(() => {})
    setWatcherPanelBackend(null)
  }, [])

  const timeString = lastRefresh.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })

  const bnlLogo = theme === 'dark' ? '/logos/bnl-logo-dark.png'     : '/logos/bnl-logo-light.png'
  const genLogo = theme === 'dark' ? '/logos/genesis-logo-dark.png' : '/logos/genesis-logo-light.png'

  return (
    <div className="min-h-screen bg-page text-primary">
      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header className="sticky top-0 z-40 border-b border-theme bg-card/95 backdrop-blur-sm">
        <div className="mx-auto max-w-7xl px-4 sm:px-6">
          <div className="flex h-16 items-center justify-between gap-4">
            <div className="flex items-center gap-4 min-w-0">
              <div className="flex items-center gap-2 flex-shrink-0">
                <img src={bnlLogo} alt="BNL" className="h-8 object-contain"
                  onError={e => { (e.target as HTMLImageElement).style.display = 'none' }} />
                <div className="h-6 w-px bg-theme" />
                <img src="/logos/tiled-logo.svg" alt="Tiled" className="h-6 object-contain"
                  onError={e => { (e.target as HTMLImageElement).style.display = 'none' }} />
              </div>
              <div className="min-w-0">
                <h1 className="text-sm font-bold text-primary tracking-tight leading-tight">EMBLASE Dashboard</h1>
                <p className="text-xs text-secondary leading-tight hidden sm:block truncate">
                  EMBeddings &amp; LAtent Space Explorer — BNL NSLS-II × AmSC
                </p>
              </div>
            </div>

            <div className="hidden lg:flex items-center gap-3 flex-shrink-0">
              <img src={genLogo} alt="Genesis" className="h-8 object-contain opacity-80"
                onError={e => { (e.target as HTMLImageElement).style.display = 'none' }} />
            </div>

            <div className="flex items-center gap-2 flex-shrink-0">
              <div className="hidden sm:flex items-center gap-1.5 text-xs text-muted">
                <Activity className="h-3 w-3" />
                <span>{timeString}</span>
              </div>
              <button onClick={toggleTheme} title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
                className="rounded-lg border border-theme p-1.5 hover:bg-row-hover text-secondary hover:text-primary transition-colors">
                {theme === 'dark' ? <Sun className="h-3.5 w-3.5" /> : <Moon className="h-3.5 w-3.5" />}
              </button>
              <button onClick={handleRefresh}
                className="flex items-center gap-1.5 rounded-lg border border-theme bg-card px-3 py-1.5 text-xs text-secondary hover:text-primary hover:border-theme-hover transition-colors">
                <RefreshCw className="h-3.5 w-3.5" />
                <span className="hidden sm:inline">Refresh</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* ── Science banner ─────────────────────────────────────────────────── */}
      <div className="border-b border-theme bg-gradient-to-r from-blue-500/5 via-transparent to-violet-500/5">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 py-3">
          <p className="text-xs text-muted leading-relaxed text-center">
            <span className="text-primary font-medium">Mission:</span>{' '}
            AI/ML latent space exploration of synchrotron X-ray scattering data, combining{' '}
            <span className="text-blue-500">Tiled</span> scientific data services,{' '}
            <span className="text-violet-500">NERSC Perlmutter</span> GPU supercomputing,{' '}
            <span className="text-pink-500">MLflow</span> model registry, and{' '}
            <span className="text-teal-500">AmSC AI</span> assistants —
            all orchestrated through the <span className="text-amber-500">American Science Cloud</span>.
          </p>
        </div>
      </div>

      {/* ── Main content ───────────────────────────────────────────────────── */}
      <main className="mx-auto max-w-7xl px-4 sm:px-6 py-6 space-y-6">
        {/* Top row: Instrument · Tiled · Compute */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          <InstrumentSection config={config} refreshTick={refreshTick} />
          <TiledSection config={config} refreshTick={refreshTick} />
          <ComputeSection
            config={config}
            refreshTick={refreshTick}
            jobHistory={jobHistory}
            watcherStatus={watcherStatus}
            onSubmitJob={setJobPanelBackend}
            onLiveWatch={setWatcherPanelBackend}
          />
        </div>

        {/* Bottom row: Models · Chatbot · OpenMetadata */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          <ModelsSection
            config={config}
            refreshTick={refreshTick}
            manualRefreshTick={manualRefreshTick}
            onModelsLoaded={setModels}
          />
          <ChatbotSection config={config} refreshTick={refreshTick} />
          <OpenMetadataSection refreshTick={refreshTick} />
        </div>

        <PipelineCard />
      </main>

      {/* ── Footer ─────────────────────────────────────────────────────────── */}
      <footer className="border-t border-theme mt-6">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 py-4 flex items-center justify-between gap-4 text-xs text-muted">
          <span>EMBLASE — BNL NSLS-II × American Science Cloud</span>
          <div className="flex items-center gap-4">
            {config?.tiled_server_uri && (
              <a href={config.tiled_server_uri} target="_blank" rel="noopener noreferrer"
                className="flex items-center gap-1 hover:text-secondary transition-colors">
                Tiled <ExternalLink className="h-3 w-3" />
              </a>
            )}
            <a href="/api/docs" target="_blank" rel="noopener noreferrer"
              className="flex items-center gap-1 hover:text-secondary transition-colors">
              API Docs <ExternalLink className="h-3 w-3" />
            </a>
          </div>
        </div>
      </footer>

      {/* ── Modals ─────────────────────────────────────────────────────────── */}
      {jobPanelBackend && (
        <JobSubmitPanel
          config={config}
          models={models}
          backend={jobPanelBackend}
          onClose={() => setJobPanelBackend(null)}
          onJobSubmitted={handleJobSubmitted}
        />
      )}

      {watcherPanelBackend && (
        <WatcherPanel
          config={config}
          models={models}
          backend={watcherPanelBackend}
          onClose={() => setWatcherPanelBackend(null)}
          onStarted={handleWatcherStarted}
        />
      )}
    </div>
  )
}

// ── Pipeline overview card ────────────────────────────────────────────────────

function PNode({ icon, label, sublabel, color, text }: {
  icon: string; label: string; sublabel: string; color: string; text: string
}) {
  return (
    <div className={`flex-shrink-0 rounded-xl border px-4 py-3 text-center min-w-[108px] ${color}`}>
      <div className="text-2xl mb-1">{icon}</div>
      <p className={`text-xs font-semibold leading-tight ${text}`}>{label}</p>
      <p className="text-[10px] text-muted leading-tight mt-0.5">{sublabel}</p>
    </div>
  )
}

function PipelineCard() {
  const steps = [
    { icon: '👤', label: 'Scientist',      sublabel: 'submits job',          color: 'border-gray-500/30 bg-gray-500/5',       text: 'text-secondary'   },
    { icon: '🔬', label: 'Synchrotron',   sublabel: 'NSLS-II beamline',    color: 'border-yellow-500/30 bg-yellow-500/5',   text: 'text-yellow-500'  },
    { icon: '🗄️', label: 'Tiled',         sublabel: 'Data ingestion',       color: 'border-blue-500/30 bg-blue-500/5',       text: 'text-blue-500'    },
    { icon: '⚡', label: 'NERSC / Orion', sublabel: 'GPU inference',         color: 'border-violet-500/30 bg-violet-500/5',   text: 'text-violet-500'  },
    { icon: '📦', label: 'MLflow',         sublabel: 'Model registry (R/O)', color: 'border-pink-500/30 bg-pink-500/5',       text: 'text-pink-500'    },
    { icon: '🗺️', label: 'Latent Space',  sublabel: 'Tiled embeddings',     color: 'border-emerald-500/30 bg-emerald-500/5', text: 'text-emerald-500' },
    { icon: '🗂️', label: 'OpenMetadata',  sublabel: 'Catalog & lineage',    color: 'border-orange-500/30 bg-orange-500/5',   text: 'text-orange-500'  },
    { icon: '🤖', label: 'AI Assistant',  sublabel: 'AmSC LLM (chat)',      color: 'border-teal-500/30 bg-teal-500/5',       text: 'text-teal-500'    },
  ]

  return (
    <div className="rounded-xl border border-theme bg-card shadow-xl px-5 py-4">
      <h3 className="text-xs font-semibold text-muted uppercase tracking-wide mb-4">
        AmSC Science Pipeline — End-to-End Data Flow
      </h3>
      <div className="flex items-center gap-2 overflow-x-auto scrollbar-thin pb-2">
        {steps.map((step, i) => (
          <React.Fragment key={step.label}>
            <PNode {...step} />
            {i < steps.length - 1 && (
              <div className="flex-shrink-0 text-muted text-lg">→</div>
            )}
          </React.Fragment>
        ))}
      </div>
      <p className="text-[10px] text-muted mt-2 italic">
        MLflow is read-only — models are pre-trained and loaded by compute jobs; no training happens here.
      </p>
    </div>
  )
}
