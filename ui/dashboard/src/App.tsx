import React, { useEffect, useState, useCallback } from 'react'
import { RefreshCw, Play, Activity, Github, ExternalLink } from 'lucide-react'
import { fetchJSON } from './api'
import type { AppConfig, MLflowModel } from './types'
import { TiledSection } from './components/sections/TiledSection'
import { ComputeSection } from './components/sections/ComputeSection'
import { ModelsSection } from './components/sections/ModelsSection'
import { ChatbotSection } from './components/sections/ChatbotSection'
import { OpenMetadataSection } from './components/sections/OpenMetadataSection'
import { JobSubmitPanel } from './components/panels/JobSubmitPanel'

const REFRESH_INTERVAL_MS = 30_000

export default function App() {
  const [config, setConfig] = useState<AppConfig | null>(null)
  const [refreshTick, setRefreshTick] = useState(0)
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date())
  const [showJobPanel, setShowJobPanel] = useState(false)
  const [models, setModels] = useState<MLflowModel[]>([])

  useEffect(() => {
    fetchJSON<AppConfig>('/config').then(setConfig).catch(console.error)
  }, [])

  // Auto-refresh every 30s
  useEffect(() => {
    const id = setInterval(() => {
      setRefreshTick(t => t + 1)
      setLastRefresh(new Date())
    }, REFRESH_INTERVAL_MS)
    return () => clearInterval(id)
  }, [])

  const handleRefresh = useCallback(() => {
    setRefreshTick(t => t + 1)
    setLastRefresh(new Date())
  }, [])

  const timeString = lastRefresh.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })

  return (
    <div className="min-h-screen bg-gray-950">
      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header className="sticky top-0 z-40 border-b border-gray-800 bg-gray-950/95 backdrop-blur-sm">
        <div className="mx-auto max-w-7xl px-4 sm:px-6">
          <div className="flex h-16 items-center justify-between gap-4">
            {/* Logo + title */}
            <div className="flex items-center gap-4 min-w-0">
              <div className="flex items-center gap-2 flex-shrink-0">
                {/* BNL logo */}
                <img
                  src="/logos/bnl-logo.png"
                  alt="Brookhaven National Laboratory"
                  className="h-8 object-contain"
                  onError={e => { (e.target as HTMLImageElement).style.display = 'none' }}
                />
                {/* Separator */}
                <div className="h-6 w-px bg-gray-800" />
                {/* Tiled logo */}
                <img
                  src="/logos/tiled-logo.svg"
                  alt="Tiled"
                  className="h-6 object-contain"
                  onError={e => { (e.target as HTMLImageElement).style.display = 'none' }}
                />
              </div>
              <div className="min-w-0">
                <h1 className="text-sm font-bold text-gray-100 tracking-tight leading-tight">
                  EMBLASE Dashboard
                </h1>
                <p className="text-xs text-gray-500 leading-tight hidden sm:block truncate">
                  EMBeddings &amp; LAtent Space Explorer — BNL NSLS-II × AmSC
                </p>
              </div>
            </div>

            {/* Genesis / AmSC logo (right side) */}
            <div className="hidden lg:flex items-center gap-3 flex-shrink-0">
              <img
                src="/logos/genesis-logo.png"
                alt="Genesis Mission"
                className="h-8 object-contain opacity-80"
                onError={e => { (e.target as HTMLImageElement).style.display = 'none' }}
              />
            </div>

            {/* Actions */}
            <div className="flex items-center gap-2 flex-shrink-0">
              <div className="hidden sm:flex items-center gap-1.5 text-xs text-gray-600">
                <Activity className="h-3 w-3" />
                <span>{timeString}</span>
              </div>
              <button
                onClick={handleRefresh}
                className="flex items-center gap-1.5 rounded-lg border border-gray-800 bg-gray-900 px-3 py-1.5 text-xs text-gray-400 hover:text-gray-200 hover:border-gray-700 transition-colors"
              >
                <RefreshCw className="h-3.5 w-3.5" />
                <span className="hidden sm:inline">Refresh</span>
              </button>
              <button
                onClick={() => setShowJobPanel(true)}
                className="flex items-center gap-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 px-3 py-1.5 text-xs text-white font-medium transition-colors"
              >
                <Play className="h-3.5 w-3.5" />
                <span>Submit Job</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* ── Science banner ─────────────────────────────────────────────────── */}
      <div className="border-b border-gray-900 bg-gradient-to-r from-bnl-blue/10 via-gray-900 to-amsc-blue/10">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 py-3">
          <p className="text-xs text-gray-500 leading-relaxed text-center">
            <span className="text-gray-300 font-medium">Mission:</span>{' '}
            AI/ML latent space exploration of synchrotron X-ray scattering data,
            combining{' '}
            <span className="text-blue-400">Tiled</span> scientific data services,{' '}
            <span className="text-violet-400">NERSC Perlmutter</span> GPU supercomputing,{' '}
            <span className="text-pink-400">MLflow</span> model registry, and{' '}
            <span className="text-teal-400">AmSC AI</span> assistants —
            all orchestrated through the <span className="text-amber-400">American Science Cloud</span>.
          </p>
        </div>
      </div>

      {/* ── Main content ───────────────────────────────────────────────────── */}
      <main className="mx-auto max-w-7xl px-4 sm:px-6 py-6 space-y-6">

        {/* Row 1: Tiled + Compute */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          <TiledSection config={config} refreshTick={refreshTick} />
          <ComputeSection config={config} refreshTick={refreshTick} />
        </div>

        {/* Row 2: Models + Chatbot + OpenMetadata */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          <ModelsSection
            config={config}
            refreshTick={refreshTick}
            onModelsLoaded={setModels}
          />
          <ChatbotSection config={config} refreshTick={refreshTick} />
          <OpenMetadataSection refreshTick={refreshTick} />
        </div>

        {/* Pipeline diagram card */}
        <PipelineCard />

      </main>

      {/* ── Footer ─────────────────────────────────────────────────────────── */}
      <footer className="border-t border-gray-900 mt-6">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 py-4 flex items-center justify-between gap-4 text-xs text-gray-700">
          <span>EMBLASE — BNL NSLS-II × American Science Cloud</span>
          <div className="flex items-center gap-4">
            {config?.tiled_server_uri && (
              <a href={config.tiled_server_uri} target="_blank" rel="noopener noreferrer" className="flex items-center gap-1 hover:text-gray-500 transition-colors">
                Tiled <ExternalLink className="h-3 w-3" />
              </a>
            )}
            <a href="/api/docs" target="_blank" rel="noopener noreferrer" className="flex items-center gap-1 hover:text-gray-500 transition-colors">
              API Docs <ExternalLink className="h-3 w-3" />
            </a>
          </div>
        </div>
      </footer>

      {/* ── Job Submit Panel ───────────────────────────────────────────────── */}
      {showJobPanel && (
        <JobSubmitPanel
          config={config}
          models={models}
          onClose={() => setShowJobPanel(false)}
        />
      )}
    </div>
  )
}

// ── Pipeline overview card ────────────────────────────────────────────────────

function PipelineCard() {
  const steps = [
    {
      icon: '🔬',
      label: 'Synchrotron',
      sublabel: 'NSLS-II beamline',
      color: 'border-yellow-800 bg-yellow-950/40',
      textColor: 'text-yellow-300',
    },
    {
      icon: '🗄️',
      label: 'Tiled',
      sublabel: 'Data ingestion',
      color: 'border-blue-800 bg-blue-950/40',
      textColor: 'text-blue-300',
    },
    {
      icon: '⚡',
      label: 'NERSC / Orion',
      sublabel: 'GPU inference',
      color: 'border-violet-800 bg-violet-950/40',
      textColor: 'text-violet-300',
    },
    {
      icon: '🧠',
      label: 'MLflow',
      sublabel: 'Model registry',
      color: 'border-pink-800 bg-pink-950/40',
      textColor: 'text-pink-300',
    },
    {
      icon: '🗺️',
      label: 'Latent Space',
      sublabel: 'Tiled embeddings',
      color: 'border-emerald-800 bg-emerald-950/40',
      textColor: 'text-emerald-300',
    },
    {
      icon: '🤖',
      label: 'AI Assistant',
      sublabel: 'AmSC LLM',
      color: 'border-teal-800 bg-teal-950/40',
      textColor: 'text-teal-300',
    },
  ]

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900 shadow-xl px-5 py-4">
      <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-4">
        AmSC Science Pipeline — End-to-End Data Flow
      </h3>
      <div className="flex items-center gap-2 overflow-x-auto scrollbar-thin pb-2">
        {steps.map((step, i) => (
          <React.Fragment key={step.label}>
            <div className={`flex-shrink-0 rounded-xl border px-4 py-3 text-center min-w-[110px] ${step.color}`}>
              <div className="text-2xl mb-1">{step.icon}</div>
              <p className={`text-xs font-semibold ${step.textColor}`}>{step.label}</p>
              <p className="text-xs text-gray-600">{step.sublabel}</p>
            </div>
            {i < steps.length - 1 && (
              <div className="flex-shrink-0 text-gray-700 text-lg">→</div>
            )}
          </React.Fragment>
        ))}
      </div>
    </div>
  )
}
