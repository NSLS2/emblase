import React, { useEffect, useState } from 'react'
import { FlaskConical, ExternalLink } from 'lucide-react'
import { fetchJSON } from '../../api'
import type { MLflowStatus, MLflowModel, AppConfig } from '../../types'
import { StatusBadge } from '../StatusBadge'
import { Card, CardHeader, CardBody } from '../Card'

interface ModelsSectionProps {
  config: AppConfig | null
  refreshTick: number        // auto-refresh — status only
  manualRefreshTick: number  // manual refresh button — also reloads model list
  onModelsLoaded?: (models: MLflowModel[]) => void
}

export function ModelsSection({ config, refreshTick, manualRefreshTick, onModelsLoaded }: ModelsSectionProps) {
  const [status, setStatus] = useState<MLflowStatus | null>(null)
  const [models, setModels] = useState<MLflowModel[]>([])
  const [loadingStatus, setLoadingStatus] = useState(true)
  const [loadingModels, setLoadingModels] = useState(true)
  const [resolvedStatus, setResolvedStatus] = useState<MLflowStatus['status']>('loading')

  // Status refreshes on every tick (lightweight — one REST call)
  useEffect(() => {
    setLoadingStatus(true)
    fetchJSON<MLflowStatus>('/mlflow/status')
      .then(s => { setStatus(s); setResolvedStatus(s.status) })
      .catch(() => { setStatus({ status: 'error', error: 'Could not check MLflow' }); setResolvedStatus('error') })
      .finally(() => setLoadingStatus(false))
  }, [refreshTick])

  // Model list only refreshes on mount and on manual refresh
  useEffect(() => {
    setLoadingModels(true)
    fetchJSON<{ models: MLflowModel[] }>('/mlflow/models')
      .then(m => { setModels(m.models); onModelsLoaded?.(m.models) })
      .catch(() => setModels([]))
      .finally(() => setLoadingModels(false))
  }, [manualRefreshTick])

  // Only show "Checking…" on the very first load — never flicker back once resolved
  const svcStatus = loadingStatus && resolvedStatus === 'loading' ? 'loading' : resolvedStatus

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-pink-500/10 border border-pink-500/30">
            <FlaskConical className="h-4 w-4 text-pink-500" />
          </div>
          <div>
            <h3 className="font-semibold text-primary">Models</h3>
            <p className="text-xs text-secondary">MLflow model registry</p>
          </div>
        </div>
        <StatusBadge status={svcStatus} size="sm" />
      </CardHeader>
      <CardBody>
        <div className="space-y-3">
          {status?.tracking_uri && (
            <div className="flex items-center justify-between">
              <span className="text-xs text-secondary">Tracking URI</span>
              <a href={status.tracking_uri} target="_blank" rel="noopener noreferrer"
                className="flex items-center gap-1 text-xs text-pink-500 hover:text-pink-400 font-mono truncate max-w-xs">
                {status.tracking_uri.replace(/^https?:\/\//, '')}
                <ExternalLink className="h-3 w-3 flex-shrink-0" />
              </a>
            </div>
          )}
          {status?.error && (
            <p className="text-xs text-red-500 bg-red-500/10 border border-red-500/20 rounded px-2 py-1.5 font-mono">
              {status.error}
            </p>
          )}
          {status?.message && <p className="text-xs text-secondary">{status.message}</p>}

          {/* Model cards — always visible, scrollable */}
          <div className="space-y-1.5 max-h-52 overflow-y-auto scrollbar-thin">
            {loadingModels && (
              <p className="text-xs text-muted italic px-1">Loading models…</p>
            )}
            {!loadingModels && models.length === 0 && (
              <p className="text-xs text-muted italic px-1">
                No models found with prefix "{config?.mlflow_model_prefix}"
              </p>
            )}
            {models.map(m => (
              <div key={m.name}
                className="flex items-start justify-between rounded bg-page border border-theme px-3 py-2">
                <div className="min-w-0">
                  <p className="text-xs text-primary font-mono truncate">{m.name}</p>
                  {m.description && <p className="text-xs text-muted truncate">{m.description}</p>}
                </div>
                <span className="ml-2 flex-shrink-0 rounded-full bg-pink-500/10 border border-pink-500/30 px-1.5 py-0.5 text-xs text-pink-500 font-mono">
                  v{m.latest_version ?? '?'}
                </span>
              </div>
            ))}
          </div>
        </div>
      </CardBody>
    </Card>
  )
}
