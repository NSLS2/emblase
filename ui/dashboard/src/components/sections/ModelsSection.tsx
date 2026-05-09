import React, { useEffect, useState } from 'react'
import { FlaskConical, ExternalLink, ChevronDown, ChevronUp } from 'lucide-react'
import { fetchJSON } from '../../api'
import type { MLflowStatus, MLflowModel, AppConfig } from '../../types'
import { StatusBadge } from '../StatusBadge'
import { Card, CardHeader, CardBody } from '../Card'

interface ModelsSectionProps {
  config: AppConfig | null
  refreshTick: number
  onModelsLoaded?: (models: MLflowModel[]) => void
}

export function ModelsSection({ config, refreshTick, onModelsLoaded }: ModelsSectionProps) {
  const [status, setStatus] = useState<MLflowStatus | null>(null)
  const [models, setModels] = useState<MLflowModel[]>([])
  const [loadingStatus, setLoadingStatus] = useState(true)
  const [loadingModels, setLoadingModels] = useState(true)
  const [expanded, setExpanded] = useState(false)

  const fetchAll = async () => {
    setLoadingStatus(true)
    setLoadingModels(true)
    try {
      const s = await fetchJSON<MLflowStatus>('/mlflow/status')
      setStatus(s)
    } catch {
      setStatus({ status: 'error', error: 'Could not check MLflow' })
    } finally {
      setLoadingStatus(false)
    }
    try {
      const m = await fetchJSON<{ models: MLflowModel[] }>('/mlflow/models')
      setModels(m.models)
      onModelsLoaded?.(m.models)
    } catch {
      setModels([])
    } finally {
      setLoadingModels(false)
    }
  }

  useEffect(() => { fetchAll() }, [refreshTick])

  const svcStatus = loadingStatus ? 'loading' : (status?.status ?? 'error')

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-pink-950 border border-pink-800">
            <FlaskConical className="h-4 w-4 text-pink-400" />
          </div>
          <div>
            <h3 className="font-semibold text-gray-100">Models</h3>
            <p className="text-xs text-gray-500">MLflow model registry</p>
          </div>
        </div>
        <StatusBadge status={svcStatus} size="sm" />
      </CardHeader>
      <CardBody>
        <div className="space-y-3">
          {status?.tracking_uri && (
            <div className="flex items-center justify-between">
              <span className="text-xs text-gray-500">Tracking URI</span>
              <a
                href={status.tracking_uri}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1 text-xs text-pink-400 hover:text-pink-300 font-mono truncate max-w-xs"
              >
                {status.tracking_uri.replace(/^https?:\/\//, '')}
                <ExternalLink className="h-3 w-3 flex-shrink-0" />
              </a>
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

          {/* Model list */}
          <div>
            <button
              onClick={() => setExpanded(!expanded)}
              className="flex w-full items-center justify-between text-xs text-gray-400 hover:text-gray-200 transition-colors"
            >
              <span>
                {loadingModels ? 'Loading models…' : `${models.length} model${models.length !== 1 ? 's' : ''} (${config?.mlflow_model_prefix || 'bnl-nsls2-'}*)`}
              </span>
              {expanded ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
            </button>

            {expanded && (
              <div className="mt-2 space-y-1.5 max-h-60 overflow-y-auto scrollbar-thin">
                {models.length === 0 && !loadingModels && (
                  <p className="text-xs text-gray-600 italic px-1">No models found with prefix "{config?.mlflow_model_prefix}"</p>
                )}
                {models.map(m => (
                  <div key={m.name} className="flex items-start justify-between rounded bg-gray-950 border border-gray-800 px-3 py-2">
                    <div className="min-w-0">
                      <p className="text-xs text-gray-200 font-mono truncate">{m.name}</p>
                      {m.description && (
                        <p className="text-xs text-gray-600 truncate">{m.description}</p>
                      )}
                    </div>
                    <span className="ml-2 flex-shrink-0 rounded-full bg-pink-950 border border-pink-800 px-1.5 py-0.5 text-xs text-pink-300 font-mono">
                      v{m.latest_version ?? '?'}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </CardBody>
    </Card>
  )
}
