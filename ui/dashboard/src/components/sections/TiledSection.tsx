import React, { useEffect, useState } from 'react'
import { Database, ExternalLink, Folder } from 'lucide-react'
import { fetchJSON } from '../../api'
import type { TiledStatus, AppConfig } from '../../types'
import { StatusBadge } from '../StatusBadge'
import { Card, CardHeader, CardBody } from '../Card'

interface TiledSectionProps {
  config: AppConfig | null
  refreshTick: number
}

export function TiledSection({ config, refreshTick }: TiledSectionProps) {
  const [status, setStatus] = useState<TiledStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [resolvedStatus, setResolvedStatus] = useState<TiledStatus['status']>('loading')

  const fetchStatus = async () => {
    setLoading(true)
    try {
      const data = await fetchJSON<TiledStatus>('/tiled/status')
      setStatus(data)
      setResolvedStatus(data.status)
    } catch {
      setStatus({ status: 'error', error: 'Failed to reach dashboard API' })
      setResolvedStatus('error')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchStatus() }, [refreshTick])

  // Only show "Checking…" on the very first load — never flicker back once resolved
  const serviceStatus = loading && resolvedStatus === 'loading' ? 'loading' : resolvedStatus

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-500/10 border border-blue-500/30">
            <Database className="h-4 w-4 text-blue-500" />
          </div>
          <div>
            <h3 className="font-semibold text-primary">Tiled Data Service</h3>
            <p className="text-xs text-secondary">Scientific data catalog</p>
          </div>
        </div>
        <StatusBadge status={serviceStatus} size="sm" />
      </CardHeader>
      <CardBody>
        {status && (
          <div className="space-y-3">
            {status.server_uri && (
              <div className="flex items-center justify-between">
                <span className="text-xs text-gray-500">Server URI</span>
                <a
                  href={status.server_uri}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-1 text-xs text-blue-500 hover:text-blue-400 font-mono truncate max-w-xs"
                >
                  {status.server_uri}
                  <ExternalLink className="h-3 w-3 flex-shrink-0" />
                </a>
              </div>
            )}
            {status.tiled_version && (
              <div className="flex items-center justify-between">
                <span className="text-xs text-secondary">Tiled Version</span>
                <span className="text-xs text-primary font-mono">{status.tiled_version}</span>
              </div>
            )}
            {status.python_version && (
              <div className="flex items-center justify-between">
                <span className="text-xs text-secondary">Python</span>
                <span className="text-xs text-primary font-mono">{status.python_version}</span>
              </div>
            )}
            {status.api_version !== undefined && status.api_version !== 0 && (
              <div className="flex items-center justify-between">
                <span className="text-xs text-secondary">API Version</span>
                <span className="text-xs text-primary font-mono">{status.api_version}</span>
              </div>
            )}
            {status.error && (
              <p className="text-xs text-red-500 bg-red-500/10 border border-red-500/20 rounded px-2 py-1.5 font-mono">
                {status.error}
              </p>
            )}
            {status.message && <p className="text-xs text-secondary">{status.message}</p>}

            {/* Container paths */}
            <div className="pt-1 space-y-2">
              <ContainerPath label="Input Container" path={config?.tiled_input_container} />
              <ContainerPath label="Output Container" path={config?.tiled_output_container} />
            </div>
          </div>
        )}
        {loading && <p className="text-xs text-muted">Checking Tiled…</p>}
      </CardBody>
    </Card>
  )
}

function ContainerPath({ label, path }: { label: string; path?: string }) {
  return (
    <div className="flex items-start gap-2 rounded-lg bg-page border border-theme px-3 py-2">
      <Folder className="h-3.5 w-3.5 text-muted mt-0.5 flex-shrink-0" />
      <div className="min-w-0">
        <p className="text-xs text-secondary">{label}</p>
        <p className="text-xs text-primary font-mono truncate">
          {path || <span className="text-muted italic">not configured</span>}
        </p>
      </div>
    </div>
  )
}
