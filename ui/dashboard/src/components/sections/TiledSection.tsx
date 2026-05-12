import React, { useEffect, useState } from 'react'
import { Database, ExternalLink, Folder, Layers } from 'lucide-react'
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

  const serviceStatus = loading && resolvedStatus === 'loading' ? 'loading' : resolvedStatus

  // Build the Tiled UI browse URL for the output container
  const browseUrl = (() => {
    if (!status?.server_uri || !status?.output_container) return null
    const base = status.server_uri.replace(/\/api\/v1\/?$/, '').replace(/\/$/, '')
    const path = status.output_container.replace(/^\//, '')
    return `${base}/ui/browse/${path}`
  })()

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
            {/* Server URI */}
            {status.server_uri && (
              <div className="flex items-center justify-between">
                <span className="text-xs text-secondary">Server</span>
                <a href={status.server_uri} target="_blank" rel="noopener noreferrer"
                  className="flex items-center gap-1 text-xs text-blue-500 hover:text-blue-400 font-mono truncate max-w-[200px]">
                  {status.server_uri.replace(/^https?:\/\//, '')}
                  <ExternalLink className="h-3 w-3 flex-shrink-0" />
                </a>
              </div>
            )}

            {/* Version */}
            {status.tiled_version && (
              <div className="flex items-center justify-between">
                <span className="text-xs text-secondary">Tiled</span>
                <span className="text-xs text-primary font-mono">{status.tiled_version}</span>
              </div>
            )}

            {/* Error */}
            {status.error && (
              <p className="text-xs text-red-500 bg-red-500/10 border border-red-500/20 rounded px-2 py-1.5 font-mono">
                {status.error}
              </p>
            )}
            {status.message && <p className="text-xs text-secondary">{status.message}</p>}

            {/* Output container */}
            {status.output_container && (
              <div className="rounded-lg border border-blue-500/20 bg-blue-500/5 px-3 py-2.5 space-y-1.5">
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-1.5 min-w-0">
                    <Folder className="h-3.5 w-3.5 text-blue-400 flex-shrink-0" />
                    <span className="text-xs text-secondary font-medium">Output container</span>
                  </div>
                  {browseUrl && (
                    <a href={browseUrl} target="_blank" rel="noopener noreferrer"
                      title="Browse in Tiled UI"
                      className="flex items-center gap-1 text-xs text-blue-500 hover:text-blue-400 transition-colors flex-shrink-0">
                      Browse <ExternalLink className="h-3 w-3" />
                    </a>
                  )}
                </div>
                <p className="text-xs text-blue-300/60 dark:text-blue-300/40 font-mono truncate pl-5">
                  {status.output_container}
                </p>
                {status.output_count != null && (
                  <div className="flex items-center gap-1.5 pl-5">
                    <Layers className="h-3 w-3 text-blue-400 flex-shrink-0" />
                    <span className="text-xs text-secondary">
                      <span className="text-primary font-medium">{status.output_count.toLocaleString()}</span>
                      {' '}result{status.output_count !== 1 ? 's' : ''}
                    </span>
                  </div>
                )}
              </div>
            )}

            <p className="text-xs text-muted leading-relaxed pt-1">
              Tiled provides structured access to raw detector frames and visualizes computed embeddings.
            </p>
          </div>
        )}
        {loading && !status && <p className="text-xs text-muted">Checking Tiled…</p>}
      </CardBody>
    </Card>
  )
}
