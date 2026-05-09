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

  const fetchStatus = async () => {
    setLoading(true)
    try {
      const data = await fetchJSON<TiledStatus>('/tiled/status')
      setStatus(data)
    } catch {
      setStatus({ status: 'error', error: 'Failed to reach dashboard API' })
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchStatus() }, [refreshTick])

  const serviceStatus = loading ? 'loading' : (status?.status ?? 'error')

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-950 border border-blue-800">
            <Database className="h-4 w-4 text-blue-400" />
          </div>
          <div>
            <h3 className="font-semibold text-gray-100">Tiled Data Service</h3>
            <p className="text-xs text-gray-500">Scientific data layer — input &amp; output</p>
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
                  className="flex items-center gap-1 text-xs text-blue-400 hover:text-blue-300 font-mono truncate max-w-xs"
                >
                  {status.server_uri}
                  <ExternalLink className="h-3 w-3 flex-shrink-0" />
                </a>
              </div>
            )}
            {status.tiled_version && (
              <div className="flex items-center justify-between">
                <span className="text-xs text-gray-500">Tiled Version</span>
                <span className="text-xs text-gray-300 font-mono">{status.tiled_version}</span>
              </div>
            )}
            {status.api_version && (
              <div className="flex items-center justify-between">
                <span className="text-xs text-gray-500">API Version</span>
                <span className="text-xs text-gray-300 font-mono">{status.api_version}</span>
              </div>
            )}
            {status.error && (
              <p className="text-xs text-red-400 bg-red-950/50 border border-red-900 rounded px-2 py-1.5 font-mono">
                {status.error}
              </p>
            )}
            {status.message && (
              <p className="text-xs text-gray-500">{status.message}</p>
            )}

            {/* Container paths */}
            <div className="pt-1 space-y-2">
              <ContainerPath label="Input Container" path={config?.tiled_input_container} />
              <ContainerPath label="Output Container" path={config?.tiled_output_container} />
            </div>
          </div>
        )}
        {loading && <p className="text-xs text-gray-600">Checking Tiled…</p>}
      </CardBody>
    </Card>
  )
}

function ContainerPath({ label, path }: { label: string; path?: string }) {
  return (
    <div className="flex items-start gap-2 rounded-lg bg-gray-950 border border-gray-800 px-3 py-2">
      <Folder className="h-3.5 w-3.5 text-gray-600 mt-0.5 flex-shrink-0" />
      <div className="min-w-0">
        <p className="text-xs text-gray-500">{label}</p>
        <p className="text-xs text-gray-300 font-mono truncate">
          {path || <span className="text-gray-600 italic">not configured</span>}
        </p>
      </div>
    </div>
  )
}
