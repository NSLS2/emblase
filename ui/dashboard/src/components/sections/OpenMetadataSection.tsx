import React, { useEffect, useState, useCallback } from 'react'
import { BookOpen, ExternalLink, RefreshCw, Search, Database, Layers, FileText } from 'lucide-react'
import { fetchJSON } from '../../api'
import type { OpenMetadataStatus, OpenMetadataArtifacts, CatalogArtifact } from '../../types'
import { StatusBadge } from '../StatusBadge'
import { Card, CardHeader, CardBody } from '../Card'

interface OpenMetadataSectionProps {
  refreshTick: number
}

const ENTITY_ICON: Record<string, React.ElementType> = {
  artifact: FileText,
  artifactCollection: Layers,
  scientificWork: Database,
}

function entityIcon(type: string): React.ElementType {
  return ENTITY_ICON[type] ?? BookOpen
}

function ArtifactRow({ artifact }: { artifact: CatalogArtifact }) {
  const Icon = entityIcon(artifact.entity_type)
  const label = artifact.display_name || artifact.name
  const shortFqn = artifact.fqn.split('.').slice(-2).join('.')

  return (
    <div className="flex items-start gap-2.5 py-2 border-b border-theme last:border-0">
      <div className="mt-0.5 flex-shrink-0">
        <Icon className="h-3.5 w-3.5 text-amber-500" />
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium text-primary truncate">{label}</span>
          <span className="text-[10px] text-muted bg-amber-500/10 border border-amber-500/20 rounded px-1 flex-shrink-0">
            {artifact.entity_type}
          </span>
        </div>
        {artifact.description && (
          <p className="text-[11px] text-secondary mt-0.5 line-clamp-2 leading-relaxed">
            {artifact.description}
          </p>
        )}
        <div className="flex items-center gap-2 mt-1">
          <span className="text-[10px] text-muted font-mono truncate" title={artifact.fqn}>
            {shortFqn}
          </span>
          {artifact.location && (
            <a
              href={artifact.location}
              target="_blank"
              rel="noopener noreferrer"
              className="flex-shrink-0 text-[10px] text-amber-500 hover:text-amber-400 flex items-center gap-0.5"
            >
              <ExternalLink className="h-2.5 w-2.5" />
              open
            </a>
          )}
        </div>
      </div>
    </div>
  )
}

export function OpenMetadataSection({ refreshTick }: OpenMetadataSectionProps) {
  const [status, setStatus] = useState<OpenMetadataStatus | null>(null)
  const [artifacts, setArtifacts] = useState<OpenMetadataArtifacts | null>(null)
  const [query, setQuery] = useState('')
  const [inputValue, setInputValue] = useState('')
  const [loadingStatus, setLoadingStatus] = useState(false)
  const [loadingArtifacts, setLoadingArtifacts] = useState(false)

  const fetchStatus = useCallback(() => {
    setLoadingStatus(true)
    fetchJSON<OpenMetadataStatus>('/openmetadata/status')
      .then(setStatus)
      .catch(() => setStatus({ status: 'error', error: 'Request failed' }))
      .finally(() => setLoadingStatus(false))
  }, [])

  const fetchArtifacts = useCallback((q: string) => {
    setLoadingArtifacts(true)
    const params = new URLSearchParams({ limit: '20' })
    if (q) params.set('query', q)
    fetchJSON<OpenMetadataArtifacts>(`/openmetadata/artifacts?${params}`)
      .then(setArtifacts)
      .catch(() => setArtifacts({ artifacts: [], total: 0, error: 'Request failed' }))
      .finally(() => setLoadingArtifacts(false))
  }, [])

  // Refresh on tick
  useEffect(() => {
    fetchStatus()
    fetchArtifacts(query)
  }, [refreshTick]) // eslint-disable-line react-hooks/exhaustive-deps

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    setQuery(inputValue)
    fetchArtifacts(inputValue)
  }

  const isOnline = status?.status === 'online'
  const isUnconfigured = status?.status === 'unconfigured'
  const catalogUrl = status?.catalog_url
  const projectCount = status?.project_count ?? 0

  return (
    <Card>
      {/* ── Header ── */}
      <CardHeader>
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-amber-500/10 border border-amber-500/30">
            <BookOpen className="h-4 w-4 text-amber-500" />
          </div>
          <div>
            <h3 className="font-semibold text-primary">AmSC Data Catalog</h3>
            <p className="text-xs text-secondary">OpenMetadata · artifact registry</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => { fetchStatus(); fetchArtifacts(query) }}
            disabled={loadingStatus || loadingArtifacts}
            className="p-1 rounded text-muted hover:text-primary transition-colors disabled:opacity-40"
            title="Refresh"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${(loadingStatus || loadingArtifacts) ? 'animate-spin' : ''}`} />
          </button>
          <StatusBadge status={status?.status ?? 'loading'} size="sm" />
        </div>
      </CardHeader>

      <CardBody>
        <div className="space-y-4">

          {/* ── Connectivity info ── */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-xs">
              <span className="text-secondary">Catalog URL</span>
              {catalogUrl ? (
                <a
                  href={catalogUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-amber-500 hover:text-amber-400 font-mono flex items-center gap-1"
                >
                  {new URL(catalogUrl).hostname}
                  <ExternalLink className="h-3 w-3" />
                </a>
              ) : (
                <span className="text-muted font-mono italic">not configured</span>
              )}
            </div>
            {status?.catalog_name && (
              <div className="flex items-center justify-between text-xs">
                <span className="text-secondary">Catalog name</span>
                <span className="text-primary font-mono">{status.catalog_name}</span>
              </div>
            )}
            {status?.root_fqn && (
              <div className="flex items-start justify-between text-xs gap-2">
                <span className="text-secondary flex-shrink-0">Root FQN</span>
                <span className="text-muted font-mono text-[10px] text-right break-all" title={status.root_fqn}>
                  {status.root_fqn}
                </span>
              </div>
            )}
            {isOnline && (
              <div className="flex items-center justify-between text-xs">
                <span className="text-secondary">Projects</span>
                <span className="text-primary">{projectCount}</span>
              </div>
            )}
            {status?.parent_fqn && (
              <div className="flex items-center justify-between text-xs">
                <span className="text-secondary">Root FQN</span>
                <span className="text-muted font-mono text-[10px] truncate max-w-[180px]" title={status.parent_fqn}>
                  {status.parent_fqn}
                </span>
              </div>
            )}
            {isUnconfigured && (
              <p className="text-xs text-secondary leading-relaxed">
                Set <code className="text-amber-400">EMBLASE_AMSC_OPENMETADATA_TOKEN</code> and{' '}
                <code className="text-amber-400">EMBLASE_AMSC_OPENMETADATA_CATALOG_NAME</code> in your .env file to enable catalog integration.
              </p>
            )}
            {status?.status === 'error' && (
              <p className="text-xs text-red-400 break-all">{status.error}</p>
            )}
          </div>

          {/* ── Recent artifacts (only when online) ── */}
          {isOnline && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-muted uppercase tracking-wide">
                  Catalog Entries
                  {artifacts && !artifacts.error && (
                    <span className="ml-1 text-amber-500 normal-case font-normal">
                      ({artifacts.total})
                    </span>
                  )}
                </span>
              </div>

              {/* Search bar */}
              <form onSubmit={handleSearch} className="flex gap-1.5">
                <div className="relative flex-1">
                  <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-3 w-3 text-muted pointer-events-none" />
                  <input
                    type="text"
                    value={inputValue}
                    onChange={e => setInputValue(e.target.value)}
                    placeholder="Search catalog…"
                    className="w-full pl-6 pr-2 py-1 text-xs bg-surface border border-theme rounded focus:outline-none focus:border-amber-500/50 text-primary placeholder:text-muted"
                  />
                </div>
                <button
                  type="submit"
                  disabled={loadingArtifacts}
                  className="px-2 py-1 text-xs bg-amber-500/10 border border-amber-500/30 rounded text-amber-500 hover:bg-amber-500/20 disabled:opacity-40 transition-colors"
                >
                  Search
                </button>
              </form>

              {/* Results */}
              {loadingArtifacts && (
                <div className="text-xs text-secondary text-center py-3">Loading…</div>
              )}
              {!loadingArtifacts && artifacts?.error && (
                <p className="text-xs text-red-400">{artifacts.error}</p>
              )}
              {!loadingArtifacts && artifacts && !artifacts.error && artifacts.artifacts.length === 0 && (
                <p className="text-xs text-secondary text-center py-3 italic">
                  {query ? `No results for "${query}"` : 'No catalog entries found'}
                </p>
              )}
              {!loadingArtifacts && artifacts && artifacts.artifacts.length > 0 && (
                <div className="max-h-72 overflow-y-auto rounded border border-theme">
                  {artifacts.artifacts.map(a => (
                    <ArtifactRow key={a.fqn || a.name} artifact={a} />
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </CardBody>
    </Card>
  )
}
