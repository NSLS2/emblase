import React, { useEffect, useState } from 'react'
import { BookOpen, Webhook, GitBranch, Search } from 'lucide-react'
import { fetchJSON } from '../../api'
import type { OpenMetadataStatus } from '../../types'
import { StatusBadge } from '../StatusBadge'
import { Card, CardHeader, CardBody } from '../Card'

interface OpenMetadataSectionProps {
  refreshTick: number
}

export function OpenMetadataSection({ refreshTick }: OpenMetadataSectionProps) {
  const [data, setData] = useState<OpenMetadataStatus | null>(null)

  useEffect(() => {
    fetchJSON<OpenMetadataStatus>('/openmetadata/status').then(setData).catch(() => {})
  }, [refreshTick])

  const featureIcons = [Webhook, GitBranch, Search, BookOpen]

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-amber-500/10 border border-amber-500/30">
            <BookOpen className="h-4 w-4 text-amber-500" />
          </div>
          <div>
            <h3 className="font-semibold text-primary">Data Catalog</h3>
            <p className="text-xs text-secondary">OpenMetadata — coming soon</p>
          </div>
        </div>
        <StatusBadge status="placeholder" size="sm" />
      </CardHeader>
      <CardBody>
        <div className="space-y-3">
          <p className="text-xs text-secondary leading-relaxed">
            {data?.message ??
              'A third-party metadata catalog (OpenMetadata) will receive automatic dataset registration updates from Tiled via webhooks — a new Tiled feature currently in development.'}
          </p>

          {data?.planned_features && (
            <div className="space-y-2">
              <p className="text-xs font-medium text-muted uppercase tracking-wide">Planned Features</p>
              {data.planned_features.map((f, i) => {
                const Icon = featureIcons[i % featureIcons.length]
                return (
                  <div key={i} className="flex items-start gap-2 text-xs text-secondary">
                    <Icon className="h-3.5 w-3.5 text-amber-500 mt-0.5 flex-shrink-0" />
                    <span>{f}</span>
                  </div>
                )
              })}
            </div>
          )}

          <div className="space-y-2 opacity-50">
            <div className="flex items-center justify-between">
              <span className="text-xs text-secondary">Catalog URL</span>
              <span className="text-xs text-muted font-mono italic">not configured</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs text-secondary">Tiled Webhook</span>
              <span className="text-xs text-muted font-mono italic">pending</span>
            </div>
          </div>
        </div>
      </CardBody>
    </Card>
  )
}
