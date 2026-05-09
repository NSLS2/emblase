import React, { useEffect, useState } from 'react'
import { MessageSquare, ExternalLink } from 'lucide-react'
import { fetchJSON } from '../../api'
import type { ChatbotStatus, AppConfig } from '../../types'
import { StatusBadge } from '../StatusBadge'
import { Card, CardHeader, CardBody } from '../Card'

interface ChatbotSectionProps {
  config: AppConfig | null
  refreshTick: number
}

export function ChatbotSection({ config, refreshTick }: ChatbotSectionProps) {
  const [status, setStatus] = useState<ChatbotStatus | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    fetchJSON<ChatbotStatus>('/chatbot/status')
      .then(setStatus)
      .catch(() => setStatus({ status: 'error', error: 'Could not check chatbot' }))
      .finally(() => setLoading(false))
  }, [refreshTick])

  const svcStatus = loading ? 'loading' : (status?.status ?? 'error')

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-teal-950 border border-teal-800">
            <MessageSquare className="h-4 w-4 text-teal-400" />
          </div>
          <div>
            <h3 className="font-semibold text-gray-100">AI Science Assistant</h3>
            <p className="text-xs text-gray-500">AmSC LLM chat service</p>
          </div>
        </div>
        <StatusBadge status={svcStatus} size="sm" />
      </CardHeader>
      <CardBody>
        <div className="space-y-3">
          {status?.url && (
            <div className="flex items-center justify-between">
              <span className="text-xs text-gray-500">Service URL</span>
              <a
                href={status.url}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1 text-xs text-teal-400 hover:text-teal-300 font-mono truncate max-w-xs"
              >
                {status.url.replace(/^https?:\/\//, '')}
                <ExternalLink className="h-3 w-3 flex-shrink-0" />
              </a>
            </div>
          )}
          {status?.model && (
            <div className="flex items-center justify-between">
              <span className="text-xs text-gray-500">Model</span>
              <span className="text-xs text-gray-300 font-mono">{status.model}</span>
            </div>
          )}
          {status?.http_status && (
            <div className="flex items-center justify-between">
              <span className="text-xs text-gray-500">HTTP Status</span>
              <span className={`text-xs font-mono ${status.http_status < 400 ? 'text-emerald-400' : 'text-red-400'}`}>
                {status.http_status}
              </span>
            </div>
          )}
          {status?.error && (
            <p className="text-xs text-red-400 bg-red-950/50 border border-red-900 rounded px-2 py-1.5 font-mono">
              {status.error}
            </p>
          )}
          <p className="text-xs text-gray-600 leading-relaxed">
            The AI assistant is integrated into the Tiled embedding scatter UI, providing
            scientific context about the latent space and synchrotron data.
            It is primed with dataset metadata (model name, Tiled path, parameter specs)
            so it can answer domain-specific questions about the experiment.
          </p>
        </div>
      </CardBody>
    </Card>
  )
}
