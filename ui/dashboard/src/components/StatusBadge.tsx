import React from 'react'
import type { ServiceStatus } from '../types'

const STATUS_CONFIG: Record<ServiceStatus, { label: string; dot: string; bg: string; text: string }> = {
  online:        { label: 'Online',        dot: 'bg-emerald-400 animate-pulse', bg: 'bg-emerald-950 border-emerald-800', text: 'text-emerald-400' },
  offline:       { label: 'Offline',       dot: 'bg-red-500',                   bg: 'bg-red-950 border-red-800',         text: 'text-red-400'     },
  error:         { label: 'Error',         dot: 'bg-red-500',                   bg: 'bg-red-950 border-red-800',         text: 'text-red-400'     },
  timeout:       { label: 'Timeout',       dot: 'bg-orange-500',                bg: 'bg-orange-950 border-orange-800',   text: 'text-orange-400'  },
  unconfigured:  { label: 'Not Configured',dot: 'bg-gray-500',                  bg: 'bg-gray-900 border-gray-700',       text: 'text-gray-400'    },
  placeholder:   { label: 'Coming Soon',   dot: 'bg-blue-500',                  bg: 'bg-blue-950 border-blue-800',       text: 'text-blue-400'    },
  degraded:      { label: 'Degraded',      dot: 'bg-yellow-500 animate-pulse',  bg: 'bg-yellow-950 border-yellow-800',   text: 'text-yellow-400'  },
  loading:       { label: 'Checking…',     dot: 'bg-gray-500 animate-pulse',    bg: 'bg-gray-900 border-gray-700',       text: 'text-gray-400'    },
}

interface StatusBadgeProps {
  status: ServiceStatus
  className?: string
  size?: 'sm' | 'md'
}

export function StatusBadge({ status, className = '', size = 'md' }: StatusBadgeProps) {
  const cfg = STATUS_CONFIG[status] ?? STATUS_CONFIG.error
  const padding = size === 'sm' ? 'px-2 py-0.5 text-xs' : 'px-3 py-1 text-sm'
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border font-medium ${padding} ${cfg.bg} ${cfg.text} ${className}`}>
      <span className={`inline-block h-2 w-2 rounded-full ${cfg.dot}`} />
      {cfg.label}
    </span>
  )
}

interface StatusDotProps {
  status: ServiceStatus
}

export function StatusDot({ status }: StatusDotProps) {
  const cfg = STATUS_CONFIG[status] ?? STATUS_CONFIG.error
  return <span className={`inline-block h-2.5 w-2.5 rounded-full ${cfg.dot}`} />
}
