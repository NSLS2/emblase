import React from 'react'
import type { ServiceStatus } from '../types'

const STATUS_CONFIG: Record<ServiceStatus, { label: string; dot: string; cls: string }> = {
  online:       { label: 'Online',         dot: 'bg-emerald-400 animate-pulse', cls: 'bg-emerald-50  dark:bg-emerald-950  border-emerald-300 dark:border-emerald-800  text-emerald-700 dark:text-emerald-400' },
  offline:      { label: 'Offline',        dot: 'bg-red-500',                   cls: 'bg-red-50     dark:bg-red-950      border-red-300     dark:border-red-800      text-red-700     dark:text-red-400'     },
  error:        { label: 'Error',          dot: 'bg-red-500',                   cls: 'bg-red-50     dark:bg-red-950      border-red-300     dark:border-red-800      text-red-700     dark:text-red-400'     },
  timeout:      { label: 'Timeout',        dot: 'bg-orange-500',                cls: 'bg-orange-50  dark:bg-orange-950   border-orange-300  dark:border-orange-800   text-orange-700  dark:text-orange-400'  },
  unconfigured: { label: 'Not Configured', dot: 'bg-slate-400',                 cls: 'bg-slate-100  dark:bg-slate-900    border-slate-300   dark:border-slate-700    text-slate-600   dark:text-slate-400'   },
  placeholder:  { label: 'Coming Soon',    dot: 'bg-blue-400',                  cls: 'bg-blue-50    dark:bg-blue-950     border-blue-300    dark:border-blue-800     text-blue-700    dark:text-blue-400'    },
  degraded:     { label: 'Degraded',       dot: 'bg-yellow-500 animate-pulse',  cls: 'bg-yellow-50  dark:bg-yellow-950   border-yellow-300  dark:border-yellow-800   text-yellow-700  dark:text-yellow-400'  },
  loading:      { label: 'Checking…',      dot: 'bg-slate-400 animate-pulse',   cls: 'bg-slate-100  dark:bg-slate-900    border-slate-300   dark:border-slate-700    text-slate-500   dark:text-slate-400'   },
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
    <span className={`inline-flex items-center gap-1.5 rounded-full border font-medium ${padding} ${cfg.cls} ${className}`}>
      <span className={`inline-block h-2 w-2 rounded-full ${cfg.dot}`} />
      {cfg.label}
    </span>
  )
}
