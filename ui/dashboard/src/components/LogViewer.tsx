import React, { useEffect, useRef } from 'react'
import { Terminal } from 'lucide-react'

interface LogViewerProps {
  lines: string[]
  title?: string
  maxLines?: number
  className?: string
  autoScroll?: boolean
}

// Very basic ANSI → HTML (handles common color codes)
function ansiToHtml(line: string): string {
  return line
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\x1b\[0m/g, '</span>')
    .replace(/\x1b\[1m/g, '<span class="font-bold">')
    .replace(/\x1b\[31m/g, '<span class="log-red">')
    .replace(/\x1b\[32m/g, '<span class="log-green">')
    .replace(/\x1b\[33m/g, '<span class="log-yellow">')
    .replace(/\x1b\[34m/g, '<span class="log-blue">')
    .replace(/\x1b\[35m/g, '<span class="log-magenta">')
    .replace(/\x1b\[36m/g, '<span class="log-cyan">')
    .replace(/\x1b\[37m/g, '<span class="log-white">')
    .replace(/\x1b\[[0-9;]*m/g, '') // strip remaining ANSI codes
}

export function LogViewer({ lines, title = 'Logs', maxLines = 500, className = '', autoScroll = true }: LogViewerProps) {
  const bottomRef = useRef<HTMLDivElement>(null)
  const displayed = lines.slice(-maxLines)

  useEffect(() => {
    if (autoScroll && bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [lines, autoScroll])

  return (
    <div className={`rounded-lg border border-gray-800 bg-gray-950 overflow-hidden ${className}`}>
      <div className="flex items-center gap-2 px-3 py-2 border-b border-gray-800 bg-gray-900">
        <Terminal className="h-3.5 w-3.5 text-gray-500" />
        <span className="text-xs font-medium text-gray-400 font-mono">{title}</span>
        <span className="ml-auto text-xs text-gray-600">{lines.length} lines</span>
      </div>
      <div className="p-3 h-64 overflow-y-auto scrollbar-thin font-mono text-xs leading-relaxed">
        {displayed.length === 0 ? (
          <span className="text-gray-600 italic">No output yet…</span>
        ) : (
          displayed.map((line, i) => (
            <div
              key={i}
              className="log-reset whitespace-pre-wrap"
              dangerouslySetInnerHTML={{ __html: ansiToHtml(line) }}
            />
          ))
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}
