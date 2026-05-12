import React, { useEffect, useRef } from 'react'
import { Terminal, Copy } from 'lucide-react'

interface LogViewerProps {
  lines: string[]
  title?: string
  maxLines?: number
  className?: string
  autoScroll?: boolean
}

function ansiToHtml(line: string): string {
  return line
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/\x1b\[0m/g, '</span>')
    .replace(/\x1b\[1m/g, '<span class="font-bold">')
    .replace(/\x1b\[31m/g, '<span class="log-red">')
    .replace(/\x1b\[32m/g, '<span class="log-green">')
    .replace(/\x1b\[33m/g, '<span class="log-yellow">')
    .replace(/\x1b\[34m/g, '<span class="log-blue">')
    .replace(/\x1b\[35m/g, '<span class="log-magenta">')
    .replace(/\x1b\[36m/g, '<span class="log-cyan">')
    .replace(/\x1b\[37m/g, '<span class="log-white">')
    .replace(/\x1b\[[0-9;]*m/g, '')
}

export function LogViewer({ lines, title = 'Logs', maxLines = 500, className = '', autoScroll = true }: LogViewerProps) {
  const bottomRef = useRef<HTMLDivElement>(null)
  const displayed = lines.slice(-maxLines)

  useEffect(() => {
    if (autoScroll && bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [lines, autoScroll])

  const copyAll = () => navigator.clipboard.writeText(lines.join('\n')).catch(() => {})

  return (
    <div className={`rounded-lg border border-theme bg-page overflow-hidden ${className}`}>
      <div className="flex items-center gap-2 px-3 py-2 border-b border-theme bg-card">
        <Terminal className="h-3.5 w-3.5 text-muted" />
        <span className="text-xs font-medium text-secondary font-mono">{title}</span>
        <span className="ml-auto text-xs text-muted mr-2">{lines.length} lines</span>
        <button onClick={copyAll} title="Copy all" className="text-muted hover:text-secondary transition-colors">
          <Copy className="h-3 w-3" />
        </button>
      </div>
      <div className="p-3 h-64 overflow-y-auto scrollbar-thin font-mono text-xs leading-relaxed">
        {displayed.length === 0 ? (
          <span className="text-muted italic">No output yet…</span>
        ) : (
          displayed.map((line, i) => (
            <div key={i} className="log-reset whitespace-pre-wrap"
              dangerouslySetInnerHTML={{ __html: ansiToHtml(line) }} />
          ))
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}
