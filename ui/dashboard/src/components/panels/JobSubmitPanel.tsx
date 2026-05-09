import React, { useState, useEffect } from 'react'
import { Play, X, Plus, Trash2, ChevronDown, ChevronUp, Loader2, CheckCircle, XCircle } from 'lucide-react'
import { postJSON } from '../../api'
import type { BatchJobRequest, StreamJobRequest, ParamSpec, MLflowModel, AppConfig, JobSubmitResult } from '../../types'
import { LogViewer } from '../LogViewer'

interface JobSubmitPanelProps {
  config: AppConfig | null
  models: MLflowModel[]
  onClose: () => void
}

type Mode = 'batch' | 'stream'
type Backend = 'nersc' | 'orion'

const NERSC_QUEUES = ['shared', 'debug', 'regular', 'premium', 'express_amsc_g', 'express_amsc']
const THUMB_MODES = ['default', 'logroi', 'log', 'roi']

export function JobSubmitPanel({ config, models, onClose }: JobSubmitPanelProps) {
  const [mode, setMode] = useState<Mode>('batch')
  const [backend, setBackend] = useState<Backend>((config?.compute_backend as Backend) || 'nersc')
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [result, setResult] = useState<JobSubmitResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [logLines, setLogLines] = useState<string[]>([])

  // Form state
  const [modelName, setModelName] = useState(models[0]?.name || 'vae')
  const [mlflowVersion, setMlflowVersion] = useState('')
  const [inputContainer, setInputContainer] = useState(config?.tiled_input_container || '')
  const [outputContainer, setOutputContainer] = useState(config?.tiled_output_container || '')
  const [batchSize, setBatchSize] = useState(1)
  const [imageKey, setImageKey] = useState('primary/pil900KW_image')
  const [thumbMode, setThumbMode] = useState('default')
  const [projector, setProjector] = useState('')
  const [classifier, setClassifier] = useState('')
  const [paramSpecs, setParamSpecs] = useState<ParamSpec[]>([])

  // NERSC
  const [nerscQueue, setNerscQueue] = useState(config?.nersc_queue || 'shared')
  const [nerscAccount, setNerscAccount] = useState(config?.nersc_account || '')
  const [nerscTimeLimit, setNerscTimeLimit] = useState(config?.nersc_time_limit || '00:30:00')
  const [nerscConstraint, setNerscConstraint] = useState('')

  // Orion
  const [orionAccount, setOrionAccount] = useState(config?.orion_account || '')

  // Stream-only
  const [runPath, setRunPath] = useState('')

  useEffect(() => {
    if (models.length > 0 && !models.find(m => m.name === modelName)) {
      setModelName(models[0].name)
    }
  }, [models])

  const addParam = () => setParamSpecs(p => [...p, { name: '', source: '', dtype: 'float', units: '' }])
  const removeParam = (i: number) => setParamSpecs(p => p.filter((_, j) => j !== i))
  const updateParam = (i: number, key: keyof ParamSpec, val: string) =>
    setParamSpecs(p => p.map((s, j) => j === i ? { ...s, [key]: val } : s))

  const handleSubmit = async () => {
    setSubmitting(true)
    setError(null)
    setResult(null)
    setLogLines([`Submitting ${mode} job to ${backend}…`])

    try {
      if (mode === 'batch') {
        const req: BatchJobRequest = {
          backend,
          model_name: modelName,
          mlflow_version: mlflowVersion,
          input_container: inputContainer,
          output_container: outputContainer,
          batch_size: batchSize,
          image_key: imageKey,
          thumb_mode: thumbMode,
          param_specs: paramSpecs.filter(p => p.name && p.source),
          projector: projector || undefined,
          classifier: classifier || undefined,
          nersc_queue: nerscQueue,
          nersc_account: nerscAccount,
          nersc_time_limit: nerscTimeLimit,
          nersc_constraint: nerscConstraint,
          orion_account: orionAccount,
        }
        const res = await postJSON<JobSubmitResult>('/jobs/batch', req)
        setResult(res)
        setLogLines(prev => [
          ...prev,
          `✓ Job submitted: ${res.job_id}`,
          `  Backend: ${res.backend}`,
          res.log_path ? `  Log path: ${res.log_path}` : '  Log path: (polling via status)',
        ])
      } else {
        const req: StreamJobRequest = {
          model_name: modelName,
          mlflow_version: mlflowVersion,
          run_path: runPath || inputContainer,
          output_container: outputContainer,
          batch_size: batchSize,
          image_key: imageKey,
          thumb_mode: thumbMode,
          param_specs: paramSpecs.filter(p => p.name && p.source),
          projector: projector || undefined,
          classifier: classifier || undefined,
          nersc_queue: nerscQueue,
          nersc_account: nerscAccount,
          nersc_time_limit: nerscTimeLimit || '02:00:00',
          nersc_constraint: nerscConstraint,
        }
        const res = await postJSON<JobSubmitResult>('/jobs/stream', req)
        setResult(res)
        setLogLines(prev => [
          ...prev,
          `✓ Streaming job submitted: ${res.job_id}`,
          `  Backend: ${res.backend}`,
          res.log_path ? `  Log path: ${res.log_path}` : '',
          '  The job is now subscribed to the Tiled WebSocket for incoming frames.',
        ])
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err)
      setError(msg)
      setLogLines(prev => [...prev, `✗ Error: ${msg}`])
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center sm:items-center p-4">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" onClick={onClose} />

      {/* Panel */}
      <div className="relative w-full max-w-2xl max-h-[90vh] overflow-y-auto scrollbar-thin rounded-2xl border border-gray-700 bg-gray-900 shadow-2xl">
        {/* Header */}
        <div className="sticky top-0 z-10 flex items-center justify-between px-6 py-4 border-b border-gray-800 bg-gray-900/95 backdrop-blur">
          <div>
            <h2 className="font-semibold text-gray-100">Submit Inference Job</h2>
            <p className="text-xs text-gray-500">Configure and launch a computation</p>
          </div>
          <button onClick={onClose} className="rounded-lg p-1.5 hover:bg-gray-800 transition-colors">
            <X className="h-4 w-4 text-gray-400" />
          </button>
        </div>

        <div className="px-6 py-5 space-y-5">
          {/* Mode + Backend selector */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-gray-500 mb-1.5">Mode</label>
              <div className="flex rounded-lg border border-gray-800 overflow-hidden text-xs">
                {(['batch', 'stream'] as Mode[]).map(m => (
                  <button
                    key={m}
                    onClick={() => setMode(m)}
                    className={`flex-1 py-2 font-medium transition-colors capitalize ${
                      mode === m ? 'bg-indigo-900 text-indigo-200' : 'bg-gray-950 text-gray-500 hover:text-gray-300'
                    }`}
                  >
                    {m === 'batch' ? '📊 Batch' : '📡 Streaming'}
                  </button>
                ))}
              </div>
            </div>
            <div>
              <label className="block text-xs text-gray-500 mb-1.5">Backend</label>
              <div className="flex rounded-lg border border-gray-800 overflow-hidden text-xs">
                {(['nersc', 'orion'] as Backend[]).map(b => (
                  <button
                    key={b}
                    onClick={() => setBackend(b)}
                    disabled={mode === 'stream' && b === 'orion'}
                    className={`flex-1 py-2 font-medium transition-colors uppercase ${
                      backend === b ? 'bg-violet-900 text-violet-200' : 'bg-gray-950 text-gray-500 hover:text-gray-300'
                    } disabled:opacity-30 disabled:cursor-not-allowed`}
                  >
                    {b}
                  </button>
                ))}
              </div>
              {mode === 'stream' && <p className="text-xs text-gray-600 mt-1">Streaming only supported on NERSC</p>}
            </div>
          </div>

          <hr className="border-gray-800" />

          {/* Model selection */}
          <div className="space-y-3">
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wide">Model</h3>
            <div className="grid grid-cols-2 gap-3">
              <Field label="Model Name">
                {models.length > 0 ? (
                  <select
                    value={modelName}
                    onChange={e => setModelName(e.target.value)}
                    className="w-full bg-gray-950 border border-gray-800 rounded px-2 py-1.5 text-xs text-gray-300 font-mono focus:outline-none focus:border-indigo-700"
                  >
                    {models.map(m => (
                      <option key={m.name} value={m.name}>{m.name} (v{m.latest_version})</option>
                    ))}
                    <option value="vae">vae (local)</option>
                    <option value="vit">vit (local)</option>
                  </select>
                ) : (
                  <Input value={modelName} onChange={setModelName} placeholder="e.g. vae or bnl-nsls2-smi-vae" mono />
                )}
              </Field>
              <Field label="MLflow Version (blank = latest)">
                <Input value={mlflowVersion} onChange={setMlflowVersion} placeholder="e.g. 3" mono />
              </Field>
            </div>
          </div>

          <hr className="border-gray-800" />

          {/* Tiled paths */}
          <div className="space-y-3">
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wide">Tiled Containers</h3>
            {mode === 'stream' ? (
              <Field label="Input Run Path (BlueskyRun Tiled path)">
                <Input value={runPath} onChange={setRunPath} placeholder={inputContainer || 'e.g. smi/sandbox/run123'} mono />
              </Field>
            ) : (
              <Field label="Input Container">
                <Input value={inputContainer} onChange={setInputContainer} placeholder="tiled/path/to/input" mono />
              </Field>
            )}
            <Field label="Output Container">
              <Input value={outputContainer} onChange={setOutputContainer} placeholder="tiled/path/to/output" mono />
            </Field>
          </div>

          <hr className="border-gray-800" />

          {/* Computation parameters */}
          <div className="space-y-3">
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wide">Computation Parameters</h3>
            <div className="grid grid-cols-3 gap-3">
              <Field label="Batch Size">
                <input
                  type="number" min={1} max={256}
                  value={batchSize}
                  onChange={e => setBatchSize(Number(e.target.value))}
                  className="w-full bg-gray-950 border border-gray-800 rounded px-2 py-1.5 text-xs text-gray-300 font-mono focus:outline-none focus:border-indigo-700"
                />
              </Field>
              <Field label="Image Key">
                <Input value={imageKey} onChange={setImageKey} placeholder="primary/pil900KW_image" mono />
              </Field>
              <Field label="Thumbnail Mode">
                <select
                  value={thumbMode}
                  onChange={e => setThumbMode(e.target.value)}
                  className="w-full bg-gray-950 border border-gray-800 rounded px-2 py-1.5 text-xs text-gray-300 font-mono focus:outline-none focus:border-indigo-700"
                >
                  {THUMB_MODES.map(m => <option key={m} value={m}>{m}</option>)}
                </select>
              </Field>
            </div>
          </div>

          {/* Backend-specific parameters */}
          <div className="space-y-3">
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wide">
              {backend.toUpperCase()} Parameters
            </h3>
            {backend === 'nersc' ? (
              <div className="grid grid-cols-2 gap-3">
                <Field label="Queue">
                  <select
                    value={nerscQueue}
                    onChange={e => setNerscQueue(e.target.value)}
                    className="w-full bg-gray-950 border border-gray-800 rounded px-2 py-1.5 text-xs text-gray-300 font-mono focus:outline-none focus:border-indigo-700"
                  >
                    {NERSC_QUEUES.map(q => <option key={q} value={q}>{q}</option>)}
                  </select>
                </Field>
                <Field label="Account">
                  <Input value={nerscAccount} onChange={setNerscAccount} placeholder="e.g. m3792_g" mono />
                </Field>
                <Field label="Time Limit (HH:MM:SS)">
                  <Input value={nerscTimeLimit} onChange={setNerscTimeLimit} placeholder="00:30:00" mono />
                </Field>
                <Field label="Constraint (optional)">
                  <Input value={nerscConstraint} onChange={setNerscConstraint} placeholder="e.g. gpu" mono />
                </Field>
              </div>
            ) : (
              <Field label="Account">
                <Input value={orionAccount} onChange={setOrionAccount} placeholder="e.g. staff" mono />
              </Field>
            )}
          </div>

          {/* Advanced */}
          <div>
            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-300 transition-colors"
            >
              {showAdvanced ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
              Advanced Options (projector, classifier, param specs)
            </button>

            {showAdvanced && (
              <div className="mt-3 space-y-3">
                <div className="grid grid-cols-2 gap-3">
                  <Field label="Projector (blank = fit scratch)">
                    <Input value={projector} onChange={setProjector} placeholder="none / model-name" mono />
                  </Field>
                  <Field label="Classifier (optional)">
                    <Input value={classifier} onChange={setClassifier} placeholder="model-name" mono />
                  </Field>
                </div>

                {/* Param specs */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <label className="text-xs text-gray-500">Experimental Parameter Specs</label>
                    <button onClick={addParam} className="flex items-center gap-1 text-xs text-indigo-400 hover:text-indigo-300">
                      <Plus className="h-3.5 w-3.5" /> Add
                    </button>
                  </div>
                  <div className="space-y-1.5">
                    <div className="grid grid-cols-4 gap-1 text-xs text-gray-600 px-1">
                      <span>Name</span><span>Source (Tiled path)</span><span>Type</span><span>Units</span>
                    </div>
                    {paramSpecs.map((p, i) => (
                      <div key={i} className="flex items-center gap-1.5">
                        <div className="grid grid-cols-4 gap-1 flex-1">
                          <Input value={p.name} onChange={v => updateParam(i, 'name', v)} placeholder="e.g. energy" mono />
                          <Input value={p.source} onChange={v => updateParam(i, 'source', v)} placeholder="primary/energy" mono />
                          <select
                            value={p.dtype}
                            onChange={e => updateParam(i, 'dtype', e.target.value)}
                            className="bg-gray-950 border border-gray-800 rounded px-1.5 py-1.5 text-xs text-gray-300 font-mono focus:outline-none focus:border-indigo-700"
                          >
                            {['float', 'int', 'str'].map(t => <option key={t} value={t}>{t}</option>)}
                          </select>
                          <Input value={p.units} onChange={v => updateParam(i, 'units', v)} placeholder="eV" mono />
                        </div>
                        <button onClick={() => removeParam(i)} className="text-gray-700 hover:text-red-400">
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    ))}
                    {paramSpecs.length === 0 && (
                      <p className="text-xs text-gray-700 italic px-1">No param specs — add one to track experimental parameters in Tiled</p>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>

          <hr className="border-gray-800" />

          {/* Submit button */}
          <button
            onClick={handleSubmit}
            disabled={submitting}
            className="w-full flex items-center justify-center gap-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:bg-indigo-900 disabled:text-indigo-500 text-white font-medium py-2.5 text-sm transition-colors"
          >
            {submitting ? (
              <><Loader2 className="h-4 w-4 animate-spin" /> Submitting…</>
            ) : (
              <><Play className="h-4 w-4" /> Submit {mode} job to {backend.toUpperCase()}</>
            )}
          </button>

          {/* Result / Error */}
          {result && (
            <div className="flex items-start gap-2 rounded-lg bg-emerald-950/50 border border-emerald-800 px-3 py-2.5">
              <CheckCircle className="h-4 w-4 text-emerald-400 mt-0.5 flex-shrink-0" />
              <div>
                <p className="text-xs text-emerald-300 font-medium">Job submitted successfully</p>
                <p className="text-xs text-emerald-600 font-mono">ID: {result.job_id}</p>
              </div>
            </div>
          )}
          {error && (
            <div className="flex items-start gap-2 rounded-lg bg-red-950/50 border border-red-800 px-3 py-2.5">
              <XCircle className="h-4 w-4 text-red-400 mt-0.5 flex-shrink-0" />
              <p className="text-xs text-red-300 font-mono">{error}</p>
            </div>
          )}

          {/* Log output */}
          {logLines.length > 0 && (
            <LogViewer lines={logLines} title="Submission output" autoScroll />
          )}
        </div>
      </div>
    </div>
  )
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="block text-xs text-gray-500 mb-1">{label}</label>
      {children}
    </div>
  )
}

function Input({
  value, onChange, placeholder, mono = false
}: {
  value: string
  onChange: (v: string) => void
  placeholder?: string
  mono?: boolean
}) {
  return (
    <input
      type="text"
      value={value}
      onChange={e => onChange(e.target.value)}
      placeholder={placeholder}
      className={`w-full bg-gray-950 border border-gray-800 rounded px-2 py-1.5 text-xs text-gray-300 ${mono ? 'font-mono' : ''} placeholder-gray-700 focus:outline-none focus:border-indigo-700`}
    />
  )
}
