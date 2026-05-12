import React, { useState, useCallback } from 'react'
import { Radio, X, ChevronDown, ChevronUp, Loader2, XCircle, RotateCcw, Info, Plus, Trash2 } from 'lucide-react'
import { postJSON } from '../../api'
import type { WatcherStartRequest, ParamSpec, MLflowModel, AppConfig } from '../../types'

const STORAGE_KEY = 'emblase-watcher-form'
const NERSC_QUEUES = [
  { value: 'shared',         label: 'shared',         desc: 'Shared GPU nodes — best for single-GPU jobs' },
  { value: 'debug',          label: 'debug',           desc: 'Fast dispatch, ≤ 30 min cap' },
  { value: 'regular',        label: 'regular',         desc: 'Standard GPU partition' },
  { value: 'premium',        label: 'premium',         desc: 'Faster turnaround, higher cost' },
  { value: 'express_amsc_g', label: 'express_amsc_g',  desc: '32 reserved AMSC GPU nodes' },
]
const THUMB_MODES = ['logroi', 'default']

type Backend = 'nersc' | 'orion'

interface Form {
  backend: Backend
  modelName: string
  mlflowVersion: string
  outputContainer: string
  batchSize: number
  imageKey: string
  thumbMode: string
  projector: string
  classifier: string
  paramSpecs: ParamSpec[]
  nerscQueue: string
  nerscAccount: string
  nerscTimeLimit: string
  nerscConstraint: string
  orionAccount: string
  replayExisting: boolean
}

function defaultForm(config: AppConfig | null): Form {
  return {
    backend: 'orion',
    modelName: 'bnl-nsls2-smi-vit',
    mlflowVersion: '',
    outputContainer: config?.tiled_output_container || '',
    batchSize: 4,
    imageKey: 'primary/pil900KW_image',
    thumbMode: 'logroi',
    projector: 'bnl-nsls2-smi-umap',
    classifier: 'bnl-nsls2-smi-class5',
    paramSpecs: [
      { name: 'temperature', source: 'primary/LinkamThermal_temperature_current', dtype: 'float', units: '°C' },
      { name: 'piezo_x', source: 'primary/piezo_x', dtype: 'float', units: 'μm' },
    ],
    nerscQueue: config?.nersc_queue || 'shared',
    nerscAccount: config?.nersc_account || '',
    nerscTimeLimit: config?.nersc_time_limit || '02:00:00',
    nerscConstraint: '',
    orionAccount: config?.orion_account || '',
    replayExisting: false,
  }
}

function loadForm(config: AppConfig | null): Form {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY)
    if (raw) return { ...defaultForm(config), ...JSON.parse(raw) }
  } catch {}
  return defaultForm(config)
}

function saveForm(f: Form) {
  try { sessionStorage.setItem(STORAGE_KEY, JSON.stringify(f)) } catch {}
}

interface WatcherPanelProps {
  config: AppConfig | null
  models: MLflowModel[]
  backend: Backend          // pre-selected from the button that was clicked
  onClose: () => void
  onStarted: () => void     // called after successful start
}

export function WatcherPanel({ config, models, backend: initialBackend, onClose, onStarted }: WatcherPanelProps) {
  const [form, setFormRaw] = useState<Form>(() => {
    const f = loadForm(config)
    return { ...f, backend: initialBackend }
  })
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const setForm = useCallback((updater: (prev: Form) => Form) => {
    setFormRaw(prev => { const next = updater(prev); saveForm(next); return next })
  }, [])

  const set = <K extends keyof Form>(key: K, val: Form[K]) => setForm(f => ({ ...f, [key]: val }))

  const reset = () => { const f = defaultForm(config); setFormRaw(f); saveForm(f); setError(null) }

  const addParam = () => setForm(f => ({ ...f, paramSpecs: [...f.paramSpecs, { name: '', source: '', dtype: 'float', units: '' }] }))
  const removeParam = (i: number) => setForm(f => ({ ...f, paramSpecs: f.paramSpecs.filter((_, j) => j !== i) }))
  const updateParam = (i: number, key: keyof ParamSpec, val: string) =>
    setForm(f => ({ ...f, paramSpecs: f.paramSpecs.map((s, j) => j === i ? { ...s, [key]: val } : s) }))

  const handleStart = async () => {
    setSubmitting(true)
    setError(null)
    try {
      const req: WatcherStartRequest = {
        backend: form.backend,
        model_name: form.modelName,
        mlflow_version: form.mlflowVersion,
        output_container: form.outputContainer,
        batch_size: form.batchSize,
        image_key: form.imageKey,
        thumb_mode: form.thumbMode,
        param_specs: form.paramSpecs.filter(p => p.name && p.source),
        projector: form.projector || undefined,
        classifier: form.classifier || undefined,
        nersc_queue: form.nerscQueue,
        nersc_account: form.nerscAccount,
        nersc_time_limit: form.nerscTimeLimit,
        nersc_constraint: form.nerscConstraint,
        orion_account: form.orionAccount,
        replay_existing: form.replayExisting,
      }
      await postJSON('/watcher/start', req)
      onStarted()
      onClose()
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setSubmitting(false)
    }
  }

  const inferenceModels = models.filter(m => m.name.endsWith('-vit') || m.name.endsWith('-vae'))
  const auxModels = models.filter(m => !m.name.endsWith('-vit') && !m.name.endsWith('-vae'))

  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center sm:items-center p-4">
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      <div className="relative w-full max-w-2xl max-h-[92vh] overflow-y-auto scrollbar-thin rounded-2xl border border-theme bg-card shadow-2xl">

        {/* Header */}
        <div className="sticky top-0 z-10 flex items-center justify-between px-6 py-4 border-b border-theme bg-card/95 backdrop-blur">
          <div className="flex items-center gap-2">
            <Radio className="h-4 w-4 text-violet-500" />
            <div>
              <h2 className="font-semibold text-primary">Live Watch</h2>
              <p className="text-xs text-secondary">Subscribe to inputs container and auto-submit a streaming job per new run</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button onClick={reset} title="Reset to defaults"
              className="flex items-center gap-1 rounded-lg px-2 py-1.5 text-xs text-muted hover:text-secondary border border-theme hover:border-theme-hover transition-colors">
              <RotateCcw className="h-3 w-3" /> Reset
            </button>
            <button onClick={onClose} className="rounded-lg p-1.5 hover:bg-row-hover transition-colors">
              <X className="h-4 w-4 text-secondary" />
            </button>
          </div>
        </div>

        <div className="px-6 py-5 space-y-5">

          {/* Backend */}
          <div>
            <label className="block text-xs text-secondary mb-1.5">Backend</label>
            <div className="flex rounded-lg border border-theme overflow-hidden text-xs">
              {(['nersc', 'orion'] as Backend[]).map(b => (
                <button key={b} onClick={() => set('backend', b)}
                  className={`flex-1 py-2 font-medium transition-colors uppercase ${
                    form.backend === b ? 'bg-violet-600 dark:bg-violet-900 text-white dark:text-violet-200'
                                      : 'bg-card text-secondary hover:text-primary'}`}>
                  {b}
                </button>
              ))}
            </div>
          </div>

          <hr className="border-theme" />

          {/* Model */}
          <Section title="Model">
            <div className="grid grid-cols-2 gap-3">
              <Field label="Model Name">
                {inferenceModels.length > 0 ? (
                  <select value={form.modelName} onChange={e => set('modelName', e.target.value)} className="w-full input-base">
                    {!inferenceModels.find(m => m.name === 'bnl-nsls2-smi-vit') && (
                      <option value="bnl-nsls2-smi-vit">bnl-nsls2-smi-vit (default)</option>
                    )}
                    {inferenceModels.map(m => <option key={m.name} value={m.name}>{m.name} (v{m.latest_version})</option>)}
                  </select>
                ) : (
                  <TInput value={form.modelName} onChange={v => set('modelName', v)} placeholder="bnl-nsls2-smi-vit" />
                )}
              </Field>
              <Field label={<>MLflow Version (blank = latest) <Tip text="Specific registered model version to load. Leave blank to always use the latest version in MLflow." /></>}>
                <TInput value={form.mlflowVersion} onChange={v => set('mlflowVersion', v)} placeholder="latest" />
              </Field>
            </div>
          </Section>

          <hr className="border-theme" />

          {/* Tiled output */}
          <Section title="Output (Tiled)">
            <Field label={<>Output container <Tip text="Tiled path where embedding results are written. Defaults to the server's EMBLASE_TILED_OUTPUT_CONTAINER if left blank." /></>}>
              <TInput value={form.outputContainer} onChange={v => set('outputContainer', v)}
                placeholder={config?.tiled_output_container || 'e.g. smi/sandbox/demo/output'} />
            </Field>
            <p className="text-xs text-muted mt-1">
              Inputs container: <code className="font-mono">{config?.tiled_input_container || 'from EMBLASE_TILED_INPUT_CONTAINER'}</code>
            </p>
          </Section>

          <hr className="border-theme" />

          {/* Inference params */}
          <Section title="Inference Parameters">
            <div className="grid grid-cols-3 gap-3">
              <Field label={<>Batch size <Tip text="Number of frames encoded per GPU call. Larger values use more GPU memory but may improve throughput." /></>}>
                <input type="number" min={1} max={256} value={form.batchSize}
                  onChange={e => set('batchSize', Number(e.target.value))} className="w-full input-base" />
              </Field>
              <Field label={<>Image key <Tip text="Tiled path to the detector image array, in 'stream/key' form. E.g. 'primary/pil900KW_image'." /></>}>
                <TInput value={form.imageKey} onChange={v => set('imageKey', v)} placeholder="primary/pil900KW_image" />
              </Field>
              <Field label={<>Thumbnail mode <Tip text="How thumbnails are generated: 'logroi' = log-scale crop of the ROI (recommended for scattering); 'default' = linear full frame." /></>}>
                <select value={form.thumbMode} onChange={e => set('thumbMode', e.target.value)} className="w-full input-base">
                  {THUMB_MODES.map(m => <option key={m} value={m}>{m}</option>)}
                </select>
              </Field>
            </div>
          </Section>

          {/* Backend params */}
          <Section title={`${form.backend.toUpperCase()} Parameters`}>
            {form.backend === 'nersc' ? (
              <div className="grid grid-cols-2 gap-3">
                <Field label="Queue">
                  <select value={form.nerscQueue} onChange={e => set('nerscQueue', e.target.value)} className="w-full input-base">
                    {NERSC_QUEUES.map(q => <option key={q.value} value={q.value} title={q.desc}>{q.label}</option>)}
                  </select>
                  <p className="text-xs text-muted mt-0.5">{NERSC_QUEUES.find(q => q.value === form.nerscQueue)?.desc}</p>
                </Field>
                <Field label="Account">
                  <TInput value={form.nerscAccount} onChange={v => set('nerscAccount', v)} placeholder="m3792_g" />
                </Field>
                <Field label="Time limit (HH:MM:SS)">
                  <TInput value={form.nerscTimeLimit} onChange={v => set('nerscTimeLimit', v)} placeholder="02:00:00" />
                </Field>
                <Field label={<>Constraint (optional) <Tip text="NERSC node feature constraint, e.g. 'gpu' or 'cpu'. Leave blank to use the queue default." /></>}>
                  <TInput value={form.nerscConstraint} onChange={v => set('nerscConstraint', v)} placeholder="gpu" />
                </Field>
              </div>
            ) : (
              <Field label="Account">
                <TInput value={form.orionAccount} onChange={v => set('orionAccount', v)} placeholder="staff" />
              </Field>
            )}
          </Section>

          {/* Advanced */}
          <div>
            <button onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center gap-1.5 text-xs text-secondary hover:text-primary transition-colors">
              {showAdvanced ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
              Advanced — projector, classifier, param specs, replay
            </button>
            {showAdvanced && (
              <div className="mt-3 space-y-4">
                {/* Projector / classifier dropdowns */}
                <div className="grid grid-cols-2 gap-3">
                  <Field label={<>Projector <Tip text="UMAP model that projects high-dimensional embeddings to 2D for visualisation. 'None' fits a fresh UMAP over all embeddings at job end." /></>}>
                    {auxModels.length > 0 ? (
                      <select value={form.projector} onChange={e => set('projector', e.target.value)} className="w-full input-base">
                        <option value="">— none —</option>
                        {!auxModels.find(m => m.name === 'bnl-nsls2-smi-umap') && (
                          <option value="bnl-nsls2-smi-umap">bnl-nsls2-smi-umap (default)</option>
                        )}
                        {auxModels.map(m => <option key={m.name} value={m.name}>{m.name} (v{m.latest_version})</option>)}
                      </select>
                    ) : (
                      <TInput value={form.projector} onChange={v => set('projector', v)} placeholder="bnl-nsls2-smi-umap" />
                    )}
                  </Field>
                  <Field label={<>Classifier <Tip text="Model that assigns a cluster label to each embedding. 'None' skips classification — no labels are written to the output." /></>}>
                    {auxModels.length > 0 ? (
                      <select value={form.classifier} onChange={e => set('classifier', e.target.value)} className="w-full input-base">
                        <option value="">— none —</option>
                        {!auxModels.find(m => m.name === 'bnl-nsls2-smi-class5') && (
                          <option value="bnl-nsls2-smi-class5">bnl-nsls2-smi-class5 (default)</option>
                        )}
                        {auxModels.map(m => <option key={m.name} value={m.name}>{m.name} (v{m.latest_version})</option>)}
                      </select>
                    ) : (
                      <TInput value={form.classifier} onChange={v => set('classifier', v)} placeholder="bnl-nsls2-smi-class5" />
                    )}
                  </Field>
                </div>

                {/* Param specs */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <label className="text-xs text-secondary flex items-center gap-1">
                      Experimental Parameter Specs
                      <Tip text="Scalar parameters recorded alongside each embedding (e.g. photon energy)." />
                    </label>
                    <button onClick={addParam} className="flex items-center gap-1 text-xs text-indigo-500 hover:text-indigo-400">
                      <Plus className="h-3.5 w-3.5" /> Add
                    </button>
                  </div>
                  {form.paramSpecs.length > 0 && (
                    <div className="grid grid-cols-4 gap-1 text-xs text-muted px-1 mb-1">
                      <span>Name</span><span>Source</span><span>Type</span><span>Units</span>
                    </div>
                  )}
                  <div className="space-y-1.5">
                    {form.paramSpecs.map((p, i) => (
                      <div key={i} className="flex items-center gap-1.5">
                        <div className="grid grid-cols-4 gap-1 flex-1">
                          <TInput value={p.name} onChange={v => updateParam(i, 'name', v)} placeholder="energy" />
                          <TInput value={p.source} onChange={v => updateParam(i, 'source', v)} placeholder="primary/energy" />
                          <select value={p.dtype} onChange={e => updateParam(i, 'dtype', e.target.value)} className="input-base">
                            {['float', 'int', 'str'].map(t => <option key={t} value={t}>{t}</option>)}
                          </select>
                          <TInput value={p.units} onChange={v => updateParam(i, 'units', v)} placeholder="eV" />
                        </div>
                        <button onClick={() => removeParam(i)} className="text-muted hover:text-red-400">
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    ))}
                    {form.paramSpecs.length === 0 && (
                      <p className="text-xs text-muted italic px-1">No param specs configured.</p>
                    )}
                  </div>
                </div>

                {/* Replay existing */}
                <div className="flex items-center gap-2">
                  <input type="checkbox" id="replay" checked={form.replayExisting}
                    onChange={e => set('replayExisting', e.target.checked)}
                    className="rounded border-theme" />
                  <label htmlFor="replay" className="text-xs text-secondary flex items-center gap-1">
                    Replay existing runs on startup
                    <Tip text="If checked, runs already in the inputs container when the watcher starts will also trigger a streaming job. Uncheck for new-data-only mode." />
                  </label>
                </div>
              </div>
            )}
          </div>

          <hr className="border-theme" />

          {/* Submit */}
          <button onClick={handleStart} disabled={submitting}
            className="w-full flex items-center justify-center gap-2 rounded-lg bg-violet-600 hover:bg-violet-500 disabled:bg-violet-900 disabled:text-violet-400 text-white font-medium py-2.5 text-sm transition-colors">
            {submitting
              ? <><Loader2 className="h-4 w-4 animate-spin" /> Starting watcher…</>
              : <><Radio className="h-4 w-4" /> Start Live Watch on {form.backend.toUpperCase()}</>}
          </button>

          {error && (
            <div className="flex items-start gap-2 rounded-lg bg-red-50 dark:bg-red-950/50 border border-red-200 dark:border-red-800 px-3 py-2.5">
              <XCircle className="h-4 w-4 text-red-500 mt-0.5 flex-shrink-0" />
              <p className="text-xs text-red-700 dark:text-red-300 font-mono">{error}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="space-y-3">
      <h3 className="text-xs font-semibold text-muted uppercase tracking-wide">{title}</h3>
      {children}
    </div>
  )
}

function Field({ label, children }: { label?: React.ReactNode; children: React.ReactNode }) {
  return (
    <div>
      {label !== undefined && <label className="flex items-center gap-1 text-xs text-secondary mb-1">{label}</label>}
      {children}
    </div>
  )
}

function TInput({ value, onChange, placeholder }: { value: string; onChange: (v: string) => void; placeholder?: string }) {
  return (
    <input type="text" value={value} onChange={e => onChange(e.target.value)} placeholder={placeholder}
      className="w-full input-base placeholder:text-muted" />
  )
}

function Tip({ text }: { text: string }) {
  return (
    <span title={text} className="cursor-help text-muted hover:text-secondary transition-colors">
      <Info className="h-3 w-3" />
    </span>
  )
}
