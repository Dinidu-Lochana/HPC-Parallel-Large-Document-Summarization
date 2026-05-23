import { useState, useRef, useEffect } from 'react'
import {
  Cpu, Upload, Zap, Loader, CheckCircle, AlertCircle,
  BarChart3, Clock, TrendingUp, Activity, Server, FileText,
} from 'lucide-react'

const BACKEND = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000'

/* ── methods ──────────────────────────────────────────────────────────────── */
const METHODS = [
  {
    id:    'serial',
    label: 'Serial',
    tag:   'Baseline',
    desc:  'Single-core sequential — use as speedup reference',
    icon:  <Clock size={14} />,
    ring:  'ring-gray-500',
    activeBg:   'bg-gray-500/10 border-gray-500',
    activeText: 'text-gray-300',
    badge:      'bg-gray-500/20 text-gray-300',
    slider:     'accent-gray-500',
    val:        'text-gray-300',
  },
  {
    id:    'mpi',
    label: 'MPI',
    tag:   'Distributed',
    desc:  'Dynamic load balancing across processes',
    icon:  <Server size={14} />,
    ring:  'ring-blue-500',
    activeBg:   'bg-blue-500/10 border-blue-500',
    activeText: 'text-blue-400',
    badge:      'bg-blue-500/20 text-blue-400',
    slider:     'accent-blue-500',
    val:        'text-blue-400',
  },
  {
    id:    'openmp',
    label: 'OpenMP',
    tag:   'Threaded',
    desc:  'Parallel threads on shared memory',
    icon:  <Activity size={14} />,
    ring:  'ring-emerald-500',
    activeBg:   'bg-emerald-500/10 border-emerald-500',
    activeText: 'text-emerald-400',
    badge:      'bg-emerald-500/20 text-emerald-400',
    slider:     'accent-emerald-500',
    val:        'text-emerald-400',
  },
  {
    id:    'hybrid',
    label: 'Hybrid',
    tag:   'MPI + OMP',
    desc:  'MPI nodes with OMP threads each',
    icon:  <TrendingUp size={14} />,
    ring:  'ring-violet-500',
    activeBg:   'bg-violet-500/10 border-violet-500',
    activeText: 'text-violet-400',
    badge:      'bg-violet-500/20 text-violet-400',
    slider:     'accent-violet-500',
    val:        'text-violet-400',
  },
]

/* ── metric card (light panel) ────────────────────────────────────────────── */
const CARD_THEMES = {
  blue:    'bg-blue-50   border-blue-100   ',
  amber:   'bg-amber-50  border-amber-100  ',
  emerald: 'bg-emerald-50 border-emerald-100',
  violet:  'bg-violet-50 border-violet-100 ',
}

function MetricCard({ icon, label, value, unit, sub, color = 'blue' }) {
  return (
    <div className={`rounded-xl border p-4 ${CARD_THEMES[color]}`}>
      <div className="flex items-center gap-1.5 text-gray-400 text-[10px] font-semibold uppercase tracking-widest mb-2">
        {icon}{label}
      </div>
      <p className="text-[1.65rem] font-bold text-gray-900 leading-none">
        {value ?? '—'}
        <span className="text-sm font-normal text-gray-400 ml-1">{unit}</span>
      </p>
      {sub && <p className="text-[11px] text-gray-400 mt-1">{sub}</p>}
    </div>
  )
}

/* ── section label for dark sidebar ──────────────────────────────────────── */
function SideLabel({ children }) {
  return (
    <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-widest mb-2">
      {children}
    </p>
  )
}

/* ── app ──────────────────────────────────────────────────────────────────── */
export default function App() {
  const [file,      setFile]      = useState(null)
  const [topic,     setTopic]     = useState('')
  const [method,    setMethod]    = useState('mpi')
  const [processes, setProcesses] = useState(4)
  const [threads,   setThreads]   = useState(4)
  const [loading,   setLoading]   = useState(false)
  const [result,    setResult]    = useState(null)
  const [error,     setError]     = useState('')
  const [dragging,  setDragging]  = useState(false)
  const [backendOk, setBackendOk] = useState(null)
  const fileRef = useRef()

  useEffect(() => {
    fetch(`${BACKEND}/health`, { signal: AbortSignal.timeout(3000) })
      .then(r => setBackendOk(r.ok))
      .catch(() => setBackendOk(false))
  }, [])

  const handleFile = (f) => {
    if (!f) return
    if (!f.name.endsWith('.pdf') && !f.name.endsWith('.txt')) {
      setError('Only PDF or TXT files are supported.')
      return
    }
    setFile(f)
    setError('')
  }

  const handleSubmit = async () => {
    if (!file)         { setError('Please upload a document.'); return }
    if (!topic.trim()) { setError('Please enter a topic.'); return }
    setLoading(true); setError(''); setResult(null)

    const form = new FormData()
    form.append('file', file)
    form.append('topic', topic.trim())
    form.append('method', method)
    form.append('processes', method === 'serial' ? 1 : processes)
    form.append('threads',   method === 'serial' ? 1 : threads)

    try {
      const res  = await fetch(`${BACKEND}/summarize`, { method: 'POST', body: form })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || `Server error ${res.status}`)
      setResult(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const active     = METHODS.find(m => m.id === method)
  const rm         = result?.metrics || {}
  const idealUnits = method === 'serial' ? 1 : method === 'mpi' ? processes : method === 'openmp' ? threads : processes * threads

  return (
    /*
     * h-full instead of h-screen — because #root is already 100vh via CSS.
     * overflow-hidden on every flex ancestor prevents page scroll.
     */
    <div className="flex flex-col h-full overflow-hidden bg-gray-950 font-sans">

      {/* ── Header — dark ─────────────────────────────────────────────────── */}
      <header className="shrink-0 h-12 bg-gray-900 border-b border-gray-800
                         flex items-center justify-between px-5 z-10">
        <div className="flex items-center gap-3">
          <div className="bg-blue-600 p-1.5 rounded-lg shadow-md">
            <Cpu size={16} className="text-white" />
          </div>
          <span className="font-semibold text-white text-sm tracking-tight">
            HPC Document Summarizer
          </span>
          <span className="hidden md:flex items-center gap-1.5 text-gray-600 text-xs">
            <span>·</span>
            <span>Serial</span><span>·</span>
            <span>MPI</span><span>·</span>
            <span>OpenMP</span><span>·</span>
            <span>Hybrid</span><span>·</span>
            <span>Llama 3.3 70B</span>
          </span>
        </div>

        {/* backend status */}
        <div className="flex items-center gap-2 text-xs text-gray-500">
          <span className={`w-1.5 h-1.5 rounded-full ${
            backendOk === null ? 'bg-yellow-500 animate-pulse' :
            backendOk          ? 'bg-green-500' : 'bg-red-500'
          }`} />
          {backendOk === null ? 'Connecting…' : backendOk ? 'Backend ready' : 'Backend offline'}
        </div>
      </header>

      {/* ── Body: sidebar + results ────────────────────────────────────────── */}
      <div className="flex flex-1 overflow-hidden min-h-0">

        {/* ══ Left sidebar — DARK ════════════════════════════════════════════ */}
        <aside className="w-72 shrink-0 flex flex-col overflow-hidden
                          bg-gray-900 border-r border-gray-800">
          {/* scrollable content area inside sidebar */}
          <div className="flex-1 overflow-y-auto flex flex-col gap-5 p-4">

            {/* File upload */}
            <div>
              <SideLabel>Document</SideLabel>
              <div
                onClick={() => fileRef.current.click()}
                onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
                onDragLeave={() => setDragging(false)}
                onDrop={(e) => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]) }}
                className={`border-2 border-dashed rounded-xl p-4 text-center cursor-pointer transition-all
                  ${dragging  ? 'border-blue-500 bg-blue-500/10'
                  : file      ? 'border-emerald-500 bg-emerald-500/10'
                              : 'border-gray-700 bg-gray-800 hover:border-gray-600 hover:bg-gray-750'}`}
              >
                <input ref={fileRef} type="file" accept=".pdf,.txt" className="hidden"
                  onChange={(e) => handleFile(e.target.files[0])} />
                {file ? (
                  <div className="flex items-center justify-center gap-2 text-emerald-400">
                    <FileText size={18} className="shrink-0" />
                    <div className="text-left min-w-0">
                      <p className="font-medium text-sm truncate">{file.name}</p>
                      <p className="text-xs text-gray-500 mt-0.5">{(file.size / 1024).toFixed(1)} KB · click to change</p>
                    </div>
                  </div>
                ) : (
                  <>
                    <Upload className="mx-auto mb-2 text-gray-600" size={22} />
                    <p className="text-sm text-gray-400 font-medium">Drop PDF or TXT</p>
                    <p className="text-xs text-gray-600 mt-0.5">or click to browse</p>
                  </>
                )}
              </div>
            </div>

            {/* Topic */}
            <div>
              <SideLabel>Topic</SideLabel>
              <input
                type="text"
                placeholder="e.g. Machine Learning…"
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                className="w-full bg-gray-800 border border-gray-700 rounded-xl px-3.5 py-2.5 text-sm
                           text-white placeholder-gray-600
                           focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500/30 transition"
              />
            </div>

            {/* Method */}
            <div>
              <SideLabel>Parallel Method</SideLabel>
              <div className="flex flex-col gap-1.5">
                {METHODS.map((mth) => {
                  const isActive = method === mth.id
                  return (
                    <button
                      key={mth.id}
                      onClick={() => setMethod(mth.id)}
                      className={`text-left rounded-xl border px-3.5 py-2.5 transition-all flex items-start gap-2.5
                        ${isActive
                          ? mth.activeBg
                          : 'border-gray-700 bg-gray-800 hover:bg-gray-750 hover:border-gray-600'}`}
                    >
                      <span className={`mt-0.5 shrink-0 ${isActive ? mth.activeText : 'text-gray-600'}`}>
                        {mth.icon}
                      </span>
                      <div className="min-w-0">
                        <div className="flex items-center gap-2">
                          <span className={`font-semibold text-sm ${isActive ? mth.activeText : 'text-gray-300'}`}>
                            {mth.label}
                          </span>
                          <span className={`text-[9px] font-semibold px-1.5 py-0.5 rounded-full
                            ${isActive ? mth.badge : 'bg-gray-700 text-gray-500'}`}>
                            {mth.tag}
                          </span>
                        </div>
                        <p className="text-[11px] text-gray-600 mt-0.5 leading-snug">{mth.desc}</p>
                      </div>
                    </button>
                  )
                })}
              </div>
            </div>

            {/* Sliders */}
            <div className="space-y-4">
              {method === 'serial' && (
                <div className="rounded-xl bg-gray-500/10 border border-gray-500/30 px-3 py-2 text-center">
                  <span className="text-xs text-gray-400">No parallelism — </span>
                  <span className="text-sm font-bold text-gray-300">1 process · 1 thread</span>
                </div>
              )}
              {(method === 'mpi' || method === 'hybrid') && (
                <div>
                  <div className="flex justify-between mb-2">
                    <SideLabel>MPI Processes</SideLabel>
                    <span className={`text-sm font-bold -mt-0.5 ${active.val}`}>{processes}</span>
                  </div>
                  <input type="range" min={2} max={8} value={processes}
                    onChange={(e) => setProcesses(+e.target.value)}
                    className={`w-full h-1 cursor-pointer rounded-full ${active.slider}`} />
                  <div className="flex justify-between text-[10px] text-gray-600 mt-1 font-medium">
                    <span>2</span><span>8</span>
                  </div>
                </div>
              )}
              {(method === 'openmp' || method === 'hybrid') && (
                <div>
                  <div className="flex justify-between mb-2">
                    <SideLabel>OMP Threads / Process</SideLabel>
                    <span className="text-sm font-bold -mt-0.5 text-emerald-400">{threads}</span>
                  </div>
                  <input type="range" min={1} max={8} value={threads}
                    onChange={(e) => setThreads(+e.target.value)}
                    className="w-full h-1 cursor-pointer rounded-full accent-emerald-500" />
                  <div className="flex justify-between text-[10px] text-gray-600 mt-1 font-medium">
                    <span>1</span><span>8</span>
                  </div>
                </div>
              )}
              {method === 'hybrid' && (
                <div className="rounded-xl bg-violet-500/10 border border-violet-500/30 px-3 py-2 text-center">
                  <span className="text-xs text-violet-400">Total parallelism: </span>
                  <span className="text-sm font-bold text-violet-300">
                    {processes} × {threads} = {processes * threads} units
                  </span>
                </div>
              )}
            </div>

            {/* Error */}
            {error && (
              <div className="flex items-start gap-2 rounded-xl bg-red-500/10 border border-red-500/30 p-3 text-red-400 text-xs leading-relaxed">
                <AlertCircle size={13} className="mt-0.5 shrink-0" />
                <span>{error}</span>
              </div>
            )}
          </div>

          {/* Run button — pinned to bottom of sidebar */}
          <div className="shrink-0 p-4 border-t border-gray-800">
            <button
              onClick={handleSubmit}
              disabled={loading || backendOk === false}
              className="w-full py-2.5 rounded-xl font-semibold text-sm transition-all
                bg-blue-600 hover:bg-blue-500 text-white shadow-lg active:scale-[0.98]
                disabled:bg-gray-800 disabled:text-gray-600 disabled:shadow-none disabled:cursor-not-allowed
                flex items-center justify-center gap-2"
            >
              {loading
                ? <><Loader size={14} className="animate-spin" />Running {method.toUpperCase()}…</>
                : <><Zap size={14} />Run Summarizer</>}
            </button>
          </div>
        </aside>

        {/* ══ Right panel — LIGHT ════════════════════════════════════════════ */}
        <main className="flex-1 overflow-y-auto bg-gray-50 min-w-0">
          <div className="p-6 flex flex-col gap-4 min-h-full">

            {/* Empty state */}
            {!result && !loading && (
              <div className="flex-1 flex items-center justify-center min-h-[calc(100vh-12rem)]">
                <div className="text-center">
                  <div className="w-16 h-16 rounded-2xl bg-white border border-gray-200 shadow-sm
                                  flex items-center justify-center mx-auto mb-4">
                    <BarChart3 size={28} className="text-gray-300" />
                  </div>
                  <p className="font-semibold text-gray-500 text-sm">No results yet</p>
                  <p className="text-gray-400 text-xs mt-1">Configure and run the summarizer</p>
                </div>
              </div>
            )}

            {/* Loading state */}
            {loading && (
              <div className="flex-1 flex items-center justify-center min-h-[calc(100vh-12rem)]">
                <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-10 text-center max-w-xs w-full">
                  <Loader size={32} className="animate-spin text-blue-500 mx-auto mb-4" />
                  <p className="font-semibold text-gray-800">Processing…</p>
                  <p className="text-gray-400 text-sm mt-1">
                    Parallel summarization via <strong>{method.toUpperCase()}</strong>
                  </p>
                  {method === 'hybrid' && (
                    <p className="text-violet-500 text-xs mt-2 font-medium">
                      {processes} × {threads} = {processes * threads} units
                    </p>
                  )}
                </div>
              </div>
            )}

            {/* Results */}
            {result && (
              <>
                {/* Badge */}
                <div className="shrink-0 flex items-center gap-2 bg-white border border-emerald-200
                                rounded-xl px-4 py-2.5 shadow-sm">
                  <CheckCircle size={15} className="text-emerald-500 shrink-0" />
                  <span className="text-sm text-gray-700">
                    Completed —&nbsp;
                    <span className="font-semibold text-gray-900">{rm.chunks}</span> chunks ·&nbsp;
                    method:&nbsp;
                    <span className="font-semibold text-gray-900">{(rm.method || '').toUpperCase()}</span>
                  </span>
                </div>

                {/* 4 metric cards */}
                <div className="grid grid-cols-2 xl:grid-cols-4 gap-3 shrink-0">
                  <MetricCard color="blue"    icon={<Clock size={11} />}     label="Execution Time"  value={rm.execution_time?.toFixed(3)}       unit="s"  sub={`Seq. est. ${rm.sequential_estimate?.toFixed(2)} s`} />
                  <MetricCard color="amber"   icon={<TrendingUp size={11}/>}  label="Speedup"         value={rm.speedup?.toFixed(2)}              unit="×"  sub={`Ideal: ${idealUnits}×`} />
                  <MetricCard color="emerald" icon={<Activity size={11} />}   label="Efficiency"      value={rm.efficiency?.toFixed(1)}           unit="%"  sub={`Scalability ${rm.scalability?.toFixed(1)}%`} />
                  <MetricCard color="violet"  icon={<Cpu size={11} />}        label="CPU Utilization" value={rm.resource_utilization?.toFixed(1)} unit="%"  sub={`${rm.cpu_cores ?? '?'} cores`} />
                </div>

                {/* Details + Summary side by side if wide enough, stacked otherwise */}
                <div className="grid grid-cols-1 xl:grid-cols-5 gap-4 shrink-0">

                  {/* Details table */}
                  <div className="xl:col-span-2 bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
                    <div className="px-4 py-2.5 border-b border-gray-100 bg-gray-50">
                      <span className="text-[10px] font-semibold text-gray-400 uppercase tracking-widest">
                        All Metrics
                      </span>
                    </div>
                    <table className="w-full">
                      <tbody>
                        {[
                          ['Method',              (rm.method || '—').toUpperCase()],
                          ['MPI Processes',        rm.processes ?? '—'],
                          ['OMP Threads',          rm.threads ?? '—'],
                          ['Chunks',               rm.chunks ?? '—'],
                          ['Exec Time',           `${rm.execution_time?.toFixed(3) ?? '—'} s`],
                          ['Seq Estimate',        `${rm.sequential_estimate?.toFixed(3) ?? '—'} s`],
                          ['Speedup',             `${rm.speedup?.toFixed(2) ?? '—'}×`],
                          ['Efficiency',          `${rm.efficiency?.toFixed(1) ?? '—'}%`],
                          ['Scalability',         `${rm.scalability?.toFixed(1) ?? '—'}%`],
                          ['CPU Cores',            rm.cpu_cores ?? '—'],
                          ['CPU Util',            `${rm.resource_utilization?.toFixed(1) ?? '—'}%`],
                        ].map(([k, v]) => (
                          <tr key={k} className="border-t border-gray-100 hover:bg-gray-50 transition-colors">
                            <td className="px-4 py-2 text-[11px] text-gray-400 font-medium">{k}</td>
                            <td className="px-4 py-2 text-[11px] text-gray-900 font-mono font-semibold text-right">{v}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {/* Final summary */}
                  <div className="xl:col-span-3 bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden flex flex-col">
                    <div className="px-4 py-2.5 border-b border-gray-100 bg-gray-50 flex items-center gap-2 shrink-0">
                      <span className="text-[10px] font-semibold text-gray-400 uppercase tracking-widest">
                        Final Summary
                      </span>
                      <span className="text-[10px] text-gray-300">· Llama 3.3 70B via Groq</span>
                    </div>
                    <div className="p-4 overflow-y-auto flex-1">
                      <p className="text-gray-700 text-sm leading-relaxed whitespace-pre-wrap select-text">
                        {result.summary || 'No summary generated.'}
                      </p>
                    </div>
                  </div>
                </div>
              </>
            )}
          </div>
        </main>
      </div>
    </div>
  )
}
