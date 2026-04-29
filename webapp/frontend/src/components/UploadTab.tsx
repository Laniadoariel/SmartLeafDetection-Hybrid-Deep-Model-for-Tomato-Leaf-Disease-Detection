import { useState, useRef, useEffect } from 'react'
import api from '../api'
import type { FlightDetail, FlightSummary } from '../api'

const STAGES = [
  'Video uploaded', 'Video decoded', 'Frames extracted',
  'Plant detection started', 'Plant tracking started', 'Plant ROI cropping',
  'Leaf detection started', 'Leaf tracking started', 'Leaf ROI extraction',
  'Disease classification', 'Temporal aggregation', 'Plant-level inference',
  'Results saved', 'Analysis completed',
]

interface Props { onFlightReady: (f: FlightDetail) => void }

export default function UploadTab({ onFlightReady }: Props) {
  const [file, setFile] = useState<File | null>(null)
  const [flight, setFlight] = useState<FlightSummary | null>(null)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState('')
  const fileRef = useRef<HTMLInputElement>(null)
  const pollRef = useRef<number | null>(null)

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    const f = e.dataTransfer.files[0]
    if (f) setFile(f)
  }

  const upload = async () => {
    if (!file) return
    setUploading(true); setError('')
    try {
      const fd = new FormData()
      fd.append('file', file)
      const { data } = await api.post('/flights/upload', fd)
      setFlight(data)
      // Start analysis
      await api.post(`/flights/${data.id}/start`)
      startPolling(data.id)
    } catch (e: any) {
      const detail = e?.response?.data?.detail
      const status = e?.response?.status
      setError(detail || `Upload failed (${status || e?.message || 'unknown error'})`)
      setUploading(false)
    }
  }

  const startPolling = (id: string) => {
    pollRef.current = window.setInterval(async () => {
      try {
        const { data } = await api.get(`/flights/${id}`)
        setFlight(data)
        if (data.status === 'completed' || data.status === 'failed') {
          clearInterval(pollRef.current!)
          setUploading(false)
          if (data.status === 'completed') onFlightReady(data)
        }
      } catch { /* ignore polling errors */ }
    }, 1500)
  }

  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current) }, [])

  const stageIdx = flight ? STAGES.findIndex(s =>
    (flight.current_stage || '').toLowerCase().includes(s.toLowerCase().split(' ')[0])
  ) : -1

  return (
    <div>
      {/* Upload area */}
      {!flight && (
        <div onDrop={handleDrop} onDragOver={e => e.preventDefault()}
          onClick={() => fileRef.current?.click()}
          style={styles.dropzone}>
          <input ref={fileRef} type="file" accept=".mp4,.mov,.avi,.mkv"
            style={{ display:'none' }} onChange={e => setFile(e.target.files?.[0] || null)} />
          <div style={{ fontSize:48, marginBottom:8 }}>📹</div>
          <p style={{ fontWeight:600, fontSize:16 }}>
            {file ? file.name : 'Drop drone video here or click to browse'}
          </p>
          {file && <p style={{ fontSize:13, color:'var(--gray-400)' }}>
            {(file.size / 1024 / 1024).toFixed(1)} MB • {file.type || file.name.split('.').pop()}
          </p>}
          <p style={{ fontSize:12, color:'var(--gray-400)', marginTop:8 }}>
            Supported: MP4, MOV, AVI, MKV
          </p>
        </div>
      )}

      {file && !flight && (
        <div style={{ textAlign:'center', marginTop:16 }}>
          <button onClick={upload} disabled={uploading} style={styles.startBtn}>
            {uploading ? 'Uploading...' : '🚀 Start Analysis'}
          </button>
        </div>
      )}

      {error && <div style={styles.error}>{error}</div>}

      {/* Processing timeline */}
      {flight && (
        <div style={styles.timeline}>
          <h3 style={{ marginBottom:16, color:'var(--green-700)' }}>
            Processing: {flight.video_filename}
          </h3>
          <div style={styles.progressBar}>
            <div style={{ ...styles.progressFill, width: `${flight.progress}%` }} />
          </div>
          <p style={{ fontSize:13, color:'var(--gray-500)', margin:'8px 0 20px' }}>
            {flight.current_stage} — {Math.round(flight.progress)}%
            {flight.processed_frames > 0 && ` (${flight.processed_frames}/${flight.total_frames} frames)`}
          </p>
          <div style={{ display:'flex', flexDirection:'column', gap:6 }}>
            {STAGES.map((s, i) => {
              const done = i <= stageIdx || flight.status === 'completed'
              const active = i === stageIdx + 1 && flight.status === 'processing'
              return (
                <div key={s} style={{ display:'flex', alignItems:'center', gap:10 }}>
                  <span style={{
                    width:24, height:24, borderRadius:'50%', display:'flex',
                    alignItems:'center', justifyContent:'center', fontSize:12, fontWeight:600,
                    background: done ? 'var(--green-500)' : active ? '#fbbf24' : 'var(--gray-200)',
                    color: done ? '#fff' : active ? '#92400e' : 'var(--gray-400)',
                    transition: 'all .3s',
                  }}>
                    {done ? '✓' : i + 1}
                  </span>
                  <span style={{
                    fontSize:13, fontWeight: active ? 600 : 400,
                    color: done ? 'var(--green-700)' : active ? '#92400e' : 'var(--gray-400)',
                  }}>{s}</span>
                  {active && <span style={styles.badge}>Running</span>}
                  {done && <span style={{ ...styles.badge, background:'var(--green-100)', color:'var(--green-700)' }}>Done</span>}
                </div>
              )
            })}
          </div>
          {flight.status === 'failed' && (
            <div style={{ ...styles.error, marginTop:16 }}>
              Analysis failed: {(flight as any).error_message || 'Unknown error'}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  dropzone: {
    border:'2px dashed var(--gray-300)', borderRadius:16, padding:'60px 40px',
    textAlign:'center', cursor:'pointer', background:'#fff',
    transition:'border-color .2s',
  },
  startBtn: {
    padding:'14px 40px', background:'var(--green-600)', color:'#fff',
    fontSize:16, fontWeight:600, borderRadius:10,
  },
  error: { background:'#fef2f2', color:'#dc2626', padding:'10px 14px', borderRadius:8, fontSize:13, marginTop:12 },
  timeline: { background:'#fff', borderRadius:16, padding:28, marginTop:20, boxShadow:'var(--shadow)' },
  progressBar: { height:8, background:'var(--gray-200)', borderRadius:4, overflow:'hidden' },
  progressFill: { height:'100%', background:'var(--green-500)', borderRadius:4, transition:'width .5s' },
  badge: {
    fontSize:10, fontWeight:600, padding:'2px 8px', borderRadius:10,
    background:'#fef3c7', color:'#92400e',
  },
}
