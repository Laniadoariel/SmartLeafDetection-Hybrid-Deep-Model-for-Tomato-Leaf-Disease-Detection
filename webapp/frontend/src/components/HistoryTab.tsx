import { useState, useEffect } from 'react'
import api from '../api'
import type { FlightDetail, FlightSummary } from '../api'

interface Props { onSelect: (f: FlightDetail) => void }

export default function HistoryTab({ onSelect }: Props) {
  const [flights, setFlights] = useState<FlightSummary[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api.get('/flights/history').then(r => {
      setFlights(r.data)
      setLoading(false)
    }).catch(() => setLoading(false))
  }, [])

  const openFlight = async (id: string) => {
    const { data } = await api.get(`/flights/${id}`)
    onSelect(data)
  }

  if (loading) return <div style={styles.empty}>Loading...</div>
  if (flights.length === 0) {
    return (
      <div style={styles.empty}>
        <div style={{ fontSize:48 }}>📂</div>
        <h3>No flight history</h3>
        <p style={{ color:'var(--gray-400)' }}>Upload your first drone video to get started</p>
      </div>
    )
  }

  return (
    <div>
      <h3 style={{ marginBottom:16 }}>Flight History ({flights.length})</h3>
      <div style={{ display:'flex', flexDirection:'column', gap:12 }}>
        {flights.map(f => (
          <div key={f.id} onClick={() => openFlight(f.id)} style={styles.row}>
            <div style={{ flex:1 }}>
              <div style={{ fontWeight:600, fontSize:14 }}>{f.video_filename}</div>
              <div style={{ fontSize:12, color:'var(--gray-400)', marginTop:2 }}>
                {new Date(f.created_at).toLocaleString()} • {f.total_frames} frames
              </div>
            </div>
            <div style={{ display:'flex', gap:12, alignItems:'center' }}>
              <StatusBadge status={f.status} />
              {f.status === 'completed' && (
                <div style={{ textAlign:'right', fontSize:12 }}>
                  <div style={{ color:'var(--green-600)' }}>✅ {f.healthy_plants} healthy</div>
                  {f.diseased_plants > 0 && (
                    <div style={{ color:'var(--red-500)' }}>⚠️ {f.diseased_plants} diseased</div>
                  )}
                </div>
              )}
              <span style={{ color:'var(--gray-400)', fontSize:18 }}>→</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, { bg:string; fg:string }> = {
    completed: { bg:'#f0fdf4', fg:'#16a34a' },
    processing: { bg:'#fef3c7', fg:'#92400e' },
    failed: { bg:'#fef2f2', fg:'#dc2626' },
    uploaded: { bg:'var(--gray-100)', fg:'var(--gray-500)' },
  }
  const c = colors[status] || colors.uploaded
  return (
    <span style={{
      padding:'3px 10px', borderRadius:10, fontSize:11, fontWeight:600,
      background:c.bg, color:c.fg,
    }}>
      {status}
    </span>
  )
}

const styles: Record<string, React.CSSProperties> = {
  empty: { textAlign:'center', padding:80, background:'#fff', borderRadius:16 },
  row: {
    display:'flex', alignItems:'center', padding:'16px 20px',
    background:'#fff', borderRadius:12, boxShadow:'var(--shadow)',
    cursor:'pointer', transition:'box-shadow .15s',
  },
}
