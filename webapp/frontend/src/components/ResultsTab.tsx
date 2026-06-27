import type { FlightDetail, PlantResult } from '../api'

interface Props { flight: FlightDetail | null }

const DISEASE_COLORS: Record<string, string> = {
  healthy: '#22c55e', 'Bacterial Spot': '#ef4444', 'Early_Blight': '#f97316',
  'Late_blight': '#dc2626', 'Leaf Mold': '#a855f7', 'Target_Spot': '#3b82f6',
  'Tomato leaf late blight': '#dc2626', 'Tomato Early blight leaf': '#f97316',
  'Tomato leaf bacterial spot': '#ef4444', 'Tomato leaf': '#22c55e',
  'Tomato mold leaf': '#a855f7', 'Tomato Septoria leaf spot': '#ec4899',
  'black spot': '#1f2937', Healthy: '#22c55e',
  // Canonical classes emitted by the dedicated disease classifier (snake_case)
  bacterial_spot: '#ef4444', early_blight: '#f97316', late_blight: '#dc2626',
  leaf_mold: '#a855f7', septoria_leaf_spot: '#ec4899', target_spot: '#3b82f6',
  spider_mites: '#0ea5e9', mosaic_virus: '#8b5cf6', yellow_leaf_curl_virus: '#eab308',
  powdery_mildew: '#64748b', black_spot: '#1f2937',
}

export default function ResultsTab({ flight }: Props) {
  if (!flight || flight.plants.length === 0) {
    return (
      <div style={styles.empty}>
        <div style={{ fontSize:48 }}>📊</div>
        <h3>No results yet</h3>
        <p style={{ color:'var(--gray-400)' }}>Process a video to see detection results</p>
      </div>
    )
  }

  const diseased = flight.plants.filter(p => p.status === 'diseased')
  const healthy = flight.plants.filter(p => p.status === 'healthy')

  // Disease distribution
  const diseaseCounts: Record<string, number> = {}
  for (const p of diseased) {
    for (const l of p.disease_labels) {
      diseaseCounts[l] = (diseaseCounts[l] || 0) + 1
    }
  }

  return (
    <div>
      {/* Summary cards */}
      <div style={{ display:'grid', gridTemplateColumns:'repeat(4,1fr)', gap:16, marginBottom:24 }}>
        <SummaryCard label="Leaves Inspected" value={flight.total_plants} icon="🍃" color="#16a34a" />
        <SummaryCard label="Healthy" value={healthy.length} icon="✅" color="#22c55e" />
        <SummaryCard label="Diseased" value={diseased.length} icon="⚠️" color="#dc2626" />
        <SummaryCard label="Frames Analyzed" value={flight.total_frames} icon="🎞️" color="#6366f1" />
      </div>

      {/* Disease distribution */}
      {Object.keys(diseaseCounts).length > 0 && (
        <div style={styles.card}>
          <h3 style={{ margin:'0 0 16px' }}>Disease Distribution</h3>
          <div style={{ display:'flex', gap:12, flexWrap:'wrap' }}>
            {Object.entries(diseaseCounts).sort((a,b) => b[1] - a[1]).map(([label, count]) => (
              <div key={label} style={{
                padding:'10px 16px', borderRadius:10,
                background: (DISEASE_COLORS[label] || '#6b7280') + '15',
                border: `1px solid ${DISEASE_COLORS[label] || '#6b7280'}30`,
              }}>
                <div style={{ fontSize:20, fontWeight:700, color: DISEASE_COLORS[label] || '#6b7280' }}>{count}</div>
                <div style={{ fontSize:12, color:'var(--gray-600)' }}>{label}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* How disease was inferred */}
      <div style={{ ...styles.card, marginTop:16 }}>
        <h3 style={{ margin:'0 0 12px', color:'var(--green-700)' }}>🔬 How Disease Was Inferred</h3>
        <div style={{ fontSize:13, color:'var(--gray-600)', lineHeight:1.7 }}>
          <p>1. The YOLOv11 leaf detector located each leaf and tracked it across frames (BoT-SORT with Global Motion Compensation, stable per-leaf IDs)</p>
          <p>2. Each tracked leaf crop was normalized to 224×224 (ImageNet) and classified by the dedicated ResNet50 disease classifier</p>
          <p>3. Per-frame class probabilities were aggregated over each leaf's track (confidence-weighted majority vote)</p>
          <p>4. The aggregated label is the leaf's final disease prediction; one result card = one tracked leaf</p>
          <p>5. The YOLO model is used only for localization/tracking — the disease prediction comes entirely from the image classifier</p>
        </div>
      </div>

      {/* Why fewer leaves than per-frame detections */}
      <div style={{ ...styles.card, marginTop:16, background:'#f8fafc', border:'1px solid var(--gray-200)' }}>
        <h3 style={{ margin:'0 0 12px', color:'var(--green-700)' }}>🍃 Why fewer leaves than detections?</h3>
        <div style={{ fontSize:13, color:'var(--gray-600)', lineHeight:1.7 }}>
          <p>
            A single frame in the Investigation tab can show many leaf <em>detections</em> — but a
            detection is just one leaf seen in one frame. The same physical leaf is detected again
            and again across consecutive frames.
          </p>
          <p>
            BoT-SORT links those repeated detections into <strong>one tracked leaf</strong> with a
            stable ID. So a leaf appearing in 8 frames produces 8 detections but only{' '}
            <strong>one</strong> result card here.
          </p>
          <p>
            Each result is therefore <strong>one unique leaf</strong>, inspected from every frame it
            appeared in — not one detection. A leaf must be seen in at least 2 frames to count, which
            filters out one-frame false positives. That is why the final leaf count is smaller than
            the number of per-frame detections.
          </p>
        </div>
      </div>

      {/* Leaf cards */}
      <h3 style={{ margin:'24px 0 12px' }}>Leaf Results ({flight.plants.length})</h3>
      <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fill,minmax(300px,1fr))', gap:16 }}>
        {flight.plants.map(p => <LeafCard key={p.id} plant={p} />)}
      </div>
    </div>
  )
}

function SummaryCard({ label, value, icon, color }: { label:string; value:number; icon:string; color:string }) {
  return (
    <div style={{ ...styles.card, textAlign:'center' }}>
      <div style={{ fontSize:28 }}>{icon}</div>
      <div style={{ fontSize:28, fontWeight:700, color, marginTop:4 }}>{value}</div>
      <div style={{ fontSize:12, color:'var(--gray-500)' }}>{label}</div>
    </div>
  )
}

function LeafCard({ plant }: { plant: PlantResult }) {
  const isDiseased = plant.status === 'diseased'
  return (
    <div style={{
      ...styles.card,
      borderLeft: `4px solid ${isDiseased ? '#dc2626' : '#22c55e'}`,
    }}>
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center' }}>
        <h4 style={{ margin:0 }}>Leaf #{plant.plant_id}</h4>
        <span style={{
          padding:'3px 10px', borderRadius:10, fontSize:11, fontWeight:600,
          background: isDiseased ? '#fef2f2' : '#f0fdf4',
          color: isDiseased ? '#dc2626' : '#16a34a',
        }}>
          {plant.status.toUpperCase()}
        </span>
      </div>
      {isDiseased && plant.disease_labels.length > 0 && (
        <div style={{ marginTop:8, display:'flex', gap:6, flexWrap:'wrap' }}>
          {plant.disease_labels.map(l => (
            <span key={l} style={{
              padding:'2px 8px', borderRadius:6, fontSize:11,
              background:'#fef2f2', color:'#dc2626',
            }}>{l}</span>
          ))}
        </div>
      )}
      <div style={{ marginTop:10, fontSize:12, color:'var(--gray-500)', display:'flex', gap:16, flexWrap:'wrap' }}>
        <span>Confidence: {(plant.confidence * 100).toFixed(0)}%</span>
        {plant.views_total > 0 && <span>Seen in {plant.views_total} frame{plant.views_total > 1 ? 's' : ''}</span>}
      </div>
      {plant.gps_lat && (
        <div style={{ marginTop:6, fontSize:11, color:'var(--gray-400)' }}>
          📍 {plant.gps_lat?.toFixed(6)}, {plant.gps_lon?.toFixed(6)}
        </div>
      )}
      {/* Leaf crops */}
      {plant.leaves.length > 0 && (
        <div style={{ marginTop:10, display:'flex', gap:6, flexWrap:'wrap' }}>
          {plant.leaves.slice(0, 4).map((l, i) => (
            l.crop_path && (
              <img key={i}
                src={`/files/output/${l.crop_path}`}
                alt={l.label}
                style={{ width:60, height:60, objectFit:'cover', borderRadius:6, border:'1px solid var(--gray-200)' }}
                onError={e => { (e.target as HTMLImageElement).style.display = 'none' }}
              />
            )
          ))}
        </div>
      )}
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  empty: { textAlign:'center', padding:80, background:'#fff', borderRadius:16 },
  card: { background:'#fff', borderRadius:12, padding:20, boxShadow:'var(--shadow)' },
}
