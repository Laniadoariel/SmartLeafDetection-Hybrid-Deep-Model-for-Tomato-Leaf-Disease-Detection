import { useState } from 'react'
import type { FlightDetail } from '../api'

interface Props { flight: FlightDetail | null }

const EXPLANATIONS: Record<string, string> = {
  original: 'Raw frame extracted from the uploaded drone video. Each frame captures a section of the tomato field from above.',
  annotated: 'The YOLOv11 leaf detector locates and tracks each individual leaf (stable per-leaf IDs via BoT-SORT with Global Motion Compensation). Every tracked leaf crop is then passed to a dedicated ResNet50 disease classifier; boxes are drawn with the predicted disease class and confidence.',
  pipeline: 'Actual pipeline: Frame → Leaf Detection (YOLOv11) → Leaf Tracking (BoT-SORT + GMC) → Leaf Crop Extraction → Normalization (224×224, ImageNet) → Disease Classification (dedicated ResNet50 classifier) → Temporal Aggregation per tracked leaf → Final disease prediction. The YOLO model only localizes/tracks leaves; the disease prediction comes from the ResNet50 classifier.',
}

export default function InvestigationTab({ flight }: Props) {
  const [frameIdx, setFrameIdx] = useState(0)
  const [showAnnotated, setShowAnnotated] = useState(true)

  if (!flight || flight.frames.length === 0) {
    return (
      <div style={styles.empty}>
        <div style={{ fontSize:48 }}>🔍</div>
        <h3>No frames to investigate</h3>
        <p style={{ color:'var(--gray-400)' }}>Upload and process a video first</p>
      </div>
    )
  }

  const frame = flight.frames[frameIdx]
  const imgSrc = showAnnotated && frame.annotated_path
    ? `/files/output/${frame.annotated_path}`
    : `/files/output/${frame.original_path}`

  return (
    <div style={{ display:'grid', gridTemplateColumns:'1fr 320px', gap:20 }}>
      {/* Main frame viewer */}
      <div>
        <div style={styles.card}>
          <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:12 }}>
            <h3 style={{ margin:0 }}>Frame {frame.frame_index}</h3>
            <div style={{ display:'flex', gap:8 }}>
              <button onClick={() => setShowAnnotated(false)}
                style={{ ...styles.toggleBtn, ...(showAnnotated ? {} : styles.toggleActive) }}>
                Original
              </button>
              <button onClick={() => setShowAnnotated(true)}
                style={{ ...styles.toggleBtn, ...(showAnnotated ? styles.toggleActive : {}) }}>
                Detections
              </button>
            </div>
          </div>
          <img src={imgSrc} alt={`Frame ${frame.frame_index}`}
            style={{ display:'block', margin:'0 auto', maxHeight:'70vh', maxWidth:'100%',
                     objectFit:'contain', borderRadius:8, background:'var(--gray-100)' }}
            onError={e => { (e.target as HTMLImageElement).src = '' }} />
          <div style={{ display:'flex', gap:16, marginTop:12, fontSize:13, color:'var(--gray-500)' }}>
            <span>🍃 {frame.leaf_count} leaf detections</span>
            <span>🪴 {frame.plant_count} tracked leaves</span>
          </div>
        </div>

        {/* Frame timeline */}
        <div style={{ ...styles.card, marginTop:16 }}>
          <h4 style={{ margin:'0 0 10px' }}>Frame Timeline ({flight.frames.length} frames)</h4>
          <div style={{ display:'flex', gap:4, flexWrap:'wrap' }}>
            {flight.frames.map((f, i) => (
              <button key={f.frame_index} onClick={() => setFrameIdx(i)}
                style={{
                  width:36, height:36, borderRadius:6, fontSize:11, fontWeight:500,
                  background: i === frameIdx ? 'var(--green-500)' : 'var(--gray-100)',
                  color: i === frameIdx ? '#fff' : 'var(--gray-600)',
                }}>
                {i + 1}
              </button>
            ))}
          </div>
          <div style={{ display:'flex', gap:8, marginTop:12 }}>
            <button disabled={frameIdx === 0} onClick={() => setFrameIdx(i => i - 1)}
              style={styles.navBtn}>← Prev</button>
            <button disabled={frameIdx === flight.frames.length - 1}
              onClick={() => setFrameIdx(i => i + 1)} style={styles.navBtn}>Next →</button>
          </div>
        </div>
      </div>

      {/* Side explanation panel */}
      <div>
        <div style={styles.card}>
          <h4 style={{ margin:'0 0 12px', color:'var(--green-700)' }}>🧠 How It Works</h4>
          <div style={styles.explainSection}>
            <h5 style={styles.explainTitle}>Frame Extraction</h5>
            <p style={styles.explainText}>{EXPLANATIONS.original}</p>
          </div>
          <div style={styles.explainSection}>
            <h5 style={styles.explainTitle}>Leaf Detection + Disease Classification</h5>
            <p style={styles.explainText}>{EXPLANATIONS.annotated}</p>
          </div>
          <div style={styles.explainSection}>
            <h5 style={styles.explainTitle}>Full Pipeline</h5>
            <p style={styles.explainText}>{EXPLANATIONS.pipeline}</p>
          </div>
        </div>

        <div style={{ ...styles.card, marginTop:16 }}>
          <h4 style={{ margin:'0 0 8px' }}>📊 Frame Stats</h4>
          <div style={styles.stat}>
            <span>Total Frames</span><span style={{ fontWeight:600 }}>{flight.total_frames}</span>
          </div>
          <div style={styles.stat}>
            <span>Processed</span><span style={{ fontWeight:600 }}>{flight.processed_frames}</span>
          </div>
          <div style={styles.stat}>
            <span>Current Frame</span><span style={{ fontWeight:600 }}>#{frame.frame_index}</span>
          </div>
          <div style={styles.stat}>
            <span>Leaves in frame</span><span style={{ fontWeight:600 }}>{frame.leaf_count}</span>
          </div>
        </div>
      </div>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  empty: { textAlign:'center', padding:80, background:'#fff', borderRadius:16 },
  card: { background:'#fff', borderRadius:12, padding:20, boxShadow:'var(--shadow)' },
  toggleBtn: { padding:'6px 14px', fontSize:12, fontWeight:500, background:'var(--gray-100)',
    color:'var(--gray-500)', borderRadius:6 },
  toggleActive: { background:'var(--green-100)', color:'var(--green-700)' },
  navBtn: { padding:'8px 16px', fontSize:13, background:'var(--gray-100)', color:'var(--gray-600)', borderRadius:6, flex:1 },
  explainSection: { marginBottom:14 },
  explainTitle: { margin:'0 0 4px', fontSize:13, fontWeight:600, color:'var(--gray-700)' },
  explainText: { margin:0, fontSize:12, color:'var(--gray-500)', lineHeight:1.5 },
  stat: { display:'flex', justifyContent:'space-between', padding:'6px 0', fontSize:13,
    borderBottom:'1px solid var(--gray-100)' },
}
