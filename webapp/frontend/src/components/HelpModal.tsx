import { useEffect } from 'react'
import { X } from 'lucide-react'

interface Props { open: boolean; onClose: () => void }

/**
 * Help / User Guide modal. Self-contained overlay — does not touch any other
 * application state or workflow. Scrollable body, closes on backdrop click,
 * the X button, or the Escape key.
 */
export default function HelpModal({ open, onClose }: Props) {
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])

  if (!open) return null

  return (
    <div style={styles.backdrop} onClick={onClose}>
      <div style={styles.modal} onClick={e => e.stopPropagation()} role="dialog" aria-modal="true" aria-label="Help and User Guide">
        {/* Header (fixed) */}
        <div style={styles.header}>
          <div style={{ display:'flex', alignItems:'center', gap:10 }}>
            <span style={{ fontSize:22 }}>🌿</span>
            <h2 style={{ margin:0, fontSize:18, color:'var(--green-700)' }}>Help &amp; User Guide</h2>
          </div>
          <button onClick={onClose} style={styles.closeBtn} aria-label="Close help">
            <X size={18} />
          </button>
        </div>

        {/* Scrollable body */}
        <div style={styles.body}>
          <p style={styles.intro}>
            SmartLeafDetection analyzes a drone video of a tomato field and reports a
            disease diagnosis for each individual leaf it detects and tracks. This guide
            walks through how to use it and how to read the results.
          </p>

          <Section emoji="🚀" title="1. Getting Started">
            <Step n={1}>Open the <b>Upload</b> tab and drop a drone/UAV video onto the box, or click to browse.</Step>
            <Step n={2}>Supported formats: <b>MP4, MOV, AVI, MKV</b>.</Step>
            <Step n={3}>Click <b>🚀 Start Analysis</b>. Upload and processing begin automatically.</Step>
            <Step n={4}>Watch the <b>processing timeline</b> — frame extraction → leaf detection → tracking → classification → aggregation.</Step>
            <Step n={5}>When it finishes, the app opens the <b>Investigation</b> tab so you can browse the analyzed frames.</Step>
            <Step n={6}>Open the <b>Results</b> tab to see the final per-leaf diagnosis, and revisit any run from the <b>History</b> tab.</Step>
          </Section>

          <Section emoji="🎥" title="2. Recommended Video Recording">
            <Bullet>Use <b>high-resolution</b> video — small, distant leaves are hard to detect at low resolution.</Bullet>
            <Bullet>Record in <b>daylight with good, even lighting</b>; avoid deep shadows and glare.</Bullet>
            <Bullet>Keep the drone/camera <b>stable and steady</b> — smooth motion helps the tracker follow each leaf.</Bullet>
            <Bullet>Fly at a <b>suitable distance</b> so leaves are clearly visible and fill a good part of the frame.</Bullet>
            <Bullet><b>Avoid blurry or out-of-focus</b> footage; fly slowly enough to keep frames sharp.</Bullet>
          </Section>

          <Section emoji="📊" title="3. Understanding the Results">
            <p style={styles.para}>
              A single frame can show <b>many leaf detections</b>, but the same physical leaf
              appears in many frames. The system tracks each leaf across frames (BoT-SORT) and
              combines all its observations into <b>one final result</b>. So one result card
              = one unique leaf, not one detection.
            </p>
            <Bullet><b>Leaf ID</b> — a stable identifier for each tracked leaf (shown as <i>Leaf #N</i>).</Bullet>
            <Bullet><b>Disease Label</b> — the predicted disease, or <i>Healthy</i>.</Bullet>
            <Bullet><b>Confidence Score</b> — how sure the classifier is about the winning class (shown as a %).</Bullet>
            <Bullet><b>Cropped Leaf Images</b> — thumbnails of the actual leaf crops used for classification.</Bullet>
            <Bullet><b>Seen in N frames</b> — how many frames (views) the leaf was observed and aggregated over.</Bullet>
            <Bullet><b>Summary cards</b> — total <i>Leaves Inspected</i>, <i>Healthy</i>, <i>Diseased</i>, and <i>Frames Analyzed</i>.</Bullet>
            <Bullet><b>Disease Distribution</b> — a breakdown of how many leaves fall under each disease.</Bullet>
            <p style={styles.para}>
              The <b>Investigation</b> tab shows per-frame detections (annotated frames with
              leaf boxes), while the <b>Results</b> tab shows the aggregated per-leaf diagnosis.
            </p>
          </Section>

          <Section emoji="💡" title="4. Tips">
            <Bullet>Fly a little <b>slower</b> — denser, sharper frames let the tracker link the same leaf across more views.</Bullet>
            <Bullet>Keep leaves in focus and well-lit for the most reliable disease predictions.</Bullet>
            <Bullet>Capture each area from <b>multiple angles</b>; more views per leaf means a more confident aggregated result.</Bullet>
            <Bullet>Treat a <b>low confidence</b> or a leaf seen in only a few frames as “needs a human look”.</Bullet>
          </Section>

          <Section emoji="🛠️" title="5. Common Issues">
            <Issue problem="Few or no leaves detected">
              The footage is likely too far, too dark, or blurry. Re-record closer, steadier, and in better light.
            </Issue>
            <Issue problem="Low confidence score">
              The leaf’s symptoms may be ambiguous or the image unclear. Treat it as uncertain and verify manually; sharper, closer footage helps.
            </Issue>
            <Issue problem="Unsupported video format">
              Only MP4, MOV, AVI, and MKV are accepted. Convert the file to one of these and upload again.
            </Issue>
            <Issue problem="Empty results">
              No trackable leaves were found in the analyzed segment. Try a clearer clip where leaves are closer and in focus.
            </Issue>
            <Issue problem="Long analysis time">
              Large or high-resolution videos take longer — the system processes frames one by one. Let the timeline finish; progress is shown live.
            </Issue>
            <Issue problem="Blurry or low-quality video">
              Blur reduces both detection and tracking quality. Re-record with a steady camera and sharper focus.
            </Issue>
          </Section>

          <p style={styles.footer}>
            Diagnoses are decision-support only and should be confirmed by a human before acting.
          </p>
        </div>
      </div>
    </div>
  )
}

function Section({ emoji, title, children }: { emoji: string; title: string; children: React.ReactNode }) {
  return (
    <section style={styles.section}>
      <h3 style={styles.sectionTitle}><span style={{ marginRight:8 }}>{emoji}</span>{title}</h3>
      <div>{children}</div>
    </section>
  )
}

function Step({ n, children }: { n: number; children: React.ReactNode }) {
  return (
    <div style={styles.step}>
      <span style={styles.stepNum}>{n}</span>
      <span style={styles.stepText}>{children}</span>
    </div>
  )
}

function Bullet({ children }: { children: React.ReactNode }) {
  return (
    <div style={styles.bullet}>
      <span style={{ color:'var(--green-600)', marginTop:1 }}>•</span>
      <span style={styles.stepText}>{children}</span>
    </div>
  )
}

function Issue({ problem, children }: { problem: string; children: React.ReactNode }) {
  return (
    <div style={styles.issue}>
      <div style={styles.issueTitle}>⚠️ {problem}</div>
      <div style={styles.issueText}>{children}</div>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  backdrop: {
    position:'fixed', inset:0, background:'rgba(15,23,42,.45)',
    display:'flex', alignItems:'center', justifyContent:'center',
    padding:20, zIndex:1000,
  },
  modal: {
    background:'#fff', borderRadius:16, width:'100%', maxWidth:720,
    maxHeight:'85vh', display:'flex', flexDirection:'column',
    boxShadow:'0 20px 60px rgba(0,0,0,.25)', overflow:'hidden',
  },
  header: {
    display:'flex', justifyContent:'space-between', alignItems:'center',
    padding:'16px 20px', borderBottom:'1px solid var(--gray-200)', background:'var(--green-50)',
  },
  closeBtn: {
    display:'flex', alignItems:'center', justifyContent:'center',
    width:32, height:32, borderRadius:8, background:'var(--gray-100)', color:'var(--gray-600)',
  },
  body: { padding:'20px 24px', overflowY:'auto' },
  intro: { margin:'0 0 18px', fontSize:14, color:'var(--gray-600)', lineHeight:1.6 },
  section: { marginBottom:22 },
  sectionTitle: { margin:'0 0 10px', fontSize:15, fontWeight:700, color:'var(--green-700)' },
  para: { margin:'8px 0', fontSize:13, color:'var(--gray-600)', lineHeight:1.6 },
  step: { display:'flex', gap:10, alignItems:'flex-start', marginBottom:8 },
  stepNum: {
    flexShrink:0, width:22, height:22, borderRadius:'50%', background:'var(--green-500)',
    color:'#fff', fontSize:12, fontWeight:700, display:'flex', alignItems:'center', justifyContent:'center',
  },
  stepText: { fontSize:13, color:'var(--gray-700)', lineHeight:1.55 },
  bullet: { display:'flex', gap:8, alignItems:'flex-start', marginBottom:7 },
  issue: { padding:'10px 12px', background:'var(--gray-50)', border:'1px solid var(--gray-200)', borderRadius:8, marginBottom:8 },
  issueTitle: { fontSize:13, fontWeight:600, color:'var(--gray-700)', marginBottom:2 },
  issueText: { fontSize:12.5, color:'var(--gray-500)', lineHeight:1.55 },
  footer: { margin:'6px 0 0', fontSize:12, color:'var(--gray-400)', fontStyle:'italic', textAlign:'center' },
}
