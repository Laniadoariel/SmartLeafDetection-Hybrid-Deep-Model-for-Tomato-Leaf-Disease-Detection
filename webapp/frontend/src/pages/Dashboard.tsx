import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import UploadTab from '../components/UploadTab'
import InvestigationTab from '../components/InvestigationTab'
import ResultsTab from '../components/ResultsTab'
import HistoryTab from '../components/HistoryTab'
import type { FlightDetail } from '../api'

const TABS = ['Upload', 'Investigation', 'Results', 'History'] as const

export default function Dashboard() {
  const [tab, setTab] = useState<number>(0)
  const [activeFlight, setActiveFlight] = useState<FlightDetail | null>(null)
  const nav = useNavigate()

  const logout = () => {
    localStorage.removeItem('token')
    localStorage.removeItem('username')
    nav('/login')
  }

  const onFlightReady = (flight: FlightDetail) => {
    setActiveFlight(flight)
    setTab(1) // jump to Investigation
  }

  const onSelectFlight = (flight: FlightDetail) => {
    setActiveFlight(flight)
    setTab(2) // jump to Results
  }

  return (
    <div style={{ minHeight: '100vh', background: 'var(--gray-50)' }}>
      {/* Header */}
      <header style={styles.header}>
        <div style={{ display:'flex', alignItems:'center', gap:10 }}>
          <span style={{ fontSize:28 }}>🌿</span>
          <span style={{ fontSize:18, fontWeight:700, color:'#15803d' }}>SmartLeafDetection</span>
        </div>
        <div style={{ display:'flex', alignItems:'center', gap:16 }}>
          <span style={{ fontSize:13, color:'var(--gray-500)' }}>
            {localStorage.getItem('username')}
          </span>
          <button onClick={logout} style={styles.logoutBtn}>Logout</button>
        </div>
      </header>

      {/* Tab bar */}
      <nav style={styles.tabBar}>
        {TABS.map((t, i) => (
          <button key={t} onClick={() => setTab(i)}
            style={{ ...styles.tabBtn, ...(tab === i ? styles.tabActive : {}) }}>
            {t}
          </button>
        ))}
      </nav>

      {/* Tab content */}
      <main style={{ maxWidth: 1200, margin: '0 auto', padding: '24px 20px' }}>
        {tab === 0 && <UploadTab onFlightReady={onFlightReady} />}
        {tab === 1 && <InvestigationTab flight={activeFlight} />}
        {tab === 2 && <ResultsTab flight={activeFlight} />}
        {tab === 3 && <HistoryTab onSelect={onSelectFlight} />}
      </main>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  header: {
    display:'flex', justifyContent:'space-between', alignItems:'center',
    padding:'12px 24px', background:'#fff', borderBottom:'1px solid var(--gray-200)',
    boxShadow:'0 1px 2px rgba(0,0,0,.04)',
  },
  logoutBtn: {
    padding:'6px 14px', background:'var(--gray-100)', color:'var(--gray-600)',
    fontSize:13, borderRadius:6,
  },
  tabBar: {
    display:'flex', gap:4, padding:'12px 24px', background:'#fff',
    borderBottom:'1px solid var(--gray-200)', maxWidth:1200, margin:'0 auto',
  },
  tabBtn: {
    padding:'8px 20px', background:'transparent', color:'var(--gray-500)',
    fontSize:14, fontWeight:500, borderRadius:8,
  },
  tabActive: {
    background:'var(--green-50)', color:'var(--green-700)', fontWeight:600,
  },
}
