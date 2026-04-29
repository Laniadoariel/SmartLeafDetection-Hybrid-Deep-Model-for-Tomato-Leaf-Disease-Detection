import { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import api from '../api'

export default function Login() {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const nav = useNavigate()

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    try {
      const { data } = await api.post('/auth/login', { username, password })
      localStorage.setItem('token', data.access_token)
      localStorage.setItem('username', data.username)
      nav('/dashboard')
    } catch (err: any) {
      setError(err?.response?.data?.detail || err?.message || 'Login failed')
    }
  }

  return (
    <div style={styles.page}>
      <form onSubmit={submit} style={styles.card}>
        <div style={styles.logo}>🌿</div>
        <h1 style={styles.title}>SmartLeafDetection</h1>
        <p style={styles.subtitle}>Drone-based tomato disease analysis</p>
        {error && <div style={styles.error}>{error}</div>}
        <input placeholder="Username" value={username}
          onChange={e => setUsername(e.target.value)} required />
        <input placeholder="Password" type="password" value={password}
          onChange={e => setPassword(e.target.value)} required style={{marginTop:10}} />
        <button type="submit" style={styles.btn}>Log In</button>
        <p style={styles.link}>Don't have an account? <Link to="/signup">Sign up</Link></p>
      </form>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  page: { minHeight:'100vh', display:'flex', alignItems:'center', justifyContent:'center',
    background:'linear-gradient(135deg,#f0fdf4 0%,#dcfce7 50%,#bbf7d0 100%)' },
  card: { background:'#fff', borderRadius:16, padding:'40px 36px', width:380,
    boxShadow:'0 20px 60px rgba(0,0,0,.08)', display:'flex', flexDirection:'column', gap:4 },
  logo: { fontSize:48, textAlign:'center' },
  title: { fontSize:22, fontWeight:700, textAlign:'center', color:'#15803d', margin:0 },
  subtitle: { fontSize:13, color:'#6b7280', textAlign:'center', marginBottom:16 },
  error: { background:'#fef2f2', color:'#dc2626', padding:'8px 12px', borderRadius:8, fontSize:13 },
  btn: { marginTop:16, padding:'12px', background:'#16a34a', color:'#fff', fontSize:15,
    borderRadius:8, fontWeight:600 },
  link: { fontSize:13, color:'#6b7280', textAlign:'center', marginTop:12 },
}
