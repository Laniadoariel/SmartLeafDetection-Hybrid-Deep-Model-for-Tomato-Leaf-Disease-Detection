import { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import api from '../api'

export default function Signup() {
  const [form, setForm] = useState({ full_name:'', username:'', password:'', confirm:'' })
  const [error, setError] = useState('')
  const nav = useNavigate()

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    if (form.password !== form.confirm) { setError('Passwords do not match'); return }
    try {
      await api.post('/auth/signup', {
        username: form.username, full_name: form.full_name, password: form.password,
      })
      nav('/login')
    } catch (err: any) {
      setError(err?.response?.data?.detail || err?.message || 'Signup failed')
    }
  }

  const set = (k: string) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setForm(f => ({ ...f, [k]: e.target.value }))

  return (
    <div style={styles.page}>
      <form onSubmit={submit} style={styles.card}>
        <div style={{ fontSize:48, textAlign:'center' }}>🌿</div>
        <h1 style={styles.title}>Create Account</h1>
        {error && <div style={styles.error}>{error}</div>}
        <input placeholder="Full Name" value={form.full_name} onChange={set('full_name')} required />
        <input placeholder="Username" value={form.username} onChange={set('username')} required style={{marginTop:10}} />
        <input placeholder="Password" type="password" value={form.password} onChange={set('password')} required style={{marginTop:10}} />
        <input placeholder="Confirm Password" type="password" value={form.confirm} onChange={set('confirm')} required style={{marginTop:10}} />
        <button type="submit" style={styles.btn}>Sign Up</button>
        <p style={styles.link}>Already have an account? <Link to="/login">Log in</Link></p>
      </form>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  page: { minHeight:'100vh', display:'flex', alignItems:'center', justifyContent:'center',
    background:'linear-gradient(135deg,#f0fdf4 0%,#dcfce7 50%,#bbf7d0 100%)' },
  card: { background:'#fff', borderRadius:16, padding:'40px 36px', width:380,
    boxShadow:'0 20px 60px rgba(0,0,0,.08)', display:'flex', flexDirection:'column', gap:4 },
  title: { fontSize:22, fontWeight:700, textAlign:'center', color:'#15803d', margin:'0 0 16px' },
  error: { background:'#fef2f2', color:'#dc2626', padding:'8px 12px', borderRadius:8, fontSize:13 },
  btn: { marginTop:16, padding:'12px', background:'#16a34a', color:'#fff', fontSize:15,
    borderRadius:8, fontWeight:600 },
  link: { fontSize:13, color:'#6b7280', textAlign:'center', marginTop:12 },
}
