import { useCallback, useEffect, useMemo, useState, type FormEvent } from 'react'
import './App.css'
import { getTitles, postChat, type TitleItem } from './api'

const SESSION_KEY = 'netflix-ai-session-id'

function App() {
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [chatInput, setChatInput] = useState('')
  const [messages, setMessages] = useState<{ role: 'user' | 'assistant'; text: string }[]>(
    [],
  )
  const [chatLoading, setChatLoading] = useState(false)

  const [filterQ, setFilterQ] = useState('')
  const [filterGenre, setFilterGenre] = useState('')
  const [filterType, setFilterType] = useState<'Movie' | 'TV Show' | ''>('')
  const [yearMin, setYearMin] = useState('')
  const [yearMax, setYearMax] = useState('')
  const [titles, setTitles] = useState<TitleItem[]>([])
  const [titlesTotal, setTitlesTotal] = useState(0)
  const [titlesLoading, setTitlesLoading] = useState(false)

  useEffect(() => {
    const existing = localStorage.getItem(SESSION_KEY)
    if (existing) setSessionId(existing)
  }, [])

  const persistSession = useCallback((sid: string) => {
    localStorage.setItem(SESSION_KEY, sid)
    setSessionId(sid)
  }, [])

  const onSendChat = async (e: FormEvent) => {
    e.preventDefault()
    const text = chatInput.trim()
    if (!text || chatLoading) return
    setChatInput('')
    setMessages((m) => [...m, { role: 'user', text }])
    setChatLoading(true)
    try {
      const res = await postChat(text, sessionId)
      persistSession(res.session_id)
      setMessages((m) => [...m, { role: 'assistant', text: res.reply }])
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Request failed'
      setMessages((m) => [...m, { role: 'assistant', text: `Error: ${msg}` }])
    } finally {
      setChatLoading(false)
    }
  }

  const onBrowse = async () => {
    setTitlesLoading(true)
    try {
      const res = await getTitles({
        q: filterQ || undefined,
        genre: filterGenre || undefined,
        type: filterType || undefined,
        year_min: yearMin ? parseInt(yearMin, 10) : undefined,
        year_max: yearMax ? parseInt(yearMax, 10) : undefined,
        limit: 30,
      })
      setTitles(res.items)
      setTitlesTotal(res.total)
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Request failed'
      setTitles([])
      setTitlesTotal(0)
      alert(msg)
    } finally {
      setTitlesLoading(false)
    }
  }

  const subtitle = useMemo(
    () =>
      sessionId
        ? `Session memory on · ${sessionId.slice(0, 8)}…`
        : 'New session · recommendations personalize as you chat',
    [sessionId],
  )

  return (
    <div className="app">
      <header className="top">
        <div>
          <h1>Netflix AI Agent</h1>
          <p className="subtitle">{subtitle}</p>
        </div>
      </header>

      <main className="grid">
        <section className="panel filters">
          <h2>Catalog filters</h2>
          <p className="hint">
            Structured query — calls <code>GET /api/titles</code> only (no LLM).
          </p>
          <label>
            Keywords
            <input
              value={filterQ}
              onChange={(e) => setFilterQ(e.target.value)}
              placeholder="Title or description contains…"
            />
          </label>
          <label>
            Genre / category substring
            <input
              value={filterGenre}
              onChange={(e) => setFilterGenre(e.target.value)}
              placeholder="e.g. Thriller, Anime"
            />
          </label>
          <label>
            Type
            <select
              value={filterType}
              onChange={(e) =>
                setFilterType(e.target.value as 'Movie' | 'TV Show' | '')
              }
            >
              <option value="">Any</option>
              <option value="Movie">Movie</option>
              <option value="TV Show">TV Show</option>
            </select>
          </label>
          <div className="row">
            <label>
              Year min
              <input
                value={yearMin}
                onChange={(e) => setYearMin(e.target.value)}
                placeholder="1990"
              />
            </label>
            <label>
              Year max
              <input
                value={yearMax}
                onChange={(e) => setYearMax(e.target.value)}
                placeholder="2024"
              />
            </label>
          </div>
          <button type="button" className="primary" onClick={onBrowse} disabled={titlesLoading}>
            {titlesLoading ? 'Loading…' : 'Browse catalog'}
          </button>
          <p className="meta">
            {titlesTotal > 0 ? `${titlesTotal} matches (showing ${titles.length})` : null}
          </p>
          <ul className="title-list">
            {titles.map((t) => (
              <li key={t.show_id}>
                <strong>{t.title}</strong>
                <span className="pill">{t.type}</span>
                {t.release_year ? (
                  <span className="muted">{t.release_year}</span>
                ) : null}
                {t.listed_in ? <p className="small">{t.listed_in}</p> : null}
              </li>
            ))}
          </ul>
        </section>

        <section className="panel chat">
          <h2>Assistant</h2>
          <p className="hint">
            RAG + LangGraph · <code>POST /api/chat</code> · uses{' '}
            <code>OPENAI_API_KEY</code>, <code>ANTHROPIC_API_KEY</code>, or{' '}
            <code>GROQ_API_KEY</code>
          </p>
          <div className="thread">
            {messages.length === 0 ? (
              <p className="muted empty">
                Ask for mood-based picks, “similar to…”, or refine with genres. Prior turns stay in
                context for this session.
              </p>
            ) : (
              messages.map((msg, i) => (
                <div key={i} className={`bubble ${msg.role}`}>
                  {msg.text}
                </div>
              ))
            )}
            {chatLoading ? <div className="bubble assistant loading">Thinking…</div> : null}
          </div>
          <form onSubmit={onSendChat}>
            <textarea
              rows={3}
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              placeholder="What should I watch tonight?"
            />
            <button type="submit" className="primary" disabled={chatLoading}>
              Send
            </button>
          </form>
        </section>
      </main>
    </div>
  )
}

export default App
