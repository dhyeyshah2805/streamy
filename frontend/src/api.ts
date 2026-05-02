export type ChatResponse = {
  session_id: string
  reply: string
}

export type TitleItem = {
  show_id: string
  type: string
  title: string
  release_year: number | null
  rating: string | null
  listed_in: string | null
  description: string | null
}

export type TitlesResponse = {
  total: number
  items: TitleItem[]
}

export async function postChat(
  message: string,
  sessionId: string | null,
): Promise<ChatResponse> {
  const r = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message,
      session_id: sessionId || undefined,
    }),
  })
  if (!r.ok) {
    throw new Error((await r.text()) || r.statusText)
  }
  return r.json()
}

export async function getTitles(
  params: Record<string, string | number | undefined>,
): Promise<TitlesResponse> {
  const sp = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== '') sp.set(k, String(v))
  }
  const q = sp.toString()
  const r = await fetch(`/api/titles${q ? `?${q}` : ''}`)
  if (!r.ok) throw new Error((await r.text()) || r.statusText)
  return r.json()
}
