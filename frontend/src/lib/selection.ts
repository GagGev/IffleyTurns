// Selection lives in the URL hash so a researcher can bookmark or share the
// exact disease or pair they are looking at.

import { useCallback, useEffect, useState } from 'react'

export type Selection =
  | { kind: 'disease'; id: string }
  | { kind: 'pair'; a: string; b: string }
  | null

function parse(hash: string): Selection {
  const params = new URLSearchParams(hash.replace(/^#/, ''))
  const disease = params.get('disease')
  if (disease) return { kind: 'disease', id: disease }
  const pair = params.get('pair')
  if (pair) {
    const [a, b] = pair.split('~')
    if (a && b) return { kind: 'pair', a, b }
  }
  return null
}

function serialise(selection: Selection): string {
  if (!selection) return ''
  const params = new URLSearchParams()
  if (selection.kind === 'disease') params.set('disease', selection.id)
  else params.set('pair', `${selection.a}~${selection.b}`)
  return `#${params.toString()}`
}

export function useHashSelection(): [Selection, (next: Selection) => void] {
  const [selection, setSelection] = useState<Selection>(() => parse(window.location.hash))

  useEffect(() => {
    const onHashChange = () => setSelection(parse(window.location.hash))
    window.addEventListener('hashchange', onHashChange)
    return () => window.removeEventListener('hashchange', onHashChange)
  }, [])

  const select = useCallback((next: Selection) => {
    const hash = serialise(next)
    if (hash !== window.location.hash) {
      // pushState keeps browser Back working without triggering hashchange.
      history.pushState(null, '', hash || window.location.pathname + window.location.search)
    }
    setSelection(next)
  }, [])

  useEffect(() => {
    const onPopState = () => setSelection(parse(window.location.hash))
    window.addEventListener('popstate', onPopState)
    return () => window.removeEventListener('popstate', onPopState)
  }, [])

  return [selection, select]
}
