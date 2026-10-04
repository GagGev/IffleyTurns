// Which page is showing: the home page, the patient view or the researcher
// view.  Kept in the ?view= query parameter so pages can be bookmarked, while
// the researcher view keeps using the #hash for its selection.

import { useCallback, useEffect, useState } from 'react'

export type AppView = 'home' | 'patient' | 'research'

function current(): AppView {
  const view = new URLSearchParams(window.location.search).get('view')
  if (view === 'patient' || view === 'research') return view
  // Links shared before the home page existed point straight at a selection.
  return /^#(disease|pair)=/.test(window.location.hash) ? 'research' : 'home'
}

export function useAppView(): [AppView, (next: AppView) => void] {
  const [view, setView] = useState<AppView>(current)

  useEffect(() => {
    const onPop = () => setView(current())
    window.addEventListener('popstate', onPop)
    return () => window.removeEventListener('popstate', onPop)
  }, [])

  const navigate = useCallback((next: AppView) => {
    const url = next === 'home' ? window.location.pathname : `${window.location.pathname}?view=${next}`
    history.pushState(null, '', url)
    window.scrollTo(0, 0)
    setView(next)
  }, [])

  return [view, navigate]
}
