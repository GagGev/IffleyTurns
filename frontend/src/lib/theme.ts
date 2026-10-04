// The graph draws on a canvas, which cannot read CSS variables directly.  This
// hook resolves the colour tokens from index.css and refreshes them when the
// OS colour scheme changes.

import { useEffect, useState } from 'react'
import type { Relationship } from '../data/types'

export interface CanvasColours {
  surface: string
  ink: string
  inkSecondary: string
  muted: string
  node: string
  selection: string
  computed: string
  relationship: Record<Relationship, string>
}

function read(): CanvasColours {
  const style = getComputedStyle(document.documentElement)
  const v = (name: string) => style.getPropertyValue(name).trim()
  return {
    surface: v('--surface-1'),
    ink: v('--text-primary'),
    inkSecondary: v('--text-secondary'),
    muted: v('--text-muted'),
    node: v('--node'),
    selection: v('--text-primary'),
    computed: v('--edge-computed'),
    relationship: {
      similar: v('--edge-similar'),
      'related but distinct': v('--edge-related'),
      unrelated: v('--edge-unrelated'),
    },
  }
}

export function useCanvasColours(): CanvasColours {
  const [colours, setColours] = useState(read)
  useEffect(() => {
    const media = window.matchMedia('(prefers-color-scheme: dark)')
    const update = () => setColours(read())
    media.addEventListener('change', update)
    return () => media.removeEventListener('change', update)
  }, [])
  return colours
}

/** `#rrggbb` plus alpha as an rgba() string. */
export function withAlpha(hex: string, alpha: number): string {
  const h = hex.replace('#', '')
  const n = parseInt(h.length === 3 ? h.replace(/./g, '$&$&') : h, 16)
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`
}
