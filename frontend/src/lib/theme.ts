// The graph draws on a canvas, which cannot read CSS variables directly.  This
// hook resolves the colour tokens from index.css and refreshes them when the
// OS colour scheme changes.

import { useEffect, useState } from 'react'
import type { Support } from '../data/types'

export interface CanvasColours {
  /** True when the page background is dark, so palettes pick lighter colours. */
  dark: boolean
  surface: string
  ink: string
  muted: string
  node: string
  selection: string
  support: Record<Support, string>
}

function read(): CanvasColours {
  const style = getComputedStyle(document.documentElement)
  const v = (name: string) => style.getPropertyValue(name).trim()
  const surface = v('--surface-1')
  return {
    dark: luminance(surface) < 0.4,
    surface,
    ink: v('--text-primary'),
    muted: v('--text-muted'),
    node: v('--node'),
    selection: v('--text-primary'),
    support: {
      curated: v('--support-curated'),
      plausible: v('--support-plausible'),
      novel: v('--support-novel'),
    },
  }
}

function luminance(hex: string): number {
  const h = hex.replace('#', '')
  const n = parseInt(h.length === 3 ? h.replace(/./g, '$&$&') : h, 16)
  return (0.2126 * ((n >> 16) & 255) + 0.7152 * ((n >> 8) & 255) + 0.0722 * (n & 255)) / 255
}

export function hslToHex(h: number, s: number, l: number): string {
  const a = s * Math.min(l, 1 - l)
  const channel = (n: number) => {
    const k = (n + h / 30) % 12
    const v = l - a * Math.max(-1, Math.min(k - 3, Math.min(9 - k, 1)))
    return Math.round(255 * v).toString(16).padStart(2, '0')
  }
  return `#${channel(0)}${channel(8)}${channel(4)}`
}

/**
 * A categorical colour for item `index`.  Hues step by the golden angle so the
 * first (largest) items are well separated, and lightness/saturation cycle
 * through three bands to keep neighbours in the sequence distinguishable.
 */
export function categoricalColour(index: number, dark: boolean): string {
  const hue = (index * 137.508 + 18) % 360
  const band = index % 3
  const saturation = [0.68, 0.78, 0.56][band]
  const lightness = (dark ? [0.62, 0.54, 0.72] : [0.5, 0.4, 0.6])[band]
  return hslToHex(hue, saturation, lightness)
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

/** Linear blend of two `#rrggbb` colours; t = 0 gives `from`, 1 gives `to`. */
export function mix(from: string, to: string, t: number): string {
  const parse = (hex: string) => {
    const h = hex.replace('#', '')
    const n = parseInt(h.length === 3 ? h.replace(/./g, '$&$&') : h, 16)
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255]
  }
  const a = parse(from)
  const b = parse(to)
  const channel = (i: number) => Math.round(a[i] + (b[i] - a[i]) * t).toString(16).padStart(2, '0')
  return `#${channel(0)}${channel(1)}${channel(2)}`
}

/** Highlight colour for "this disease carries the chosen gene / symptom". */
export const accentColour = (dark: boolean) => (dark ? '#ffb347' : '#d9480f')

/** Colour of a claim verdict, for the graph and the panels. */
export function verdictColour(verdict: string, dark: boolean): string {
  const light: Record<string, string> = {
    confirmed: '#2b8a3e',
    partial: '#e08700',
    unsupported: '#d6336c',
    'edge-only': '#1c7ed6',
    missing: '#868e96',
  }
  const night: Record<string, string> = {
    confirmed: '#51cf66',
    partial: '#ffc53d',
    unsupported: '#ff6b9d',
    'edge-only': '#74a9ff',
    missing: '#adb5bd',
  }
  return (dark ? night : light)[verdict] ?? '#868e96'
}
