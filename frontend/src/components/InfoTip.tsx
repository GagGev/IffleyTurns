import { useCallback, useEffect, useId, useLayoutEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { HELP, type HelpTopic } from '../lib/help'

const WIDTH = 300
const GAP = 8

/**
 * A small "?" that explains part of the interface.  Opens on mouse hover and
 * keyboard focus, and toggles on tap/click for touch screens.  The popover is
 * rendered at the page level so the scrolling sidebar cannot clip it.
 */
export function InfoTip({ topic }: { topic: HelpTopic }) {
  const { title, body } = HELP[topic]
  const id = useId()
  const button = useRef<HTMLButtonElement>(null)
  const popover = useRef<HTMLDivElement>(null)
  const [open, setOpen] = useState(false)
  const [pinned, setPinned] = useState(false)
  const [position, setPosition] = useState({ top: -9999, left: -9999 })

  const close = useCallback(() => {
    setOpen(false)
    setPinned(false)
  }, [])

  // Place beside the button when there is room (the sidebar always has the
  // graph to its right), otherwise below or above it.  It must never cover
  // the button: hovering a popover on top of it would close and reopen it.
  useLayoutEffect(() => {
    if (!open || !button.current || !popover.current) return
    const r = button.current.getBoundingClientRect()
    const height = popover.current.offsetHeight
    const clampTop = (top: number) => Math.min(Math.max(GAP, top), window.innerHeight - height - GAP)
    if (window.innerWidth - r.right - GAP >= WIDTH + GAP) {
      setPosition({ top: clampTop(r.top - 12), left: r.right + GAP })
      return
    }
    const left = Math.min(Math.max(GAP, r.left - 12), window.innerWidth - WIDTH - GAP)
    const below = r.bottom + GAP
    const top = below + height > window.innerHeight - GAP ? Math.max(GAP, r.top - GAP - height) : below
    setPosition({ top, left })
  }, [open])

  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && close()
    const onPointer = (e: PointerEvent) => {
      if (!button.current?.contains(e.target as Node) && !popover.current?.contains(e.target as Node)) close()
    }
    window.addEventListener('keydown', onKey)
    window.addEventListener('pointerdown', onPointer)
    // A fixed popover would drift from its button when the sidebar scrolls.
    window.addEventListener('scroll', close, true)
    window.addEventListener('resize', close)
    return () => {
      window.removeEventListener('keydown', onKey)
      window.removeEventListener('pointerdown', onPointer)
      window.removeEventListener('scroll', close, true)
      window.removeEventListener('resize', close)
    }
  }, [open, close])

  return (
    <>
      <button
        ref={button}
        type="button"
        className="info-tip"
        aria-label={`About ${title}`}
        aria-expanded={open}
        aria-describedby={open ? id : undefined}
        onPointerEnter={(e) => e.pointerType === 'mouse' && setOpen(true)}
        onPointerLeave={(e) => e.pointerType === 'mouse' && !pinned && setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => !pinned && setOpen(false)}
        onClick={(e) => {
          // Inside a <label>, a click must not also toggle the label's control.
          e.preventDefault()
          e.stopPropagation()
          if (pinned) close()
          else {
            setPinned(true)
            setOpen(true)
          }
        }}
      >
        ?
      </button>
      {open &&
        createPortal(
          <div ref={popover} id={id} role="tooltip" className="info-popover" style={{ ...position, width: WIDTH }}>
            <strong className="info-title">{title}</strong>
            {body}
          </div>,
          document.body,
        )}
    </>
  )
}
