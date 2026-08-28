import { useState, useRef, useLayoutEffect, useEffect } from 'react'
import { createPortal } from 'react-dom'
import './HelpButton.css'

interface HelpButtonProps {
  /** The help text shown on hover. Use \n for line breaks. */
  content: string
  /** Which side of the ? icon the tooltip appears on. Defaults to 'top'. */
  placement?: 'top' | 'bottom' | 'right' | 'left'
}

const GAP = 8 // px gap between trigger and tooltip edge
const MARGIN = 8 // px minimum gap kept between tooltip and the viewport edge

/**
 * A small circular ? icon that reveals a tooltip on hover or keyboard focus.
 * All tooltip content is sourced from src/helpContent.ts.
 *
 * The tooltip is rendered into document.body via a React portal so it
 * escapes every stacking context and always paints above all other elements.
 */
export default function HelpButton({ content, placement = 'top' }: HelpButtonProps) {
  const [visible, setVisible] = useState(false)
  const [rect, setRect] = useState<DOMRect | null>(null)
  // The tooltip's clamped top-left corner, in viewport space. Computed after
  // the bubble mounts so we know its measured width/height. Null until measured.
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null)
  const triggerRef = useRef<HTMLSpanElement>(null)
  const tooltipRef = useRef<HTMLSpanElement>(null)
  // Pending hide timer. A short delay lets the pointer travel from the trigger
  // across the gap onto the tooltip (to read/scroll it) without it closing.
  const hideTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  const cancelHide = () => {
    if (hideTimer.current !== null) {
      clearTimeout(hideTimer.current)
      hideTimer.current = null
    }
  }

  const show = () => {
    cancelHide()
    if (triggerRef.current) setRect(triggerRef.current.getBoundingClientRect())
    setPos(null) // hide until re-measured to avoid a flash at a stale position
    setVisible(true)
  }
  // Hide after a grace period so hovering onto the tooltip keeps it open.
  const scheduleHide = () => {
    cancelHide()
    hideTimer.current = setTimeout(() => setVisible(false), 120)
  }
  // Immediate hide, used for keyboard blur where there is no pointer to track.
  const hide = () => {
    cancelHide()
    setVisible(false)
  }

  // Clear any pending timer if the component unmounts mid-delay.
  useEffect(() => cancelHide, [])

  // Once the tooltip is in the DOM we know its real size, so we can place it
  // next to the trigger and then clamp it so it never spills past the viewport.
  useLayoutEffect(() => {
    if (!visible || !rect || !tooltipRef.current) return
    const tip = tooltipRef.current.getBoundingClientRect()
    const vw = window.innerWidth
    const vh = window.innerHeight
    const cx = rect.left + rect.width / 2
    const cy = rect.top + rect.height / 2

    // Preferred top-left corner based on the requested placement.
    let top: number
    let left: number
    switch (placement) {
      case 'top':
        top = rect.top - GAP - tip.height
        left = cx - tip.width / 2
        break
      case 'bottom':
        top = rect.bottom + GAP
        left = cx - tip.width / 2
        break
      case 'right':
        top = cy - tip.height / 2
        left = rect.right + GAP
        break
      case 'left':
        top = cy - tip.height / 2
        left = rect.left - GAP - tip.width
        break
    }

    // Clamp into the viewport, keeping at least MARGIN px of breathing room.
    // max() before min() so that if the tooltip is larger than the viewport we
    // still pin its top-left corner in view rather than pushing it off-screen.
    left = Math.min(Math.max(left, MARGIN), Math.max(MARGIN, vw - tip.width - MARGIN))
    top = Math.min(Math.max(top, MARGIN), Math.max(MARGIN, vh - tip.height - MARGIN))

    setPos({ top, left })
  }, [visible, rect, placement])

  return (
    <span
      className="help-btn"
      onMouseEnter={show}
      onMouseLeave={scheduleHide}
      onFocus={show}
      onBlur={hide}
    >
      {/*
        Use a <span> rather than a <button> so this can safely be placed
        inside other interactive elements (e.g. tab buttons) without
        producing invalid nested-button HTML.
      */}
      <span
        ref={triggerRef}
        className="help-btn__trigger"
        tabIndex={0}
        role="button"
        aria-label="Show help"
      >
        ?
      </span>

      {visible &&
        createPortal(
          <span
            ref={tooltipRef}
            role="tooltip"
            className={`help-btn__tooltip help-btn__tooltip--${placement}`}
            // Keep the tooltip open while the pointer is over it so it can be
            // read and scrolled; close once the pointer leaves it.
            onMouseEnter={cancelHide}
            onMouseLeave={scheduleHide}
            style={
              pos
                ? { top: pos.top, left: pos.left }
                : // First paint before measurement: render invisibly so it can be
                  // sized without flashing at an unclamped position.
                  { top: 0, left: 0, visibility: 'hidden' }
            }
          >
            {/* Inner element scrolls; the outer keeps the (unclipped) arrow. */}
            <span className="help-btn__tooltip-content">{content}</span>
          </span>,
          document.body,
        )}
    </span>
  )
}
