import { useState, useRef } from 'react'
import { createPortal } from 'react-dom'
import './HelpButton.css'

interface HelpButtonProps {
  /** The help text shown on hover. Use \n for line breaks. */
  content: string
  /** Which side of the ? icon the tooltip appears on. Defaults to 'top'. */
  placement?: 'top' | 'bottom' | 'right' | 'left'
}

const GAP = 8 // px gap between trigger and tooltip edge

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
  const triggerRef = useRef<HTMLSpanElement>(null)

  const show = () => {
    if (triggerRef.current) setRect(triggerRef.current.getBoundingClientRect())
    setVisible(true)
  }
  const hide = () => setVisible(false)

  // Compute fixed-position style so the tooltip sits next to the trigger.
  // All coordinates are in viewport space (position: fixed).
  const tooltipStyle = (): React.CSSProperties => {
    if (!rect) return {}
    const cx = rect.left + rect.width / 2
    const cy = rect.top + rect.height / 2
    switch (placement) {
      case 'top':
        // anchor point: center-top of trigger; tooltip bottom aligns there
        return { top: rect.top, left: cx, transform: `translate(-50%, calc(-100% - ${GAP}px))` }
      case 'bottom':
        // anchor point: center-bottom of trigger; tooltip top aligns there
        return { top: rect.bottom + GAP, left: cx, transform: 'translateX(-50%)' }
      case 'right':
        // anchor point: right-center of trigger; tooltip left aligns there
        return { top: cy, left: rect.right + GAP, transform: 'translateY(-50%)' }
      case 'left':
        // anchor point: left-center of trigger; tooltip right aligns there
        return { top: cy, left: rect.left - GAP, transform: 'translate(calc(-100%), -50%)' }
    }
  }

  return (
    <span
      className="help-btn"
      onMouseEnter={show}
      onMouseLeave={hide}
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
            role="tooltip"
            className={`help-btn__tooltip help-btn__tooltip--${placement}`}
            style={tooltipStyle()}
          >
            {content}
          </span>,
          document.body,
        )}
    </span>
  )
}
