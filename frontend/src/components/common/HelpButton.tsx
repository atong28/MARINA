import './HelpButton.css'

interface HelpButtonProps {
  /** The help text shown on hover. Use \n for line breaks. */
  content: string
  /** Which side of the ? icon the tooltip appears on. Defaults to 'top'. */
  placement?: 'top' | 'bottom' | 'right' | 'left'
}

/**
 * A small circular ? icon that reveals a tooltip on hover or keyboard focus.
 * All tooltip content is sourced from src/helpContent.ts.
 */
export default function HelpButton({ content, placement = 'top' }: HelpButtonProps) {
  return (
    <span className={`help-btn help-btn--${placement}`}>
      {/*
        Use a <span> rather than a <button> so this can safely be placed
        inside other interactive elements (e.g. tab buttons) without
        producing invalid nested-button HTML.
      */}
      <span
        className="help-btn__trigger"
        tabIndex={0}
        role="button"
        aria-label="Show help"
      >
        ?
      </span>
      <span role="tooltip" className="help-btn__tooltip">
        {content}
      </span>
    </span>
  )
}
