import { useHealth } from '../../services/api'
import { useAppStore } from '../../store/store'
import { HELP } from '../../helpContent'
import HelpButton from './HelpButton'
import './HighlightToggle.css'

/**
 * Switches result cards between the similarity-map depiction and the plain one.
 *
 * Every card carries both images, so this is a pure display choice — flipping it
 * never re-runs a prediction. The preference persists across sessions.
 */
function HighlightToggle() {
  const { data: health } = useHealth()
  const highlightEnabled = useAppStore((s) => s.highlightEnabled)
  const setHighlightEnabled = useAppStore((s) => s.setHighlightEnabled)

  // Hidden when the server renders plain depictions only: there would be nothing
  // to switch to. An unfetched health response leaves it visible.
  if (health?.highlight_available === false) return null

  // The help button sits outside the <label> so clicking it does not flip the box.
  return (
    <div className="highlight-toggle-group">
      <label className="highlight-toggle">
        <input
          type="checkbox"
          className="highlight-toggle__input"
          checked={highlightEnabled}
          onChange={(e) => setHighlightEnabled(e.target.checked)}
        />
        <span className="highlight-toggle__text">Similarity map</span>
      </label>
      <HelpButton content={HELP.controls.highlight} placement="bottom" />
    </div>
  )
}

export default HighlightToggle
