import { useAppStore } from '../../store/store'
import './ModelSelector.css'

function ModelSelector() {
  const { availableModels, selectedModelId, defaultModelId, setSelectedModelId } = useAppStore()
  const selected = selectedModelId ?? defaultModelId ?? availableModels?.[0]?.id ?? ''

  if (!availableModels || availableModels.length === 0) {
    return (
      <div className="model-selector">
        <span className="model-selector__label">Model</span>
        <span className="model-selector__loading">Loading…</span>
      </div>
    )
  }

  return (
    <div className="model-selector">
      <label className="model-selector__label" htmlFor="model-select">Model</label>
      <select
        id="model-select"
        className="model-selector__select"
        value={selected}
        onChange={(e) => setSelectedModelId(e.target.value)}
      >
        {availableModels.map((m) => {
          const tags: string[] = []
          if (m.default) tags.push('default')
          if (m.type) tags.push(m.type)
          const suffix = tags.length ? ` (${tags.join(', ')})` : ''
          return (
            <option key={m.id} value={m.id}>
              {m.display_name ?? m.id}{suffix}
            </option>
          )
        })}
      </select>
    </div>
  )
}

export default ModelSelector
