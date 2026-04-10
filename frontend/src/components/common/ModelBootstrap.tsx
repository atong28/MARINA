import { useEffect } from 'react'
import { useModels } from '../../services/api'
import { useAppStore } from '../../store/store'

/**
 * Fetches available models once on mount and seeds the global model selection.
 * Renders nothing – place near the top of the component tree.
 */
function ModelBootstrap() {
  const { data, isError } = useModels()
  const initializeModelSelection = useAppStore((s) => s.initializeModelSelection)
  const setAvailableModels = useAppStore((s) => s.setAvailableModels)

  useEffect(() => {
    if (data) initializeModelSelection(data.models, data.default_model_id)
  }, [data, initializeModelSelection])

  useEffect(() => {
    if (isError) setAvailableModels([], '')
  }, [isError, setAvailableModels])

  return null
}

export default ModelBootstrap
