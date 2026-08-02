/**
 * Ephemeral notice pinned to the top-left of the viewport.
 *
 * Dismisses itself; a new `id` on the same text restarts the timer, so
 * pasting the same table twice still reads as two separate confirmations.
 */
import { useEffect } from 'react'
import './Toast.css'

export interface ToastMessage {
  id: number
  text: string
  tone: 'info' | 'error'
}

const DISMISS_MS = { info: 3200, error: 7000 }

interface ToastProps {
  message: ToastMessage | null
  onDismiss: () => void
}

function Toast({ message, onDismiss }: ToastProps) {
  const id = message?.id
  const tone = message?.tone

  useEffect(() => {
    if (id === undefined || !tone) return
    const timer = setTimeout(onDismiss, DISMISS_MS[tone])
    return () => clearTimeout(timer)
  }, [id, tone, onDismiss])

  if (!message) return null

  return (
    <div className={`toast toast--${message.tone}`} role="status" aria-live="polite">
      {message.text}
    </div>
  )
}

export default Toast
