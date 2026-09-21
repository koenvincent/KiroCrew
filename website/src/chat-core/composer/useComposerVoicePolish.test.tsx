import { describe, it, expect, vi, beforeEach } from 'vitest'
import { act, renderHook, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

/* The AI transcript polish, from the composer's side.
 *
 * What is worth pinning here is not that a tidier string can replace a rougher
 * one -- it is the three refusals, because each one is a way this feature could
 * cost a user their words:
 *
 *   - the switch is the consent, so with `stt.polish` off nothing is sent at all;
 *   - the replacement only ever rewrites the span it wrote itself, and only while
 *     that span is untouched, so text typed after the transcript landed survives;
 *   - a decline, a failure or an unchanged reply leaves the composer alone.
 *
 * The engine (`useVoiceInput`) is faked so the test can invoke the delivery
 * callback directly: that callback IS the seam the real streaming final and the
 * real batch result both arrive through, so driving it exercises both origins. */

type Delivery = (text: string, sessionId: string | null, origin: 'stream' | 'batch') => void

const fx = vi.hoisted(() => {
  const captured: { onText: ((...a: never[]) => void) | null } = { onText: null }
  const polish = { fn: null as unknown }
  return { captured, polish }
})

vi.mock('../../hooks/useVoiceInput', () => ({
  // The hook hands its delivery callback in as the first argument; keep the
  // latest one so the test can play the part of the recogniser.
  useVoiceInput: (onText: (...a: never[]) => void) => {
    fx.captured.onText = onText
    return {
      recording: false, transcribing: false, sessionOwner: null, streamEnabled: false,
      toggle: vi.fn(), start: vi.fn(async () => {}), stop: vi.fn(), cancel: vi.fn(), prewarm: vi.fn(),
      error: null, level: 0, deviceLabel: '', deviceId: '', clearError: vi.fn(), partial: '',
      download: null, sampleRef: { current: {} }, switchDevice: vi.fn(), deviceSwitchIsLive: false,
    }
  },
  voiceInputSupported: true,
}))
vi.mock('../../hooks/usePushToTalk', () => ({ usePushToTalk: () => undefined }))
vi.mock('../../api/client', () => ({
  api: {
    sttConfig: vi.fn().mockResolvedValue({ enabled: true, available: true, provider: 'local' }),
    sttPolish: (...args: unknown[]) => (fx.polish.fn as (...a: unknown[]) => unknown)(...args),
  },
}))

import { useComposerVoice, _resetMicOwner } from './useComposerVoice'

const BASE = { enabled: true, available: true, streaming: false, dictation_panel: true, provider: 'local' }

function mount(polish: boolean) {
  const inputRef = { current: '' }
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  qc.setQueryData(['sttConfig'], { ...BASE, polish })
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={qc}>{children}</QueryClientProvider>
  )
  const view = renderHook(
    () => useComposerVoice({
      sessionId: 'slot-a',
      inputRef,
      setInput: (v: string) => { inputRef.current = v },
    }),
    { wrapper },
  )
  return { view, inputRef, deliver: fx.captured.onText as unknown as Delivery }
}

let sttPolish: ReturnType<typeof vi.fn>

beforeEach(() => {
  _resetMicOwner()
  fx.captured.onText = null
  sttPolish = vi.fn()
  fx.polish.fn = sttPolish
})

describe('transcript polish — the switch is the consent', () => {
  it('sends nothing when stt.polish is off', async () => {
    const { inputRef, deliver } = mount(false)

    await act(async () => { deliver('hello there', 'slot-a', 'batch') })

    // The transcript still lands: the recogniser's own text is never withheld.
    expect(inputRef.current).toBe('hello there')
    // And no request was made at all, so an install with the switch off sends
    // nothing anywhere even if the server would have refused it.
    expect(sttPolish).not.toHaveBeenCalled()
  })
})

describe('transcript polish — replacement', () => {
  it('replaces the transcript it wrote, after the transcript is already usable', async () => {
    sttPolish.mockResolvedValue({ ok: true, changed: true, text: 'Hello there.', original: 'hello there' })
    const { inputRef, deliver } = mount(true)

    await act(async () => { deliver('hello there', 'slot-a', 'batch') })

    // Delivered first, corrected second: the value is the raw transcript at the
    // moment the request goes out.
    expect(sttPolish).toHaveBeenCalledWith('hello there')
    await waitFor(() => expect(inputRef.current).toBe('Hello there.'))
  })

  it('rewrites only its own span, leaving text that was already in the composer', async () => {
    sttPolish.mockResolvedValue({ ok: true, changed: true, text: 'Hello there.', original: 'hello there' })
    const { inputRef, deliver } = mount(true)
    inputRef.current = 'draft:'

    await act(async () => { deliver('hello there', 'slot-a', 'batch') })

    await waitFor(() => expect(inputRef.current).toBe('draft: Hello there.'))
  })

  it('keeps the transcript when the model declines', async () => {
    sttPolish.mockResolvedValue({ ok: true, changed: false, text: 'hello there', original: 'hello there' })
    const { inputRef, deliver } = mount(true)

    await act(async () => { deliver('hello there', 'slot-a', 'batch') })
    await act(async () => { await Promise.resolve() })

    expect(inputRef.current).toBe('hello there')
  })

  it('keeps the transcript when the request fails, and surfaces no error', async () => {
    sttPolish.mockRejectedValue(new Error('boom'))
    const { view, inputRef, deliver } = mount(true)

    await act(async () => { deliver('hello there', 'slot-a', 'batch') })
    await act(async () => { await Promise.resolve() })

    expect(inputRef.current).toBe('hello there')
    // A failed polish costs the user nothing, so it is not their problem to read.
    expect(view.result.current.voice.error).toBeFalsy()
  })
})

describe('transcript polish — never overwrites the user', () => {
  it('drops the replacement when the user typed after the transcript landed', async () => {
    let release: (v: unknown) => void = () => {}
    sttPolish.mockReturnValue(new Promise(res => { release = res }))
    const { inputRef, deliver } = mount(true)

    await act(async () => { deliver('hello there', 'slot-a', 'batch') })
    expect(inputRef.current).toBe('hello there')

    // The user keeps typing while the correction is in flight.
    inputRef.current = 'hello there, and one more thing'

    await act(async () => {
      release({ ok: true, changed: true, text: 'Hello there.', original: 'hello there' })
      await Promise.resolve()
    })

    // Untouched. A merge here would have deleted what they wrote.
    expect(inputRef.current).toBe('hello there, and one more thing')
  })

  it('drops the replacement when the composer was cleared by a send', async () => {
    let release: (v: unknown) => void = () => {}
    sttPolish.mockReturnValue(new Promise(res => { release = res }))
    const { inputRef, deliver } = mount(true)

    await act(async () => { deliver('hello there', 'slot-a', 'batch') })
    inputRef.current = ''

    await act(async () => {
      release({ ok: true, changed: true, text: 'Hello there.', original: 'hello there' })
      await Promise.resolve()
    })

    // The message is already gone; putting a corrected copy of it back into an
    // empty composer would be a phantom draft the user never asked for.
    expect(inputRef.current).toBe('')
  })
})
