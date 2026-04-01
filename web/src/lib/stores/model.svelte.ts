/**
 * Model loading state and text encoding store.
 *
 * Uses a Web Worker for Transformers.js inference so the main thread
 * stays responsive during model download (~23MB quantized).
 *
 * Usage:
 *   import { modelStatus, initModel, encodeText } from '$lib/stores/model';
 *   // In component onMount: initModel()
 *   // $modelStatus.status is 'idle' | 'loading' | 'ready' | 'error'
 */

import type { ProjectionWeights } from '$lib/inference/projection';
import { projectText } from '$lib/inference/projection';

export type ModelStatus = 'idle' | 'loading' | 'ready' | 'error';

export interface ModelState {
  status: ModelStatus;
  /** 0-1 download progress for the Transformers.js model */
  progress: number;
  progressFile: string;
  error: string | null;
}

// Svelte 5 reactive state: exported as a plain object updated in-place so
// component-level $derived / $state can observe it via the returned ref.
// We export a getter so callers always read the live value.
let _state: ModelState = $state({
  status: 'idle',
  progress: 0,
  progressFile: '',
  error: null,
});

export function modelStatus(): ModelState {
  return _state;
}

let worker: Worker | null = null;
let weights: ProjectionWeights | null = null;

// Promise-based request queue: id -> { resolve, reject }
const pending = new Map<
  number,
  {
    resolve: (emb: Float32Array) => void;
    reject: (err: Error) => void;
  }
>();
let nextId = 0;

type WorkerOut =
  | { type: 'ready' }
  | { type: 'progress'; loaded: number; total: number; file: string }
  | { type: 'result'; id: number; embedding: number[] }
  | { type: 'error'; message: string };

function handleWorkerMessage(event: MessageEvent<WorkerOut>): void {
  const msg = event.data;
  if (msg.type === 'ready') {
    _state.status = 'ready';
    _state.progress = 1;
  } else if (msg.type === 'progress') {
    _state.progress = msg.total > 0 ? msg.loaded / msg.total : 0;
    _state.progressFile = msg.file;
  } else if (msg.type === 'result') {
    const entry = pending.get(msg.id);
    if (entry) {
      pending.delete(msg.id);
      entry.resolve(new Float32Array(msg.embedding));
    }
  } else if (msg.type === 'error') {
    // Reject all pending requests on a generic error
    _state.status = 'error';
    _state.error = msg.message;
    for (const [id, entry] of pending) {
      pending.delete(id);
      entry.reject(new Error(msg.message));
    }
  }
}

interface ModelConfig {
  textEncoder: 'default' | 'custom';
}

export function initModel(): void {
  if (_state.status !== 'idle') return;
  _state.status = 'loading';

  // Load projection weights in parallel with model config
  fetch('/data/text_projection.json')
    .then((r) => r.json() as Promise<ProjectionWeights>)
    .then((w) => {
      weights = w;
    })
    .catch((err) => {
      _state.status = 'error';
      _state.error = `Failed to load projection weights: ${err}`;
    });

  // Load model config to determine which text encoder to use,
  // then start the worker with the appropriate model path.
  Promise.all([
    fetch('/data/model_config.json')
      .then((r) => (r.ok ? (r.json() as Promise<ModelConfig>) : null))
      .catch(() => null),
    import('$lib/inference/worker.ts?worker'),
  ]).then(([modelConfig, { default: WorkerCtor }]) => {
    worker = new WorkerCtor() as Worker;
    worker.addEventListener('message', handleWorkerMessage);
    const modelPath =
      modelConfig?.textEncoder === 'custom' ? '/models/' : undefined;
    worker.postMessage({ type: 'init', modelPath });
  });
}

export async function encodeText(text: string): Promise<Float32Array> {
  if (!worker || !weights) {
    throw new Error('Model not ready');
  }

  const id = nextId++;
  const rawEmbedding = await new Promise<Float32Array>((resolve, reject) => {
    pending.set(id, { resolve, reject });
    worker!.postMessage({ type: 'encode', id, text });
  });

  return projectText(rawEmbedding, weights);
}
