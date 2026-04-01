/**
 * Web Worker: loads Transformers.js and encodes text queries.
 *
 * Communication protocol (postMessage):
 *
 * Main -> Worker:
 *   { type: 'init' }
 *   { type: 'encode', id: number, text: string }
 *
 * Worker -> Main:
 *   { type: 'ready' }
 *   { type: 'progress', loaded: number, total: number, file: string }
 *   { type: 'result', id: number, embedding: number[] }
 *   { type: 'error', message: string }
 */

import {
  pipeline,
  env,
  type FeatureExtractionPipeline,
  type Tensor,
} from '@huggingface/transformers';

env.allowLocalModels = false;

type InMessage =
  | { type: 'init'; modelPath?: string }
  | { type: 'encode'; id: number; text: string };

type OutMessage =
  | { type: 'ready' }
  | { type: 'progress'; loaded: number; total: number; file: string }
  | { type: 'result'; id: number; embedding: number[] }
  | { type: 'error'; message: string };

function post(msg: OutMessage): void {
  self.postMessage(msg);
}

let extractor: FeatureExtractionPipeline | null = null;

const DEFAULT_MODEL = 'Xenova/all-MiniLM-L6-v2';

async function init(modelPath?: string): Promise<void> {
  const model = modelPath ?? DEFAULT_MODEL;
  const isCustom = model !== DEFAULT_MODEL;
  // Cast via unknown to work around TS2590 (union type too complex)
  const pipelineFn = pipeline as unknown as (
    task: string,
    model: string,
    options: Record<string, unknown>
  ) => Promise<FeatureExtractionPipeline>;
  const options: Record<string, unknown> = {
    progress_callback: (info: unknown) => {
      const p = info as {
        status: string;
        loaded?: number;
        total?: number;
        file?: string;
        name?: string;
      };
      if (p.status === 'downloading' || p.status === 'progress') {
        post({
          type: 'progress',
          loaded: p.loaded ?? 0,
          total: p.total ?? 1,
          file: p.file ?? p.name ?? '',
        });
      }
    },
  };
  // Custom local models are already quantized during export;
  // stock Xenova model needs dtype: 'q8' to request quantized variant.
  if (!isCustom) {
    options.dtype = 'q8';
  }
  if (isCustom) {
    options.local_files_only = true;
  }
  extractor = await pipelineFn('feature-extraction', model, options);
  post({ type: 'ready' });
}

async function encode(id: number, text: string): Promise<void> {
  if (!extractor) {
    post({ type: 'error', message: 'Model not loaded' });
    return;
  }
  try {
    const output = (await extractor(text, {
      pooling: 'mean',
      normalize: true,
    })) as Tensor;
    const embedding = Array.from(output.data as Float32Array);
    post({ type: 'result', id, embedding });
  } catch (err) {
    post({ type: 'error', message: String(err) });
  }
}

self.addEventListener('message', (event: MessageEvent<InMessage>) => {
  const msg = event.data;
  if (msg.type === 'init') {
    init(msg.modelPath).catch((err) =>
      post({ type: 'error', message: `Init failed: ${err}` })
    );
  } else if (msg.type === 'encode') {
    encode(msg.id, msg.text).catch((err) =>
      post({ type: 'error', message: String(err) })
    );
  }
});
