/**
 * TypeScript reimplementation of ProjectionHead forward pass.
 *
 * Matches the Python model exactly at inference time (no dropout).
 * Architecture: Linear -> ReLU -> ... -> Linear -> L2 normalize
 *
 * The weight format matches the output of export_web_data.py:
 *   { layers: [ { weight: number[][], bias: number[] }, ... ] }
 * where each layer's weight is shape (out_features, in_features).
 */

export interface LayerWeights {
  weight: number[][];
  bias: number[];
}

export interface ProjectionWeights {
  layers: LayerWeights[];
}

function linearForward(
  x: Float32Array,
  weight: number[][],
  bias: number[]
): Float32Array {
  const outDim = weight.length;
  const out = new Float32Array(outDim);
  for (let i = 0; i < outDim; i++) {
    let sum = bias[i];
    const row = weight[i];
    for (let j = 0; j < x.length; j++) {
      sum += x[j] * row[j];
    }
    out[i] = sum;
  }
  return out;
}

function relu(x: Float32Array): Float32Array {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) {
    out[i] = x[i] > 0 ? x[i] : 0;
  }
  return out;
}

function l2normalize(x: Float32Array): Float32Array {
  let norm = 0;
  for (let i = 0; i < x.length; i++) norm += x[i] * x[i];
  norm = Math.sqrt(norm);
  if (norm < 1e-12) return x.slice();
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = x[i] / norm;
  return out;
}

/**
 * Run the text projection head on a raw text encoder embedding.
 *
 * @param x - Raw encoder output (Float32Array of length d_in, e.g. 384)
 * @param weights - Loaded from text_projection.json
 * @returns L2-normalized projected embedding (Float32Array of length 256)
 */
export function projectText(
  x: Float32Array,
  weights: ProjectionWeights
): Float32Array {
  let h: Float32Array = x;
  const { layers } = weights;
  for (let i = 0; i < layers.length; i++) {
    const { weight, bias } = layers[i];
    h = linearForward(h, weight, bias);
    if (i < layers.length - 1) {
      h = relu(h);
    }
  }
  return l2normalize(h);
}
