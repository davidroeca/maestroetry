/**
 * Similarity scoring over L2-normalized embeddings.
 *
 * Since all embeddings are L2-normalized to unit length,
 * dot product equals cosine similarity.
 */

export function dotProduct(a: Float32Array, b: Float32Array): number {
  let sum = 0
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i]
  return sum
}

export interface RankedResult {
  index: number
  score: number
}

/**
 * Rank all embeddings by similarity to the query.
 *
 * @param query - L2-normalized query embedding
 * @param embeddings - Array of L2-normalized candidate embeddings
 * @returns Indices sorted descending by similarity score
 */
export function rankBySimilarity(
  query: Float32Array,
  embeddings: Float32Array[],
): RankedResult[] {
  return embeddings
    .map((emb, index) => ({ index, score: dotProduct(query, emb) }))
    .sort((a, b) => b.score - a.score)
}
