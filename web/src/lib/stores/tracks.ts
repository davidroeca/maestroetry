/**
 * Track metadata and embedding store.
 *
 * Loads tracks.json and audio_embeddings.json from static/data/,
 * provides text-query search and similar-track finding.
 */

import { rankBySimilarity } from '$lib/inference/similarity'

export interface Track {
  id: number
  title: string
  composer: string
  filename: string
  description: string
  era: string
}

export interface ScoredTrack extends Track {
  score: number
}

export interface TracksData {
  tracks: Track[]
  embeddings: Float32Array[]
}

let cache: TracksData | null = null

export async function loadTracks(): Promise<TracksData> {
  if (cache) return cache

  const [tracksResp, embeddingsResp] = await Promise.all([
    fetch('/data/tracks.json'),
    fetch('/data/audio_embeddings.json'),
  ])

  const tracks = (await tracksResp.json()) as Track[]
  const rawEmbeddings = (await embeddingsResp.json()) as number[][]
  const embeddings = rawEmbeddings.map((e) => new Float32Array(e))

  cache = { tracks, embeddings }
  return cache
}

export function searchByEmbedding(
  query: Float32Array,
  data: TracksData,
): ScoredTrack[] {
  const ranked = rankBySimilarity(query, data.embeddings)
  return ranked.map(({ index, score }) => ({ ...data.tracks[index], score }))
}

export function findSimilar(trackId: number, data: TracksData): ScoredTrack[] {
  const query = data.embeddings[trackId]
  return searchByEmbedding(query, data).filter((t) => t.id !== trackId)
}
