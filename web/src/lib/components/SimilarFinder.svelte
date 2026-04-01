<script lang="ts">
  import type { TracksData, ScoredTrack } from '$lib/stores/tracks'
  import { findSimilar } from '$lib/stores/tracks'
  import TrackCard from './TrackCard.svelte'

  interface Props {
    tracksData: TracksData
  }
  let { tracksData }: Props = $props()

  let selectedId: number | null = $state(null)
  let results: ScoredTrack[] = $state([])

  function selectTrack(id: number) {
    if (selectedId === id) {
      selectedId = null
      results = []
    } else {
      selectedId = id
      results = findSimilar(id, tracksData)
    }
  }
</script>

<section class="similar-finder">
  <p class="section-hint">
    Select a track to find the most similar pieces by embedding distance. Works
    without loading the text model.
  </p>

  <div class="track-selector-grid">
    {#each tracksData.tracks as track}
      <button
        class="selector-btn"
        class:active={selectedId === track.id}
        onclick={() => selectTrack(track.id)}
      >
        <span class="selector-title">{track.title}</span>
        <span class="selector-composer">{track.composer}</span>
      </button>
    {/each}
  </div>

  {#if results.length > 0 && selectedId !== null}
    <div class="similar-results">
      <h3 class="results-heading">
        Similar to <em>{tracksData.tracks[selectedId].title}</em>
      </h3>
      {#each results as track}
        <TrackCard {track} />
      {/each}
    </div>
  {/if}
</section>

<style>
  .similar-finder {
    max-width: 800px;
    margin: 0 auto;
  }

  .section-hint {
    text-align: center;
    font-size: 0.92rem;
    color: var(--dark-brown);
    opacity: 0.8;
    margin: 0 0 1.5rem;
  }

  .track-selector-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 0.5rem;
    margin-bottom: 2rem;
  }

  .selector-btn {
    background: rgba(255, 255, 255, 0.5);
    border: 1px solid var(--border);
    border-radius: 2px;
    padding: 0.6rem 0.75rem;
    text-align: left;
    cursor: pointer;
    transition:
      background 0.15s,
      border-color 0.15s;
  }

  .selector-btn:hover {
    background: rgba(197, 165, 90, 0.15);
    border-color: var(--gold);
  }

  .selector-btn.active {
    background: rgba(114, 47, 55, 0.1);
    border-color: var(--burgundy);
  }

  .selector-title {
    display: block;
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 0.85rem;
    color: var(--ink);
    line-height: 1.3;
  }

  .selector-composer {
    display: block;
    font-size: 0.75rem;
    color: var(--dark-brown);
    opacity: 0.7;
    margin-top: 0.1rem;
  }

  .similar-results {
    margin-top: 1rem;
  }

  .results-heading {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.1rem;
    color: var(--dark-brown);
    text-align: center;
    margin: 0 0 1rem;
  }

  .results-heading em {
    color: var(--burgundy);
    font-style: italic;
  }
</style>
