<script lang="ts">
  import type { ScoredTrack } from '$lib/stores/tracks';

  interface Props {
    track: ScoredTrack;
    showScore?: boolean;
  }
  let { track, showScore = true }: Props = $props();

  let audioSrc = $derived(`/audio/${track.filename}`);
</script>

<article class="track-card">
  <div class="track-header">
    <div class="track-meta">
      <h3 class="track-title">{track.title}</h3>
      <div class="track-byline">
        <span class="track-composer">{track.composer}</span>
        <span class="track-era">{track.era}</span>
      </div>
    </div>
    {#if showScore}
      <div class="track-score" title="Cosine similarity">
        <span class="score-label">similarity</span>
        <span class="score-value">{track.score.toFixed(3)}</span>
      </div>
    {/if}
  </div>
  <p class="track-description">{track.description}</p>
  <audio controls src={audioSrc} class="audio-player" preload="none">
    Your browser does not support audio playback.
  </audio>
</article>

<style>
  .track-card {
    background: rgba(255, 255, 255, 0.45);
    border: 1px solid var(--border);
    border-radius: 2px;
    padding: 1.1rem 1.25rem;
    margin-bottom: 0.75rem;
    transition: box-shadow 0.2s;
  }

  .track-card:hover {
    box-shadow: 0 2px 12px rgba(60, 36, 21, 0.12);
  }

  .track-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 0.4rem;
  }

  .track-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.05rem;
    color: var(--burgundy);
    margin: 0 0 0.2rem;
    line-height: 1.3;
  }

  .track-byline {
    display: flex;
    gap: 0.6rem;
    font-size: 0.85rem;
    color: var(--dark-brown);
    opacity: 0.8;
  }

  .track-era::before {
    content: '·';
    margin-right: 0.6rem;
  }

  .track-score {
    text-align: right;
    flex-shrink: 0;
  }

  .score-label {
    display: block;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--dark-brown);
    opacity: 0.6;
  }

  .score-value {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.1rem;
    color: var(--gold);
    font-weight: 700;
  }

  .track-description {
    font-size: 0.9rem;
    color: var(--ink);
    margin: 0.3rem 0 0.75rem;
    line-height: 1.5;
    opacity: 0.85;
  }

  .audio-player {
    width: 100%;
    height: 32px;
    accent-color: var(--burgundy);
  }
</style>
