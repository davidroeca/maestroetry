<script lang="ts">
  interface Props {
    progress?: number
    file?: string
  }
  let { progress = 0, file = '' }: Props = $props()

  let pct = $derived(Math.round(progress * 100))
</script>

<div class="overlay" role="status" aria-live="polite">
  <div class="overlay-box">
    <div class="overlay-ornament" aria-hidden="true">❧ ✦ ❧</div>
    <h2 class="overlay-title">Loading Musical Intelligence</h2>
    <p class="overlay-sub">Fetching the text understanding model&hellip;</p>
    {#if file}
      <p class="overlay-file">{file}</p>
    {/if}
    <div
      class="progress-track"
      role="progressbar"
      aria-valuenow={pct}
      aria-valuemin={0}
      aria-valuemax={100}
    >
      <div class="progress-fill" style="width: {pct}%"></div>
    </div>
    <p class="progress-pct">{pct}%</p>
    <p class="overlay-note">
      Text search available once loaded.<br />
      Similar track finder works immediately.
    </p>
    <div class="overlay-ornament" aria-hidden="true">❧ ✦ ❧</div>
  </div>
</div>

<style>
  .overlay {
    position: fixed;
    inset: 0;
    background: rgba(44, 24, 16, 0.75);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 50;
    backdrop-filter: blur(2px);
  }

  .overlay-box {
    background: var(--cream);
    border: 2px solid var(--gold);
    border-radius: 2px;
    padding: 2.5rem 3rem;
    max-width: 420px;
    width: 90%;
    text-align: center;
    box-shadow: 0 8px 40px rgba(44, 24, 16, 0.4);
  }

  .overlay-ornament {
    color: var(--gold);
    letter-spacing: 0.3em;
    font-size: 1rem;
    margin: 0.5rem 0;
  }

  .overlay-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.4rem;
    color: var(--burgundy);
    margin: 0.75rem 0 0.5rem;
  }

  .overlay-sub {
    color: var(--dark-brown);
    margin: 0 0 0.75rem;
    font-size: 0.95rem;
  }

  .overlay-file {
    font-size: 0.78rem;
    color: var(--dark-brown);
    opacity: 0.7;
    margin: 0 0 0.75rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .progress-track {
    background: rgba(197, 165, 90, 0.2);
    border: 1px solid var(--border);
    border-radius: 2px;
    height: 8px;
    overflow: hidden;
    margin: 0.5rem 0;
  }

  .progress-fill {
    height: 100%;
    background: var(--gold);
    transition: width 0.3s ease;
  }

  .progress-pct {
    font-size: 0.85rem;
    color: var(--dark-brown);
    margin: 0.25rem 0 1rem;
  }

  .overlay-note {
    font-size: 0.82rem;
    color: var(--dark-brown);
    opacity: 0.8;
    line-height: 1.5;
    margin: 0 0 0.5rem;
  }
</style>
