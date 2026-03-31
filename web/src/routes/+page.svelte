<script lang="ts">
  import { onMount } from 'svelte'
  import { modelStatus, initModel, encodeText } from '$lib/stores/model.svelte'
  import { loadTracks, searchByEmbedding } from '$lib/stores/tracks'
  import type { TracksData, ScoredTrack } from '$lib/stores/tracks'
  import SearchBar from '$lib/components/SearchBar.svelte'
  import TrackCard from '$lib/components/TrackCard.svelte'
  import SimilarFinder from '$lib/components/SimilarFinder.svelte'
  import LoadingOverlay from '$lib/components/LoadingOverlay.svelte'

  let tracksData: TracksData | null = $state(null)
  let searchResults: ScoredTrack[] = $state([])
  let searching = $state(false)
  let searchError: string | null = $state(null)
  let hasSearched = $state(false)
  let activeTab: 'search' | 'similar' = $state('search')

  const ms = modelStatus()

  onMount(async () => {
    tracksData = await loadTracks()
    initModel()
  })

  async function handleSearch(query: string) {
    searching = true
    searchError = null
    hasSearched = true
    try {
      const embedding = await encodeText(query)
      if (tracksData) {
        searchResults = searchByEmbedding(embedding, tracksData)
      }
    } catch (err) {
      searchError = String(err)
    } finally {
      searching = false
    }
  }

  function clearSearch() {
    searchResults = []
    hasSearched = false
    searchError = null
  }

  const PROMPT_CHIPS = [
    'slow and searching solo piano, dark and melancholy, introspective',
    'bright and energetic strings, lively and celebratory, full of vitality',
    'gentle and dreamy piano, lyrical and floating, quietly expressive',
    'building orchestral tension, ominous and relentless, growing darker',
    'delicate and whimsical, magical lightness with a playful character',
    'aggressive and martial orchestra, powerful and unrelenting',
    'warm contemplative solo cello, gently flowing and unhurried',
    'syncopated and upbeat ragtime piano, rhythmic and full of joy',
  ]
</script>

{#if ms.status === 'loading'}
  <LoadingOverlay progress={ms.progress} file={ms.progressFile} />
{/if}

<div class="tabs-wrapper">
  <nav class="tab-nav" aria-label="Search mode">
    <button
      class="tab-btn"
      class:active={activeTab === 'search'}
      onclick={() => activeTab = 'search'}
    >
      Text Search
    </button>
    <span class="tab-sep" aria-hidden="true">✦</span>
    <button
      class="tab-btn"
      class:active={activeTab === 'similar'}
      onclick={() => activeTab = 'similar'}
    >
      Similar Finder
    </button>
  </nav>

  {#if activeTab === 'search'}
    <section class="tab-panel search-section">
      <p class="section-sub">
        Describe the mood, instrumentation, or character of the music you seek.
      </p>

      <SearchBar
        disabled={ms.status !== 'ready'}
        onSearch={handleSearch}
      />

      {#if !hasSearched}
        <div class="prompt-chips">
          {#each PROMPT_CHIPS as chip}
            <button
              class="chip"
              disabled={ms.status !== 'ready'}
              onclick={() => handleSearch(chip)}
            >{chip}</button>
          {/each}
        </div>
      {/if}

      {#if ms.status === 'loading'}
        <p class="status-hint">Loading text model&hellip; similar track finder available in the other tab.</p>
      {:else if ms.status === 'error'}
        <p class="status-error">Failed to load text model: {ms.error}</p>
      {/if}

      {#if searching}
        <p class="status-hint">Searching&hellip;</p>
      {:else if searchError}
        <p class="status-error">{searchError}</p>
      {:else if hasSearched && searchResults.length > 0}
        <div class="results-header">
          <span class="results-count">{searchResults.length} results</span>
          <button class="clear-btn" onclick={clearSearch}>Clear</button>
        </div>
        <div class="search-results">
          {#each searchResults as track}
            <TrackCard {track} />
          {/each}
        </div>
      {:else if hasSearched}
        <p class="status-hint">No results found.</p>
      {/if}
    </section>
  {:else}
    <section class="tab-panel">
      {#if tracksData}
        <SimilarFinder {tracksData} />
      {:else}
        <p class="status-hint">Loading tracks&hellip;</p>
      {/if}
    </section>
  {/if}
</div>

<style>
  .tabs-wrapper {
    max-width: 800px;
    margin: 0 auto;
  }

  .tab-nav {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1.25rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
  }

  .tab-btn {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.05rem;
    color: var(--dark-brown);
    background: none;
    border: none;
    border-bottom: 2px solid transparent;
    padding: 0.5rem 0.25rem;
    margin-bottom: -1px;
    cursor: pointer;
    opacity: 0.65;
    transition: color 0.2s, opacity 0.2s, border-color 0.2s;
  }

  .tab-btn:hover {
    opacity: 0.9;
    color: var(--burgundy);
  }

  .tab-btn.active {
    color: var(--burgundy);
    border-bottom-color: var(--burgundy);
    opacity: 1;
  }

  .tab-sep {
    color: var(--gold);
    font-size: 0.65rem;
    opacity: 0.7;
    user-select: none;
  }

  .tab-panel {
    /* panels share the same container space */
  }

  .search-section {
    /* inherits from tabs-wrapper max-width */
  }

  .section-sub {
    text-align: center;
    font-size: 0.92rem;
    color: var(--dark-brown);
    opacity: 0.8;
    margin: 0 0 1.5rem;
  }

  .status-hint {
    text-align: center;
    font-size: 0.9rem;
    color: var(--dark-brown);
    opacity: 0.7;
    margin-top: 1rem;
  }

  .status-error {
    text-align: center;
    font-size: 0.9rem;
    color: var(--burgundy);
    margin-top: 1rem;
  }

  .results-header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    margin-top: 1.5rem;
    margin-bottom: 0.5rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--border);
  }

  .results-count {
    font-size: 0.85rem;
    color: var(--dark-brown);
    opacity: 0.65;
    font-style: italic;
  }

  .clear-btn {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 0.85rem;
    color: var(--dark-brown);
    background: none;
    border: none;
    padding: 0;
    cursor: pointer;
    opacity: 0.55;
    text-decoration: underline;
    text-underline-offset: 3px;
    transition: opacity 0.15s, color 0.15s;
  }

  .clear-btn:hover {
    opacity: 0.9;
    color: var(--burgundy);
  }

  .prompt-chips {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 0.4rem;
    margin-top: 1rem;
  }

  .chip {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 0.82rem;
    font-style: italic;
    color: var(--dark-brown);
    background: rgba(255, 255, 255, 0.4);
    border: 1px solid var(--border);
    border-radius: 2px;
    padding: 0.25rem 0.65rem;
    cursor: pointer;
    opacity: 0.75;
    transition: background 0.15s, border-color 0.15s, opacity 0.15s, color 0.15s;
  }

  .chip:hover:not(:disabled) {
    background: rgba(197, 165, 90, 0.15);
    border-color: var(--gold);
    opacity: 1;
    color: var(--ink);
  }

  .chip:disabled {
    opacity: 0.35;
    cursor: not-allowed;
  }

  .search-results {
    margin-top: 0.5rem;
  }
</style>
