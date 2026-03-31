<script lang="ts">
  import { onMount } from 'svelte'
  import { modelStatus, initModel, encodeText } from '$lib/stores/model.svelte'
  import { loadTracks, searchByEmbedding } from '$lib/stores/tracks'
  import type { TracksData, ScoredTrack } from '$lib/stores/tracks'
  import SearchBar from '$lib/components/SearchBar.svelte'
  import TrackCard from '$lib/components/TrackCard.svelte'
  import SimilarFinder from '$lib/components/SimilarFinder.svelte'
  import LoadingOverlay from '$lib/components/LoadingOverlay.svelte'
  import Ornament from '$lib/components/Ornament.svelte'

  let tracksData: TracksData | null = $state(null)
  let searchResults: ScoredTrack[] = $state([])
  let searching = $state(false)
  let searchError: string | null = $state(null)
  let hasSearched = $state(false)

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
</script>

{#if ms.status === 'loading'}
  <LoadingOverlay progress={ms.progress} file={ms.progressFile} />
{/if}

<section class="search-section">
  <h2 class="section-heading">Text-to-Music Search</h2>
  <p class="section-sub">
    Describe the mood, instrumentation, or character of the music you want to hear.
  </p>

  <SearchBar
    disabled={ms.status !== 'ready'}
    onSearch={handleSearch}
  />

  {#if ms.status === 'loading'}
    <p class="status-hint">Loading text model&hellip; similar track finder available below.</p>
  {:else if ms.status === 'error'}
    <p class="status-error">Failed to load text model: {ms.error}</p>
  {/if}

  {#if searching}
    <p class="status-hint">Searching&hellip;</p>
  {:else if searchError}
    <p class="status-error">{searchError}</p>
  {:else if hasSearched && searchResults.length > 0}
    <div class="search-results">
      {#each searchResults as track}
        <TrackCard {track} />
      {/each}
    </div>
  {:else if hasSearched}
    <p class="status-hint">No results found.</p>
  {/if}
</section>

{#if tracksData}
  <Ornament variant="divider" />
  <SimilarFinder {tracksData} />
{/if}

<style>
  .search-section {
    max-width: 800px;
    margin: 0 auto;
  }

  .search-results {
    margin-top: 1.5rem;
  }
</style>
