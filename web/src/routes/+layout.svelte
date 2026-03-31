<script lang="ts">
  import '../app.css'
  import favicon from '$lib/assets/favicon.svg'

  let { children } = $props()
</script>

<svelte:head>
  <link rel="icon" href={favicon} />
  <title>Maestroetry: Text-to-Music Search</title>
  <meta name="description" content="Search classical music by description using a contrastive text-audio model." />
</svelte:head>

<div class="app-shell">
  <header class="app-header">
    <span class="header-crown" aria-hidden="true">✦ ✦ ✦</span>
    <h1 class="app-title">Maestroetry</h1>
    <p class="app-subtitle">Text-to-Music Search &amp; Discovery</p>
  </header>

  <main class="app-main">
    {@render children()}
  </main>

  <footer class="app-footer">
    <p>15 public domain recordings &mdash; Musopen &amp; Internet Archive</p>
  </footer>
</div>
