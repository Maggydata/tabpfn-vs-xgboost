<script setup lang="ts">
import MetricsTable from './components/MetricsTable.vue'
import PredictionChart from './components/PredictionChart.vue'
import ShapPanel from './components/ShapPanel.vue'
</script>

<template>
  <main>
    <header>
      <h1>TabPFN vs XGBoost</h1>
      <p>
        Régression sur California Housing : 2 000 quartiers, valeur médiane des maisons
        en centaines de milliers de $.
      </p>
    </header>
    <MetricsTable />
    <PredictionChart />
    <ShapPanel />
  </main>
</template>

<style>
body { margin: 0; font-family: system-ui, sans-serif; color: #222; background: #fafafa; }
main { max-width: 1200px; margin: 0 auto; padding: 1rem 1.5rem 3rem; }
section { background: white; border-radius: 8px; padding: 1rem 1.5rem; margin-top: 1.5rem; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08); }
h1 { margin-bottom: 0.2rem; }
header p { color: #666; margin-top: 0; }
</style>
