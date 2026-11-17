<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from 'vue'
import Plotly from 'plotly.js-dist-min'
import { getPredictions } from '../api'

const chart = ref<HTMLDivElement | null>(null)
const error = ref('')

onMounted(async () => {
  try {
    const p = await getPredictions()

    // Tri des points de test par valeur réelle croissante
    const order = p.y_true.map((_, i) => i).sort((a, b) => p.y_true[a]! - p.y_true[b]!)
    const x = order.map((_, rank) => rank + 1)
    const pick = (arr: number[]) => order.map((i) => arr[i]!)

    const data: Partial<Plotly.Data>[] = [
      { x, y: pick(p.y_true), name: 'Valeur réelle', mode: 'lines', line: { color: 'black', width: 2 } },
      { x, y: pick(p.y_xgb), name: 'XGBoost', mode: 'markers', marker: { size: 5, opacity: 0.6 } },
      { x, y: pick(p.y_tabpfn), name: 'TabPFN', mode: 'markers', marker: { size: 5, opacity: 0.6 } },
    ]

    const layout: Partial<Plotly.Layout> = {
      xaxis: { title: { text: 'Points de test (triés par valeur réelle)' } },
      yaxis: { title: { text: 'MedHouseVal (centaines de milliers de $)' } },
      // Plafond de la cible à 5.0 (500 000 $)
      shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 5, y1: 5, line: { dash: 'dot', color: 'gray' } }],
      annotations: [{ xref: 'paper', x: 0, y: 5, text: 'plafond 500 k$', showarrow: false, yshift: 10, xanchor: 'left' }],
      legend: { orientation: 'h', y: 1.1 },
      margin: { t: 40 },
    }

    await Plotly.newPlot(chart.value!, data, layout, { responsive: true })
  } catch {
    error.value = "Impossible de charger les prédictions : l'API est-elle lancée ?"
  }
})

// Libère la mémoire de Plotly quand le composant disparaît
onBeforeUnmount(() => {
  if (chart.value) Plotly.purge(chart.value)
})
</script>

<template>
  <section>
    <h2>Valeurs réelles vs prédictions</h2>
    <p v-if="error" class="error">{{ error }}</p>
    <div ref="chart" class="chart"></div>
  </section>
</template>

<style scoped>
.chart { width: 100%; height: 450px; }
.error { color: #c62828; }
</style>
