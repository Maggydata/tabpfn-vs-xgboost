<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { getMetrics, type Metrics, type ModelName } from '../api'

const metrics = ref<Record<ModelName, Metrics> | null>(null)
const error = ref('')

onMounted(async () => {
  try {
    metrics.value = await getMetrics()
  } catch {
    error.value = "Impossible de joindre l'API : est-elle lancée (uvicorn api:app) ?"
  }
})

const MODELS: { key: ModelName; label: string }[] = [
  { key: 'xgboost', label: 'XGBoost' },
  { key: 'tabpfn', label: 'TabPFN' },
]

const formatTime = (s: number) => (s < 1 ? `${(s * 1000).toFixed(0)} ms` : `${s.toFixed(2)} s`)

// Une ligne par métrique ; "better" indique si la meilleure valeur est la plus basse ou la plus haute
const ROWS: { key: keyof Metrics; label: string; better: 'low' | 'high'; fmt: (v: number) => string }[] = [
  { key: 'rmse', label: 'RMSE', better: 'low', fmt: (v) => v.toFixed(3) },
  { key: 'mae', label: 'MAE', better: 'low', fmt: (v) => v.toFixed(3) },
  { key: 'r2', label: 'R²', better: 'high', fmt: (v) => v.toFixed(3) },
  { key: 'fit_time', label: 'Temps de fit', better: 'low', fmt: formatTime },
  { key: 'predict_time', label: 'Temps de predict', better: 'low', fmt: formatTime },
]

function isBest(row: (typeof ROWS)[number], model: ModelName) {
  const m = metrics.value!
  const [a, b] = [m[model][row.key], m[model === 'xgboost' ? 'tabpfn' : 'xgboost'][row.key]]
  return row.better === 'low' ? a < b : a > b
}
</script>

<template>
  <section>
    <h2>Métriques sur le jeu de test</h2>
    <p v-if="error" class="error">{{ error }}</p>
    <p v-else-if="!metrics">Chargement…</p>
    <table v-else>
      <thead>
        <tr>
          <th>Métrique</th>
          <th v-for="m in MODELS" :key="m.key">{{ m.label }}</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="row in ROWS" :key="row.key">
          <td>{{ row.label }}</td>
          <td v-for="m in MODELS" :key="m.key" :class="{ best: isBest(row, m.key) }">
            {{ row.fmt(metrics[m.key][row.key]) }}
          </td>
        </tr>
      </tbody>
    </table>
  </section>
</template>

<style scoped>
table { border-collapse: collapse; }
th, td { padding: 0.4rem 1rem; border-bottom: 1px solid #ddd; text-align: right; }
th:first-child, td:first-child { text-align: left; }
.best { font-weight: bold; color: #2e7d32; }
.error { color: #c62828; }
</style>
