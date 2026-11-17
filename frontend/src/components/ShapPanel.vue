<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { beeswarmUrl, getShapPoints, waterfallUrl, type ModelName, type ShapPoint } from '../api'

const MODELS: { key: ModelName; label: string }[] = [
  { key: 'xgboost', label: 'XGBoost' },
  { key: 'tabpfn', label: 'TabPFN' },
]

const points = ref<ShapPoint[]>([])
const idx = ref(0) // point sélectionné (position 0 à 99)
const error = ref('')
const point = computed(() => points.value[idx.value])

onMounted(async () => {
  try {
    points.value = await getShapPoints()
  } catch {
    error.value = "Impossible de charger les points SHAP : l'API est-elle lancée ?"
  }
})

const fmt = (v: number) => v.toFixed(2)
</script>

<template>
  <section>
    <h2>Interprétation SHAP</h2>
    <p v-if="error" class="error">{{ error }}</p>

    <h3>Vue globale : beeswarm sur {{ points.length || 100 }} points</h3>
    <div class="pair">
      <figure v-for="m in MODELS" :key="m.key">
        <figcaption>{{ m.label }}</figcaption>
        <img :src="beeswarmUrl(m.key)" :alt="`Beeswarm SHAP ${m.label}`" />
      </figure>
    </div>

    <h3>Détail d'une prédiction : waterfall</h3>
    <div v-if="point" class="selector">
      <button :disabled="idx === 0" @click="idx--">◀</button>
      <select v-model.number="idx">
        <option v-for="(p, i) in points" :key="i" :value="i">
          Point {{ i }} (réel {{ fmt(p.y_true) }})
        </option>
      </select>
      <button :disabled="idx === points.length - 1" @click="idx++">▶</button>
      <span class="values">
        Réel : <b>{{ fmt(point.y_true) }}</b> · XGBoost : <b>{{ fmt(point.y_xgb) }}</b> ·
        TabPFN : <b>{{ fmt(point.y_tabpfn) }}</b>
      </span>
    </div>

    <div class="pair">
      <figure v-for="m in MODELS" :key="m.key">
        <figcaption>{{ m.label }}</figcaption>
        <img :src="waterfallUrl(m.key, idx)" :alt="`Waterfall SHAP ${m.label}, point ${idx}`" />
      </figure>
    </div>

    <details v-if="point">
      <summary>Valeurs des variables du point {{ idx }}</summary>
      <table>
        <tr v-for="(v, name) in point.features" :key="name">
          <td>{{ name }}</td>
          <td>{{ fmt(v) }}</td>
        </tr>
      </table>
    </details>
  </section>
</template>

<style scoped>
.pair { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
figure { margin: 0; }
figcaption { font-weight: bold; margin-bottom: 0.3rem; }
img { width: 100%; height: auto; }
.selector { display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 1rem; }
.values { margin-left: 1rem; }
td { padding: 0.2rem 1rem 0.2rem 0; }
.error { color: #c62828; }
</style>
