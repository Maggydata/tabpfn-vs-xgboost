import axios from 'axios'

// Adresse de l'API FastAPI (uvicorn api:app)
export const API_URL = 'http://localhost:8000'

const http = axios.create({ baseURL: API_URL })

// ---- Types : forme exacte des JSON produits par pipeline.py ----
export type ModelName = 'xgboost' | 'tabpfn'

export interface Metrics {
  rmse: number
  mae: number
  r2: number
  fit_time: number
  predict_time: number
}

export interface Predictions {
  index: number[]
  y_true: number[]
  y_xgb: number[]
  y_tabpfn: number[]
}

// ---- Routes JSON ----
export async function getMetrics() {
  const res = await http.get<Record<ModelName, Metrics>>('/metrics')
  return res.data
}

export async function getPredictions() {
  const res = await http.get<Predictions>('/predictions')
  return res.data
}

export interface ShapPoint {
  features: Record<string, number>
  y_true: number
  y_xgb: number
  y_tabpfn: number
}

export async function getShapPoints() {
  const res = await http.get<ShapPoint[]>('/shap/points')
  return res.data
}

// ---- Routes images : on renvoie juste l'URL, utilisée dans <img :src="..."> ----
export const beeswarmUrl = (model: ModelName) => `${API_URL}/shap/${model}/beeswarm`

export const waterfallUrl = (model: ModelName, idx: number) =>
  `${API_URL}/shap/${model}/waterfall/${idx}`
