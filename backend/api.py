"""API : sert les résultats pré-calculés par pipeline.py (aucun modèle chargé ici).

Lancement : uvicorn api:app --reload   (depuis backend/)
"""
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

RESULTS_DIR = Path(__file__).parent / "results"
MODELS = ("xgboost", "tabpfn")

app = FastAPI(
    title="TabPFN vs XGBoost",
    description="Résultats pré-calculés : métriques, prédictions et graphiques SHAP.",
)

# Autorise le frontend Vue (serveur de dev Vite) à appeler l'API depuis le navigateur
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def read_json(name):
    """Lit un fichier JSON de results/ ; erreur 503 s'il n'a pas encore été généré."""
    path = RESULTS_DIR / name
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"{name} absent : lancer d'abord python pipeline.py",
        )
    return json.loads(path.read_text())


@app.get("/metrics")
def get_metrics():
    """RMSE, MAE, R², temps de fit et de predict pour les deux modèles."""
    return read_json("metrics.json")


@app.get("/predictions")
def get_predictions():
    """Valeurs réelles et prédictions des deux modèles sur les 400 points de test."""
    return read_json("predictions.json")


@app.get("/shap/points")
def get_shap_points():
    """Les points expliqués par SHAP : valeurs des variables, valeur réelle et prédictions."""
    shap_values = read_json("shap_values.json")
    preds = read_json("predictions.json")
    pos = {idx: i for i, idx in enumerate(preds["index"])}  # index du quartier → position dans le test
    points = []
    for idx, row in zip(shap_values["index"], shap_values["data"]):
        i = pos[idx]
        points.append({
            "features": dict(zip(shap_values["feature_names"], row)),
            "y_true": preds["y_true"][i],
            "y_xgb": preds["y_xgb"][i],
            "y_tabpfn": preds["y_tabpfn"][i],
        })
    return points


def check_model(model):
    """Erreur 404 si le modèle demandé dans l'URL n'existe pas."""
    if model not in MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Modèle inconnu : {model}. Choix possibles : {', '.join(MODELS)}",
        )


@app.get("/shap/{model}/beeswarm")
def get_beeswarm(model: str):
    """Beeswarm SHAP du modèle (100 points expliqués)."""
    check_model(model)
    path = RESULTS_DIR / model / "beeswarm.png"
    if not path.exists():
        raise HTTPException(503, f"{model}/beeswarm.png absent : lancer d'abord python pipeline.py")
    return FileResponse(path, media_type="image/png")


@app.get("/shap/{model}/waterfall/{idx}")
def get_waterfall(model: str, idx: int):
    """Waterfall SHAP du point idx (position dans les points expliqués, à partir de 0)."""
    check_model(model)
    path = RESULTS_DIR / model / f"waterfall_{idx}.png"
    if not path.exists():
        n = len(list((RESULTS_DIR / model).glob("waterfall_*.png")))
        raise HTTPException(404, f"Pas de waterfall pour le point {idx} : index valides de 0 à {n - 1}")
    return FileResponse(path, media_type="image/png")
