"""Pipeline : entraîne les deux modèles et pré-calcule tous les résultats dans results/.

Usage : python pipeline.py [--n-shap 100]
"""
import argparse
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # pas d'affichage : on écrit des PNG

import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

from data import N_SHAP, RANDOM_STATE, load_split

# ⚠️ xgboost et tabpfn (torch) ne doivent JAMAIS être importés dans le même processus :
# chacun embarque son propre OpenMP → segfault ou blocage sur Mac.
# On les importe uniquement dans les fonctions, exécutées dans des processus séparés.

RESULTS_DIR = Path(__file__).parent / "results"
SHAP_MAX_EVALS = 100  # TabPFN : 500 par défaut, réduit pour tenir en ~1 h 30 sur M1
CHECKPOINT = RESULTS_DIR / "_tabpfn_shap_checkpoint.npz"


def train_xgb(X_train, y_train):
    """Même tuning que le notebook 01."""
    from xgboost import XGBRegressor

    param_grid = {
        "n_estimators": [200, 500],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1],
    }
    search = GridSearchCV(
        XGBRegressor(random_state=RANDOM_STATE),
        param_grid, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    search.fit(X_train, y_train)
    model = XGBRegressor(random_state=RANDOM_STATE, **search.best_params_)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    return model, time.perf_counter() - t0


def train_tabpfn(X_train, y_train):
    """Paramètres par défaut, GPU Apple (MPS)."""
    from tabpfn import TabPFNRegressor

    model = TabPFNRegressor(device="mps", random_state=RANDOM_STATE)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    return model, time.perf_counter() - t0


def evaluate(model, fit_time, X_test, y_test):
    """Mêmes métriques que dans les notebooks. Renvoie (métriques, prédictions)."""
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    predict_time = time.perf_counter() - t0
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "fit_time": fit_time,
        "predict_time": predict_time,
    }
    return metrics, y_pred


def shap_xgb(model, X_background, X_shap):
    explainer = shap.TreeExplainer(model, X_background)
    return explainer(X_shap)


def shap_tabpfn(model, X_background, X_shap):
    """Point par point, avec reprise : un checkpoint est écrit après chaque point."""
    explainer = shap.PermutationExplainer(model.predict, X_background, seed=RANDOM_STATE)
    index = X_shap.index.to_numpy()
    values, base_values = [], []

    if CHECKPOINT.exists():
        with np.load(CHECKPOINT) as ckpt:
            n_done = len(ckpt["values"])
            if np.array_equal(ckpt["index"][:n_done], index[:n_done]):
                values, base_values = list(ckpt["values"]), list(ckpt["base_values"])
                print(f"  reprise au point {n_done + 1}/{len(X_shap)}", flush=True)

    for i in range(len(values), len(X_shap)):
        t0 = time.perf_counter()
        exp = explainer(X_shap.iloc[[i]], max_evals=SHAP_MAX_EVALS, silent=True)
        values.append(exp.values[0])
        base_values.append(exp.base_values[0])
        np.savez(CHECKPOINT, index=index, values=values, base_values=base_values)
        print(f"  point {i + 1}/{len(X_shap)} : {time.perf_counter() - t0:.0f} s", flush=True)

    CHECKPOINT.unlink()  # calcul terminé : le checkpoint n'est plus utile
    return shap.Explanation(
        values=np.array(values), base_values=np.array(base_values),
        data=X_shap.to_numpy(), feature_names=list(X_shap.columns),
    )


def check_additivity(name, exp, model, X_shap, atol):
    recon = exp.base_values + exp.values.sum(axis=1)
    ok = np.allclose(recon, model.predict(X_shap), atol=atol)
    print(f"  additivité {name} : {'OK' if ok else 'ÉCHEC'}", flush=True)


def run_xgb(d, X_shap):
    """Tout XGBoost dans un processus isolé."""
    print("[XGBoost] entraînement…", flush=True)
    model, fit_time = train_xgb(d["X_train"], d["y_train"])
    metrics, y_pred = evaluate(model, fit_time, d["X_test"], d["y_test"])
    print(f"  fit : {fit_time:.2f} s, RMSE : {metrics['rmse']:.3f}", flush=True)
    exp = shap_xgb(model, d["X_background"], X_shap)
    check_additivity("xgboost", exp, model, X_shap, atol=1e-4)
    return metrics, y_pred, exp.values, np.asarray(exp.base_values)


def run_tabpfn(d, X_shap):
    """Tout TabPFN dans un processus isolé."""
    print("[TabPFN] entraînement…", flush=True)
    model, fit_time = train_tabpfn(d["X_train"], d["y_train"])
    metrics, y_pred = evaluate(model, fit_time, d["X_test"], d["y_test"])
    print(f"  fit : {fit_time:.2f} s, RMSE : {metrics['rmse']:.3f}", flush=True)
    print(f"[TabPFN] SHAP ({len(X_shap)} points, ~50 s/point)…", flush=True)
    exp = shap_tabpfn(model, d["X_background"], X_shap)
    check_additivity("tabpfn", exp, model, X_shap, atol=1e-3)
    return metrics, y_pred, exp.values, exp.base_values


def run_isolated(fn, *args):
    """Exécute fn dans un processus Python neuf (« spawn ») et renvoie son résultat."""
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
        return pool.submit(fn, *args).result()


def to_explanation(values, base_values, X_shap):
    return shap.Explanation(values=values, base_values=base_values,
                            data=X_shap.to_numpy(), feature_names=list(X_shap.columns))


def save_json(obj, name):
    path = RESULTS_DIR / name
    path.write_text(json.dumps(obj, indent=2))
    print(f"  → {path.name}")


def save_shap(explanations, X_shap):
    """Un seul JSON pour les deux modèles (mêmes points, mêmes variables)."""
    out = {
        "feature_names": list(X_shap.columns),
        "index": X_shap.index.tolist(),
        "data": X_shap.to_numpy().tolist(),
    }
    for name, exp in explanations.items():
        out[name] = {
            "base_values": np.asarray(exp.base_values).tolist(),
            "values": exp.values.tolist(),
        }
    save_json(out, "shap_values.json")


def save_plots(name, exp):
    """results/<modèle>/beeswarm.png et results/<modèle>/waterfall_<i>.png"""
    out_dir = RESULTS_DIR / name
    out_dir.mkdir(exist_ok=True)

    shap.plots.beeswarm(exp, show=False)
    plt.savefig(out_dir / "beeswarm.png", dpi=120, bbox_inches="tight")
    plt.close("all")

    for i in range(len(exp)):
        shap.plots.waterfall(exp[i], show=False)
        plt.savefig(out_dir / f"waterfall_{i}.png", dpi=120, bbox_inches="tight")
        plt.close("all")
    print(f"  → {name}/ : beeswarm + {len(exp)} waterfalls")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-shap", type=int, default=N_SHAP,
                        help="nombre de points expliqués par SHAP (5 pour un test rapide)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    d = load_split()
    X_shap = d["X_shap"].iloc[: args.n_shap]

    xgb_metrics, y_xgb, xgb_vals, xgb_base = run_isolated(run_xgb, d, X_shap)
    tab_metrics, y_tab, tab_vals, tab_base = run_isolated(run_tabpfn, d, X_shap)

    print("Sauvegarde…")
    save_json({"xgboost": xgb_metrics, "tabpfn": tab_metrics}, "metrics.json")
    save_json({
        "index": d["X_test"].index.tolist(),
        "y_true": d["y_test"].tolist(),
        "y_xgb": y_xgb.tolist(),
        "y_tabpfn": y_tab.tolist(),
    }, "predictions.json")
    shap_x = to_explanation(xgb_vals, xgb_base, X_shap)
    shap_t = to_explanation(tab_vals, tab_base, X_shap)
    save_shap({"xgboost": shap_x, "tabpfn": shap_t}, X_shap)
    save_plots("xgboost", shap_x)
    save_plots("tabpfn", shap_t)
    print("Terminé.")


if __name__ == "__main__":
    main()
