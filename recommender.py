# recommender.py
# ─────────────────────────────────────────────────────────────────────────────
# Motor de inferencia.  Lee los artefactos generados por train_and_save.py
# y expone las funciones que usa la API FastAPI.
# NO reentrena en startup: carga pesos → listo en < 1 segundo.
# ─────────────────────────────────────────────────────────────────────────────

import os
from typing import Dict, List, Optional

import joblib
import numpy as np
import torch
import torch.nn as nn

# ─── Rutas ──────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(__file__)
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_PATH    = os.path.join(ARTIFACTS_DIR, "travel_recomendations.pth")
STATE_PATH    = os.path.join(ARTIFACTS_DIR, "travel_recomendations_state.joblib")

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

PREF_TO_TYPE: Dict[str, str] = {
    "Beaches": "Beach", "Historical": "Historical",
    "Nature": "Nature", "Adventure": "Adventure", "City": "City",
}

# ──────────────────────────────────────────────────────────────────────────────
# Mapeo de mejor época de viaje (BestTimeToVisit) a meses (1-12).
# El mes se usa como FILTRO DE CONTENIDO (no como feature del modelo).
# Razón: análisis empírico mostró que VisitDate carece de señal predictiva:
#   - ANOVA rating vs mes: F=0.047, p=0.9545 (independientes)
#   - Tasa coincidencia vis-vs-mejor-temporada: 54.85% << baseline aleatorio 72.6%
# ──────────────────────────────────────────────────────────────────────────────
MESES_DESTINO: Dict[str, set] = {
    "Sep-Mar":    {9, 10, 11, 12, 1, 2, 3},
    "Nov-Feb":    {11, 12, 1, 2},
    "Apr-Jun":    {4, 5, 6},
    "Nov-Mar":    {11, 12, 1, 2, 3},
    "Oct-Mar":    {10, 11, 12, 1, 2, 3},
    "Mar-Jun":    {3, 4, 5, 6},
    "Jul-Sep":    {7, 8, 9},
    "Sep-Nov":    {9, 10, 11},
    "Year-round": set(range(1, 13)),
}


# ══════════════════════════════════════════════════════════════════════════════
# Arquitectura  (debe coincidir EXACTAMENTE con train_and_save.py)
# ══════════════════════════════════════════════════════════════════════════════
class HybridNCF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, n_user_features,
                 layers, dropout=0.3):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        dims = [embedding_dim * 2 + n_user_features] + layers
        self.fc_layers = nn.ModuleList(
            [nn.Linear(dims[k], dims[k+1]) for k in range(len(dims)-1)]
        )
        self.output_layer = nn.Linear(dims[-1], 1)
        self.activation   = nn.ReLU()
        self.dropout      = nn.Dropout(dropout)

    def forward(self, user_input, item_input, user_features):
        u = self.user_embedding(user_input)
        i = self.item_embedding(item_input)
        x = torch.cat([u, i, user_features], dim=-1)
        for layer in self.fc_layers:
            x = self.dropout(self.activation(layer(x)))
        return self.output_layer(x).squeeze()


# ══════════════════════════════════════════════════════════════════════════════
# Motor
# ══════════════════════════════════════════════════════════════════════════════
class TravelRecommender:

    def __init__(self):
        self.model: Optional[HybridNCF] = None
        self.ready: bool = False
        self._state: Optional[dict] = None  # estado completo del joblib

    # ─────────────────────────────────────────────────────────────────────
    # Startup: sólo carga artefactos
    # ─────────────────────────────────────────────────────────────────────
    def startup(self):
        if not os.path.exists(MODEL_PATH) or not os.path.exists(STATE_PATH):
            raise FileNotFoundError(
                "Artefactos no encontrados. "
                "Ejecuta primero:  python train_and_save.py"
            )

        import time
        t0 = time.perf_counter()

        print("[recommender] Cargando estado de inferencia…")
        self._state = joblib.load(STATE_PATH)
        s = self._state  # alias corto

        print("[recommender] Reconstruyendo arquitectura NCF y cargando pesos…")
        self.model = HybridNCF(
            n_users         = s["n_users"],
            n_items         = s["n_items"],
            embedding_dim   = s["embedding_dim"],
            n_user_features = s["n_user_features"],
            layers          = s["layers"],
            dropout         = s["dropout"],
        ).to(DEVICE)
        self.model.load_state_dict(
            torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        )
        self.model.eval()

        elapsed = time.perf_counter() - t0
        self.ready = True
        print(f"[recommender] OK - Listo en {elapsed:.2f} s  |  "
              f"dispositivo={DEVICE}  |  "
              f"RMSE val guardado={s['best_rmse_val']:.4f}")

    # ─────────────────────────────────────────────────────────────────────
    # Propiedades de acceso rápido
    # ─────────────────────────────────────────────────────────────────────
    @property
    def _s(self):  # shortcut al estado
        return self._state

    # ─────────────────────────────────────────────────────────────────────
    # Inferencia NCF
    # ─────────────────────────────────────────────────────────────────────
    def _score_ncf(self, u_idx: int, candidatos: np.ndarray,
                   feat_override: Optional[np.ndarray] = None) -> np.ndarray:
        feat = feat_override if feat_override is not None else \
               self._s["feat_por_usuario"].get(
                   u_idx, np.zeros(self._s["n_user_features"], dtype=np.float32)
               )
        with torch.no_grad():
            ut = torch.tensor([u_idx] * len(candidatos), dtype=torch.long).to(DEVICE)
            it = torch.tensor(candidatos, dtype=torch.long).to(DEVICE)
            ft = torch.tensor(
                np.tile(feat, (len(candidatos), 1)), dtype=torch.float32
            ).to(DEVICE)
            return self.model(ut, it, ft).cpu().numpy()

    # ─────────────────────────────────────────────────────────────────────
    # Helper: calcula bonus de temporada
    # ─────────────────────────────────────────────────────────────────────
    def _season_bonus(self, destination_id: int, best_time_map: dict,
                      mes_viaje: Optional[int]) -> float:
        if not mes_viaje:
            return 0.0
        best_time = best_time_map.get(destination_id, "Year-round")
        meses_validos = MESES_DESTINO.get(best_time, set(range(1, 13)))
        return 1.0 if mes_viaje in meses_validos else 0.0

    # ─────────────────────────────────────────────────────────────────────
    # Score híbrido: NCF + contenido (con opcional bonus por temporada)
    # ─────────────────────────────────────────────────────────────────────
    def _score_hybrid(self, u_idx: int, candidatos: np.ndarray,
                      tipos_preferidos: set,
                      feat_override: Optional[np.ndarray] = None,
                      alpha: float = 0.7,
                      mes_viaje: Optional[int] = None) -> np.ndarray:
        ncf_scores = self._score_ncf(u_idx, candidatos, feat_override)

        dest_ids = self._s["dest_encoder"].inverse_transform(candidatos)
        dest_df = self._s["destinations_df"].set_index("DestinationID")
        tipo_map = dest_df["Type"].to_dict()
        best_time_map = dest_df["BestTimeToVisit"].to_dict()

        content  = np.array([
            (1.0 if tipo_map.get(int(did), "") in tipos_preferidos else 0.0)
            + self._s["pop_norm"].get(int(did), 0.0)
            + (self._season_bonus(int(did), best_time_map, mes_viaje))
            for did in dest_ids
        ])

        ncf_min, ncf_max = ncf_scores.min(), ncf_scores.max()
        ncf_norm = ((ncf_scores - ncf_min) / (ncf_max - ncf_min)
                    if ncf_max > ncf_min else np.zeros_like(ncf_scores))
        content_max  = content.max()
        content_norm = content / content_max if content_max > 0 else content

        return alpha * ncf_norm + (1 - alpha) * content_norm

    # ─────────────────────────────────────────────────────────────────────
    # Recomendación para usuario existente
    # ─────────────────────────────────────────────────────────────────────
    def recommend_for_user(self, user_id: int, top_k: int = 5,
                          mes_viaje: Optional[int] = None) -> dict:
        s = self._s
        user_row = s["users_df"][s["users_df"]["UserID"] == user_id]
        if user_row.empty:
            raise ValueError(f"Usuario {user_id} no encontrado.")
        user_row = user_row.iloc[0]

        tipos = {PREF_TO_TYPE.get(p) for p in user_row["PrefList"]}

        if user_id in s["user_encoder"].classes_:
            u_idx  = int(s["user_encoder"].transform([user_id])[0])
            vistos = s["items_train_por_usuario"].get(u_idx, set())
        else:
            u_idx, vistos = 0, set()

        candidatos = np.array([i for i in range(s["n_items"]) if i not in vistos])
        scores     = self._score_hybrid(u_idx, candidatos, tipos, mes_viaje=mes_viaje)
        orden      = np.argsort(-scores)

        # Obtener más candidatos para compensar deduplicación por nombre
        fetch_k = min(len(candidatos), top_k * 3)
        top_idx    = candidatos[orden[:fetch_k]]
        top_scores = scores[orden[:fetch_k]]

        dest_ids = s["dest_encoder"].inverse_transform(top_idx)
        recommendations = self._build_recs(dest_ids, top_scores, tipos, mes_viaje=mes_viaje)
        # Deduplicar por nombre y recortar a top_k
        recommendations = self._deduplicate_recommendations(recommendations, top_k)

        return {
            "user_id":          user_id,
            "user_name":        user_row["Name"],
            "email":            user_row["Email"],
            "preferences":      user_row["Preferences"],
            "recommendations":  recommendations,
        }

    # ─────────────────────────────────────────────────────────────────────
    # Recomendación para usuario nuevo (cold-start)
    # ─────────────────────────────────────────────────────────────────────
    def recommend_new_user(self, preferences: str, gender: str,
                           n_adults: int, n_children: int, top_k: int = 5,
                           mes_viaje: Optional[int] = None) -> dict:
        s = self._s
        pref_list = [p.strip() for p in preferences.split(",") if p.strip()]
        tipos     = {PREF_TO_TYPE.get(p) for p in pref_list}

        pref_ohe   = s["mlb"].transform([pref_list])[0].astype(np.float32)
        gender_enc = 1.0 if gender.lower() in ("male", "masculino", "m") else 0.0
        feat       = np.concatenate(
            [pref_ohe, [gender_enc, float(n_adults), float(n_children)]]
        ).astype(np.float32)

        candidatos = np.arange(s["n_items"])
        scores     = self._score_hybrid(0, candidatos, tipos,
                                        feat_override=feat, alpha=0.3,
                                        mes_viaje=mes_viaje)
        orden      = np.argsort(-scores)

        # Obtener más candidatos para compensar deduplicación por nombre
        fetch_k = min(len(candidatos), top_k * 3)
        top_idx    = candidatos[orden[:fetch_k]]
        top_scores = scores[orden[:fetch_k]]

        dest_ids = s["dest_encoder"].inverse_transform(top_idx)
        recommendations = self._build_recs(dest_ids, top_scores, tipos, mes_viaje=mes_viaje)
        # Deduplicar por nombre y recortar a top_k
        recommendations = self._deduplicate_recommendations(recommendations, top_k)

        return {
            "user_id":         None,
            "user_name":       "Nuevo usuario",
            "email":           None,
            "preferences":     preferences,
            "recommendations": recommendations,
        }

    # ─────────────────────────────────────────────────────────────────────
    # Usuarios target para un destino nuevo
    # ─────────────────────────────────────────────────────────────────────
    def users_for_new_destination(self, destination_type: str, top_k: int = 10) -> list:
        s = self._s
        df = s["users_df"].copy()
        df = df.loc[:, ~df.columns.duplicated()]
        objetivo = {p for p, t in PREF_TO_TYPE.items() if t == destination_type}
        mask = df["PrefList"].apply(
            lambda L: len(set(L) & objetivo) > 0
        )
        return [
            {
                "user_id":     int(r["UserID"]),
                "name":        r["Name"],
                "email":       r["Email"],
                "preferences": r["Preferences"],
                "group_size":  int(r["NumberOfAdults"]) + int(r["NumberOfChildren"]),
            }
            for _, r in df[mask].head(top_k).iterrows()
        ]

    # ─────────────────────────────────────────────────────────────────────
    # Listados para la UI
    # ─────────────────────────────────────────────────────────────────────
    def list_users(self, page: int = 1, limit: int = 20, search: str = "") -> dict:
        df = self._s["users_df"].copy()
        df = df.loc[:, ~df.columns.duplicated()]
        if search:
            mask = (
                df["Name"].str.contains(search, case=False, na=False) |
                df["Email"].str.contains(search, case=False, na=False)
            )
            df = df[mask]
        total  = len(df)
        start  = (page - 1) * limit
        page_df = df.iloc[start: start + limit]
        return {
            "users": [
                {"user_id": int(r["UserID"]), "name": r["Name"],
                 "email": r["Email"], "preferences": r["Preferences"],
                 "gender": r["Gender"], "n_adults": int(r["NumberOfAdults"]),
                 "n_children": int(r["NumberOfChildren"])}
                for _, r in page_df.iterrows()
            ],
            "total": total, "page": page, "limit": limit,
        }

    def list_destinations(self, page: int = 1, limit: int = 20,
                          dest_type: str = "") -> dict:
        df = self._s["destinations_df"].copy()
        df = df.loc[:, ~df.columns.duplicated()]
        if dest_type:
            df = df[df["Type"].str.lower() == dest_type.lower()]
        total  = len(df)
        start  = (page - 1) * limit
        page_df = df.iloc[start: start + limit]
        return {
            "destinations": [
                {"destination_id": int(r["DestinationID"]), "name": r["Name"],
                 "state": r["State"], "type": r["Type"],
                 "popularity": round(float(r["Popularity"]), 2),
                 "best_time_to_visit": r["BestTimeToVisit"]}
                for _, r in page_df.iterrows()
            ],
            "total": total, "page": page, "limit": limit,
        }

    def destination_types(self) -> list:
        return sorted(self._s["destinations_df"]["Type"].unique().tolist())

    def preference_options(self) -> list:
        return list(PREF_TO_TYPE.keys())

    # ─────────────────────────────────────────────────────────────────────
    # Deduplicación de presentación (arregla duplicados de nombre)
    # ─────────────────────────────────────────────────────────────────────
    def _deduplicate_recommendations(self, recs_list: list, top_k: int) -> list:
        """
        Deduplica recomendaciones por nombre, conservando el representante de
        mayor score. Orden: sort desc → drop_duplicates → head top_k.
        """
        if not recs_list:
            return recs_list

        import pandas as pd
        df = pd.DataFrame(recs_list)
        # 1) Ordenar por score descendente (mayor primero)
        df = df.sort_values("score", ascending=False)
        # 2) Deduplicar por nombre, conservando el PRIMERO (= mayor score)
        df = df.drop_duplicates(subset=["name"], keep="first")
        # 3) Recortar a top_k después de deduplicar
        df = df.head(top_k)
        return df.to_dict("records")

    # ─────────────────────────────────────────────────────────────────────
    # Helper
    # ─────────────────────────────────────────────────────────────────────
    def _build_recs(self, dest_ids, scores, tipos_preferidos,
                    mes_viaje: Optional[int] = None) -> list:
        dest_map = self._s["destinations_df"].set_index("DestinationID").to_dict("index")
        recs = []
        for did, sc in zip(dest_ids, scores):
            did_int = int(did)
            rec = {
                "destination_id":     did_int,
                "name":               dest_map[did_int]["Name"],
                "state":              dest_map[did_int]["State"],
                "type":               dest_map[did_int]["Type"],
                "popularity":         round(float(dest_map[did_int]["Popularity"]), 2),
                "best_time_to_visit": dest_map[did_int]["BestTimeToVisit"],
                "score":              round(float(sc), 4),
                "match_reason":       (
                    "Coincide con tus preferencias"
                    if dest_map[did_int]["Type"] in tipos_preferidos
                    else "Popular en tu región"
                ),
            }
            if mes_viaje is not None:
                best_time = dest_map[did_int]["BestTimeToVisit"]
                meses_validos = MESES_DESTINO.get(best_time, set(range(1, 13)))
                rec["season_match"] = mes_viaje in meses_validos
            recs.append(rec)
        return recs
