"""
SimpliRoute — Route Analyzer v2
Analiza request.json + response.json del router:
- Validación pre-vuelo del request
- Detección de errores E500
- Análisis de nodos filtrados (W00001, distancia geográfica)
- Diagnóstico de nodos sin atender
- Detección de max_visit como causa de exclusión
- Recomendaciones priorizadas con mensajes claros
"""

import streamlit as st
import json
import math
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from collections import defaultdict, Counter

st.set_page_config(
    page_title="SR Route Analyzer",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #f7f8fc; }
    [data-testid="stFileUploader"] section {
        background: white !important;
        border: 1.5px dashed #d1d5db !important;
        border-radius: 10px !important;
    }
    [data-testid="stFileUploader"] section:hover { border-color: #6366f1 !important; }
    .main-header { font-size: 26px; font-weight: 700; color: #1a1d2e; margin-bottom: 0; }
    .sub-header  { font-size: 14px; color: #6b7280; margin-top: 4px; margin-bottom: 1.5rem; }
    .upload-label { font-size: 14px; font-weight: 600; color: #374151; margin-bottom: 6px; }
    .metric-card {
        background: white; border-radius: 12px; padding: 18px 20px;
        border: 1px solid #e8eaf0; margin-bottom: 0;
    }
    .metric-val { font-size: 28px; font-weight: 700; line-height: 1.1; }
    .metric-lbl { font-size: 12px; color: #6b7280; margin-top: 4px; }
    .rec-card {
        background: white; border-radius: 12px; padding: 18px 20px;
        margin-bottom: 12px; border: 1px solid #e8eaf0;
    }
    .rec-title  { font-size: 15px; font-weight: 600; color: #1a1d2e; margin-bottom: 6px; }
    .rec-detail { font-size: 13px; color: #4b5563; line-height: 1.55; }
    .rec-field  { font-size: 11px; color: #6b7280; margin-top: 8px; }
    .badge {
        display: inline-block; padding: 2px 10px; border-radius: 20px;
        font-size: 11px; font-weight: 600; margin-right: 6px;
    }
    .badge-red   { background:#fee2e2; color:#dc2626; }
    .badge-amber { background:#fef3c7; color:#b45309; }
    .badge-blue  { background:#dbeafe; color:#1d4ed8; }
    .badge-gray  { background:#f3f4f6; color:#4b5563; }
    .badge-green { background:#dcfce7; color:#16a34a; }
    .section-title { font-size: 16px; font-weight: 600; color: #1a1d2e; margin: 1.5rem 0 .75rem; }
    .error-banner {
        background: #fef2f2; border: 1.5px solid #fca5a5; border-radius: 12px;
        padding: 20px 24px; margin-bottom: 1.5rem;
    }
    .error-title { font-size: 18px; font-weight: 700; color: #dc2626; margin-bottom: 6px; }
    .preflight-card {
        background: white; border-radius: 12px; padding: 18px 20px;
        margin-bottom: 10px; border: 1px solid #e8eaf0;
    }
    .preflight-ok {
        background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 12px;
        padding: 14px 18px; margin-bottom: 10px; font-size: 13px;
        color: #15803d; font-weight: 500;
    }
    hr.divider { border: none; border-top: 1px solid #e8eaf0; margin: 1.5rem 0; }
    code { background: #f3f4f6; padding: 1px 5px; border-radius: 4px; font-size: 12px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def try_int(val):
    try:
        return int(str(val))
    except Exception:
        return None

def parse_time(t):
    if not t:
        return None
    try:
        h, m = str(t).split(":")
        return int(h) * 60 + int(m)
    except Exception:
        return None

def time_overlap(ws1, we1, ws2, we2):
    s1, e1 = parse_time(ws1), parse_time(we1)
    s2, e2 = parse_time(ws2), parse_time(we2)
    if None in (s1, e1, s2, e2):
        return True
    return s1 <= e2 and s2 <= e1

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def detect_outliers_iqr(values):
    arr = [v for v in values if v is not None and v > 0]
    if len(arr) < 4:
        return set()
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    # Multiplier 3.0 (vs standard 1.5) to avoid false positives
    # Only flag extreme outliers, not legitimate variation
    if iqr == 0:
        return set()
    upper = q3 + 3.0 * iqr
    return {v for v in arr if v > upper}

def format_time_min(minutes):
    if minutes is None:
        return "N/A"
    return f"{int(minutes // 60)}h {int(minutes % 60):02d}m"

def parse_upload(file) -> dict:
    """
    Parsea un archivo subido. Soporta:
    - JSON puro (.json o .txt)
    - Formato curl con --data-raw '...' o --data '...'
    """
    content = file.read().decode("utf-8", errors="replace").strip()

    # Intento 1: JSON directo
    try:
        return json.loads(content)
    except Exception:
        pass

    # Intento 2: curl con --data-raw o --data
    import re
    patterns = [
        r"--data-raw\s+'(.*)'$",
        r'--data-raw\s+"(.*)"$',
        r"--data\s+'(.*)'$",
        r'--data\s+"(.*)"$',
    ]
    for pattern in patterns:
        match = re.search(pattern, content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass

    # Intento 3: buscar primer { hasta el último } (JSON embebido)
    start = content.find("{")
    end   = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(content[start:end+1])
        except Exception:
            pass

    raise ValueError("No se pudo parsear el archivo. Asegúrate de que sea JSON o formato curl.")


    return "errors" in res and "vehicles" not in res

VALID_FMV = {1.0, 1.5, 2.0, 3.0}

ISSUE_LABELS = {
    "duration_anomaly":      ("🕐", "Duration outlier",         "badge-red"),
    "zone_time_overflow":    ("⏱️", "Desborde de tiempo",       "badge-red"),
    "window_shift_mismatch": ("🚫", "Ventana vs turno",         "badge-red"),
    "narrow_window":         ("📏", "Ventana estrecha",          "badge-red"),
    "zero_window":           ("⛔", "Ventana de 0 min",          "badge-red"),
    "inverted_window":       ("🔄", "Ventana invertida",         "badge-red"),
    "capacity_overflow":     ("⚖️", "Capacidad excedida",       "badge-amber"),
    "zone_mismatch":         ("🗺️", "Zona sin vehículo",        "badge-amber"),
    "skills_mismatch":       ("🔧", "Skills faltantes",         "badge-amber"),
    "max_visit_limit":       ("🔢", "Límite max_visit",          "badge-amber"),
    "tight_window":          ("⏳", "Ventana = duración",        "badge-red"),
    "exc_so_001":            ("🕐", "Excluido por ventana",      "badge-red"),
    "clustering_preference": ("✦",  "Excluido por clustering",  "badge-gray"),
    "capacity_time_general": ("⚠️", "Cap/tiempo general",       "badge-amber"),
    "zero_coordinates":      ("📍", "Coordenadas (0,0)",         "badge-red"),
    "unknown":               ("❓", "Causa desconocida",         "badge-gray"),
}


# ─────────────────────────────────────────────────────────────────────────────
# PRE-FLIGHT VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_request(req: dict) -> list:
    issues = []

    # fmv (E02005)
    fmv = req.get("fmv")
    if fmv is not None and fmv not in VALID_FMV:
        issues.append({
            "code": "E02005", "severity": "critical",
            "field": "fmv", "value": str(fmv),
            "title": f"fmv={fmv} — valor no permitido",
            "detail": (
                f"El campo fmv tiene el valor {fmv}, que no pertenece al conjunto válido "
                f"(1.0=Tráfico bajo, 1.5=Medio, 2.0=Alto, 3.0=Intenso). "
                f"Puede causar un error E500 en el router."
            ),
            "fix": "Cambiar fmv a 1.0, 1.5, 2.0 o 3.0",
            "nodes": [],
        })

    zero_window_nodes = []
    inverted_nodes    = []
    narrow_nodes      = []
    zero_coord_nodes  = []

    for node in req.get("nodes", []):
        ident = node.get("ident", "?")
        ws    = node.get("window_start", "00:00")
        we    = node.get("window_end",   "23:59")
        ws2   = node.get("window_start_2", "23:59")
        we2   = node.get("window_end_2",   "23:59")
        dur   = try_int(node.get("duration"))
        lat   = node.get("lat", None)
        lon   = node.get("lon", None)
        ws_m  = parse_time(ws)
        we_m  = parse_time(we)

        if ws == we and dur and dur > 0:
            zero_window_nodes.append({
                "ident": ident, "address": node.get("address", "")[:55],
                "window": f"{ws}–{we}", "duration": dur,
            })
        elif ws_m is not None and we_m is not None:
            if not (ws2 == "23:59" and we2 == "23:59") and ws_m > we_m:
                inverted_nodes.append({
                    "ident": ident, "address": node.get("address", "")[:55],
                    "window": f"{ws}–{we}",
                })
            elif dur and 0 < (we_m - ws_m) < dur:
                narrow_nodes.append({
                    "ident": ident, "address": node.get("address", "")[:55],
                    "window_size": we_m - ws_m, "duration": dur,
                })

        if lat == 0 and lon == 0:
            zero_coord_nodes.append(ident)

    if zero_window_nodes:
        issues.append({
            "code": "ZERO_WINDOW", "severity": "high",
            "field": "window_start / window_end",
            "value": f"{len(zero_window_nodes)} nodo(s)",
            "title": f"Ventana de 0 minutos con duration > 0 ({len(zero_window_nodes)} nodo(s))",
            "detail": (
                f"{len(zero_window_nodes)} nodo(s) tienen window_start igual a window_end "
                f"con duration > 0. El nodo no puede ser atendido en una ventana de 0 minutos."
            ),
            "fix": "Ampliar window_end o establecer duration=0",
            "nodes": zero_window_nodes,
        })

    if inverted_nodes:
        issues.append({
            "code": "E01006", "severity": "high",
            "field": "window_start / window_end",
            "value": f"{len(inverted_nodes)} nodo(s)",
            "title": f"Ventana invertida: window_start > window_end ({len(inverted_nodes)} nodo(s))",
            "detail": f"{len(inverted_nodes)} nodo(s) tienen window_start posterior a window_end (E01006).",
            "fix": "Intercambiar window_start y window_end",
            "nodes": inverted_nodes,
        })

    if narrow_nodes:
        issues.append({
            "code": "NARROW_WINDOW", "severity": "high",
            "field": "window_end / duration",
            "value": f"{len(narrow_nodes)} nodo(s)",
            "title": f"Ventana más pequeña que la duración de servicio ({len(narrow_nodes)} nodo(s))",
            "detail": f"{len(narrow_nodes)} nodo(s) tienen ventana inferior a su duración de servicio.",
            "fix": "Ampliar window_end o reducir duration",
            "nodes": narrow_nodes,
        })

    if zero_coord_nodes:
        issues.append({
            "code": "E02004", "severity": "critical",
            "field": "lat / lon",
            "value": f"{len(zero_coord_nodes)} nodo(s)",
            "title": f"Coordenadas (0, 0) detectadas ({len(zero_coord_nodes)} nodo(s))",
            "detail": f"{len(zero_coord_nodes)} nodo(s) tienen lat=0 y lon=0. El router no puede ubicarlos (E02004).",
            "fix": "Corregir las coordenadas lat/lon",
            "nodes": zero_coord_nodes,
        })

    for v in req.get("vehicles", []):
        ss = parse_time(v.get("shift_start", "00:01"))
        se = parse_time(v.get("shift_end",   "23:59"))
        if ss is not None and se is not None and ss >= se:
            issues.append({
                "code": "E01006_VEH", "severity": "high",
                "field": "shift_start / shift_end",
                "value": f"veh {v.get('ident')}",
                "title": f"Turno inválido en vehículo {v.get('ident')}",
                "detail": f"shift_start ({v.get('shift_start')}) ≥ shift_end ({v.get('shift_end')}).",
                "fix": "Corregir shift_start y shift_end del vehículo",
                "nodes": [],
            })

    return issues


# ─────────────────────────────────────────────────────────────────────────────
# FILTERED NODES ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyze_filtered_nodes(req: dict, res: dict) -> list:
    filtered  = res.get("filteredClientsNodes", [])
    node_map  = {n["ident"]: n for n in req.get("nodes", [])}
    vehicles  = req.get("vehicles", [])
    results   = []

    for fn in filtered:
        ident    = fn["ident"]
        codes    = fn.get("cause", {}).get("codes", [])
        req_node = node_map.get(ident, {})
        nlat     = fn.get("lat") or req_node.get("lat")
        nlon     = fn.get("lon") or req_node.get("lon")

        min_dist, nearest_v, shift_min = None, None, None
        for v in vehicles:
            vlat = v.get("location_start", {}).get("lat")
            vlon = v.get("location_start", {}).get("lon")
            if vlat and vlon and nlat and nlon:
                d = haversine_km(vlat, vlon, nlat, nlon)
                if min_dist is None or d < min_dist:
                    min_dist  = d
                    nearest_v = v
                    ss = parse_time(v.get("shift_start", "00:01"))
                    se = parse_time(v.get("shift_end",   "23:59"))
                    shift_min = (se - ss) if (ss and se) else None

        geo_issue, geo_detail = False, ""
        if min_dist is not None and shift_min is not None:
            travel_min = (min_dist / 60) * 60
            if travel_min * 2 > shift_min:
                geo_issue  = True
                geo_detail = (
                    f"El vehículo más cercano está a ~{min_dist:.0f} km "
                    f"({travel_min:.0f} min solo ida a 60 km/h). "
                    f"El turno disponible es de {shift_min} min — insuficiente para ir y volver."
                )

        results.append({
            "ident":      ident,
            "address":    req_node.get("address", "N/A")[:60],
            "codes":      codes,
            "load":       fn.get("load", 0),
            "load_2":     fn.get("load_2", 0),
            "load_3":     fn.get("load_3", 0),
            "dist_km":    round(min_dist, 1) if min_dist else None,
            "geo_issue":  geo_issue,
            "geo_detail": geo_detail,
            "nearest_v":  nearest_v.get("ident") if nearest_v else None,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# COMPARATIVE ANALYSIS & FLEET UTILIZATION
# ─────────────────────────────────────────────────────────────────────────────

def comparative_analysis(req: dict, res: dict) -> dict:
    """Compara nodos enrutados vs sin atender para identificar el diferenciador real."""
    node_map   = {n["ident"]: n for n in req.get("nodes", [])}
    unattended = res.get("unattendedClientsNodes", [])

    routed_ids = set()
    for v in res.get("vehicles", []):
        for tour in v.get("tours", []):
            for n in tour.get("nodes", []):
                if not n["ident"].startswith("vehicle-"):
                    routed_ids.add(n["ident"])

    un_ids = {u["ident"] for u in unattended}
    routed_nodes = [node_map[i] for i in routed_ids  if i in node_map]
    un_nodes     = [node_map[i] for i in un_ids      if i in node_map]

    if not routed_nodes or not un_nodes:
        return {}

    def avg(vals):
        return round(float(np.mean(vals)), 1) if vals else 0

    def window_min(n):
        try:
            ws = n.get("window_start", "")
            we = n.get("window_end",   "")
            h1, m1 = int(ws.split(":")[0]), int(ws.split(":")[1])
            h2, m2 = int(we.split(":")[0]), int(we.split(":")[1])
            return (h2*60+m2) - (h1*60+m1)
        except:
            return None

    # Stats per group
    def group_stats(nodes):
        loads    = [n.get("load", 0) for n in nodes]
        durs     = [try_int(n.get("duration")) for n in nodes]
        durs     = [d for d in durs if d is not None]
        windows  = [w for n in nodes if (w := window_min(n)) is not None]
        return {
            "load_mean":    avg(loads),
            "load_median":  round(float(np.median(loads)), 1) if loads else 0,
            "dur_mean":     avg(durs),
            "window_mean":  avg(windows),
            "count":        len(nodes),
        }

    r_stats = group_stats(routed_nodes)
    u_stats = group_stats(un_nodes)

    # Detect differentiators
    differentiators = []

    # Load differentiator
    if r_stats["load_mean"] > 0 and u_stats["load_mean"] > 0:
        ratio = r_stats["load_mean"] / max(u_stats["load_mean"], 0.01)
        if ratio > 2.0:
            differentiators.append({
                "field":      "load (carga)",
                "routed":     f"{r_stats['load_mean']:.0f}",
                "unattended": f"{u_stats['load_mean']:.0f}",
                "diff":       f"{ratio:.1f}×",
                "severity":   "high",
                "conclusion": (
                    f"Los nodos enrutados tienen {ratio:.1f}× más carga que los sin atender "
                    f"({r_stats['load_mean']:.0f} vs {u_stats['load_mean']:.0f}). "
                    f"El router priorizó los de mayor carga — los de menor carga quedaron sin vehículo disponible."
                ),
                "fix": "Verificar si hay suficientes vehículos para cubrir todos los nodos, o revisar la lógica de priorización de carga.",
            })
        elif ratio < 0.5:
            differentiators.append({
                "field":      "load (carga)",
                "routed":     f"{r_stats['load_mean']:.0f}",
                "unattended": f"{u_stats['load_mean']:.0f}",
                "diff":       f"{1/ratio:.1f}× mayor en sin atender",
                "severity":   "high",
                "conclusion": (
                    f"Los nodos sin atender tienen {1/ratio:.1f}× más carga que los enrutados. "
                    f"La capacidad de los vehículos puede ser insuficiente para los nodos más pesados."
                ),
                "fix": "Revisar si la carga de los nodos sin atender excede la capacidad de los vehículos disponibles.",
            })

    # Window differentiator
    if r_stats["window_mean"] > 0 and u_stats["window_mean"] > 0:
        w_ratio = r_stats["window_mean"] / max(u_stats["window_mean"], 0.01)
        if w_ratio > 1.5:
            differentiators.append({
                "field":      "ventana de tiempo",
                "routed":     f"{r_stats['window_mean']:.0f} min",
                "unattended": f"{u_stats['window_mean']:.0f} min",
                "diff":       f"{w_ratio:.1f}×",
                "severity":   "high",
                "conclusion": (
                    f"Los nodos enrutados tienen ventanas {w_ratio:.1f}× más amplias "
                    f"({r_stats['window_mean']:.0f} vs {u_stats['window_mean']:.0f} min). "
                    f"Las ventanas estrechas de los nodos sin atender no permiten encadenar visitas."
                ),
                "fix": "Ampliar las ventanas de tiempo de los nodos sin atender para permitir encadenamiento de visitas.",
            })
        elif w_ratio < 0.67:
            differentiators.append({
                "field":      "ventana de tiempo",
                "routed":     f"{r_stats['window_mean']:.0f} min",
                "unattended": f"{u_stats['window_mean']:.0f} min",
                "diff":       f"{1/w_ratio:.1f}× mayor en sin atender",
                "severity":   "medium",
                "conclusion": (
                    f"Los nodos sin atender tienen ventanas más amplias pero aun así no fueron cubiertos. "
                    f"La ventana no es el diferenciador principal."
                ),
                "fix": "Revisar otros factores como geografía o disponibilidad de vehículos.",
            })

    # Duration differentiator
    if r_stats["dur_mean"] > 0 and u_stats["dur_mean"] > 0:
        d_ratio = r_stats["dur_mean"] / max(u_stats["dur_mean"], 0.01)
        if d_ratio > 2.0 or d_ratio < 0.5:
            differentiators.append({
                "field":      "duration (servicio)",
                "routed":     f"{r_stats['dur_mean']:.0f} min",
                "unattended": f"{u_stats['dur_mean']:.0f} min",
                "diff":       f"{max(d_ratio, 1/d_ratio):.1f}×",
                "severity":   "medium",
                "conclusion": (
                    f"Diferencia significativa en tiempos de servicio entre grupos "
                    f"({r_stats['dur_mean']:.0f} vs {u_stats['dur_mean']:.0f} min)."
                ),
                "fix": "Revisar si la duración de servicio de los nodos sin atender es correcta.",
            })

    return {
        "routed_stats":   r_stats,
        "unatt_stats":    u_stats,
        "differentiators": differentiators,
    }


def fleet_utilization(req: dict, res: dict) -> dict:
    """Analiza la utilización real de cada vehículo: visitas, carga y % de capacidad."""
    vehicles   = {v["ident"]: v for v in req.get("vehicles", [])}
    node_map   = {n["ident"]: n for n in req.get("nodes",    [])}

    # Build load per vehicle from response
    veh_data = {}
    for v_res in res.get("vehicles", []):
        vid   = v_res["ident"]
        v_req = vehicles.get(vid, {})
        visits = 0
        load1  = 0.0
        for tour in v_res.get("tours", []):
            for n in tour.get("nodes", []):
                if not n["ident"].startswith("vehicle-"):
                    visits += 1
                    req_n   = node_map.get(n["ident"], {})
                    load1  += req_n.get("load", 0) or 0

        cap1 = v_req.get("capacity", 0) or 0
        pct  = round(load1 / cap1 * 100, 1) if cap1 > 0 and cap1 < 1e15 else None

        veh_data[vid] = {
            "visits":   visits,
            "load1":    round(load1, 1),
            "cap1":     cap1,
            "cap_pct":  pct,
            "at_cap":   pct is not None and pct >= 90,
        }

    active  = {k: v for k, v in veh_data.items() if v["visits"] > 0}
    idle    = {k: v for k, v in veh_data.items() if v["visits"] == 0}
    at_cap  = {k: v for k, v in active.items()   if v["at_cap"]}

    visits_list = [v["visits"] for v in active.values()]
    avg_visits  = round(float(np.mean(visits_list)), 1) if visits_list else 0

    # Flag: each vehicle only does 1 visit
    single_visit_ratio = sum(1 for v in visits_list if v == 1) / max(len(visits_list), 1)

    return {
        "total":              len(veh_data),
        "active":             len(active),
        "idle":               len(idle),
        "at_cap":             len(at_cap),
        "avg_visits":         avg_visits,
        "max_visits":         max(visits_list) if visits_list else 0,
        "min_visits":         min(visits_list) if visits_list else 0,
        "single_visit_ratio": round(single_visit_ratio * 100, 1),
        "veh_data":           veh_data,
        "alert_single":       single_visit_ratio > 0.8,
        "alert_idle":         len(idle) > len(veh_data) * 0.2,
        "alert_at_cap":       len(at_cap) > len(active) * 0.5,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def analyze(req: dict, res: dict) -> dict:
    findings = {
        "summary": {}, "unattended": [], "zone_stats": {},
        "recommendations": [], "raw_issue_counts": Counter(),
    }

    node_map   = {n["ident"]: n for n in req.get("nodes", [])}
    vehicles   = {v["ident"]: v for v in req.get("vehicles", [])}
    unattended = res.get("unattendedClientsNodes", [])
    filtered   = res.get("filteredClientsNodes", [])

    # ── Nodos enrutados y stats de max_visit ──────────────────────────────────
    all_routed_ids     = set()
    routed_per_vehicle = {}
    for v in res.get("vehicles", []):
        vid   = v["ident"]
        count = 0
        for tour in v.get("tours", []):
            for n in tour.get("nodes", []):
                if not n["ident"].startswith("vehicle-"):
                    all_routed_ids.add(n["ident"])
                    count += 1
        routed_per_vehicle[vid] = count

    max_visit_global = None
    vehicles_at_max  = []
    vehicles_idle    = []
    for vid, v in vehicles.items():
        mv = v.get("max_visit")
        if mv is not None:
            max_visit_global = mv
            if routed_per_vehicle.get(vid, 0) >= mv:
                vehicles_at_max.append(vid)
        if routed_per_vehicle.get(vid, 0) == 0:
            vehicles_idle.append(vid)

    # ── Zonas ─────────────────────────────────────────────────────────────────
    zone_to_vehicles = defaultdict(list)
    for vid, v in vehicles.items():
        for z in v.get("zones", []):
            zone_to_vehicles[z].append(vid)
    vehicles_no_zone = [vid for vid, v in vehicles.items() if not v.get("zones")]

    zone_nodes = defaultdict(list)
    for n in req.get("nodes", []):
        zones = n.get("zones", [])
        if zones:
            for z in zones:
                zone_nodes[z].append(n)
        else:
            zone_nodes["sin_zona"].append(n)

    for zone, nodes in zone_nodes.items():
        durations_valid = [d for d in [try_int(n.get("duration")) for n in nodes] if d is not None]
        outlier_vals    = detect_outliers_iqr(durations_valid)
        total_dur       = sum(durations_valid)
        outlier_count   = sum(1 for d in durations_valid if d in outlier_vals)
        normal_durs     = [d for d in durations_valid if d not in outlier_vals]
        median_normal   = float(np.median(normal_durs)) if normal_durs else 0
        corrected_dur   = sum(normal_durs) + outlier_count * median_normal

        veh_ids         = zone_to_vehicles.get(zone, vehicles_no_zone if zone == "sin_zona" else [])
        avail_min_veh   = 0
        avail_min_total = 0
        shift_info      = ("00:01", "23:59")
        if veh_ids:
            v  = vehicles.get(veh_ids[0], {})
            ss = v.get("shift_start", "00:01")
            se = v.get("shift_end",   "23:59")
            ps, pe = parse_time(ss), parse_time(se)
            if ps is not None and pe is not None:
                avail_min_veh   = pe - ps
                avail_min_total = avail_min_veh * len(veh_ids)
                shift_info      = (ss, se)

        findings["zone_stats"][zone] = {
            "total_nodes":    len(nodes),
            "total_dur":      total_dur,
            "avail_min":      avail_min_total,
            "avail_min_veh":  avail_min_veh,
            "num_vehicles":   len(veh_ids),
            "overflow":       total_dur > avail_min_total if avail_min_total > 0 else False,
            "outlier_vals":   outlier_vals,
            "outlier_count":  outlier_count,
            "vehicles":       veh_ids,
            "shift_info":     shift_info,
            "mean_dur":       np.mean(durations_valid) if durations_valid else 0,
            "corrected_dur":  corrected_dur,
        }

    # ── Análisis por nodo sin atender ─────────────────────────────────────────
    issue_counts  = Counter()
    cause_details = []

    for un in unattended:
        ident      = un["ident"]
        cause_code = un.get("cause", {}).get("code", "")
        cause_msg  = un.get("cause", {}).get("details", "")
        req_node   = node_map.get(ident, {})

        node_zones  = req_node.get("zones", [])
        node_dur    = try_int(req_node.get("duration"))
        node_ws     = req_node.get("window_start", "00:00")
        node_we     = req_node.get("window_end",   "23:59")
        node_skills = req_node.get("skills_required", [])
        node_group  = req_node.get("group", [ident])
        issues      = []

        cand_vehicles = []
        for z in node_zones:
            cand_vehicles.extend(zone_to_vehicles.get(z, []))
        if not cand_vehicles:
            cand_vehicles = vehicles_no_zone[:]

        ws_m = parse_time(node_ws)
        we_m = parse_time(node_we)

        # 1. Ventana de 0 minutos
        if node_ws == node_we and node_dur and node_dur > 0:
            issues.append({
                "type": "zero_window", "severity": "high",
                "field": "window_start / window_end",
                "value": f"{node_ws} == {node_we}, duration={node_dur}",
                "detail": f"Ventana de 0 minutos con duration={node_dur} min. El nodo no puede ser atendido.",
                "fix":    "Ampliar window_end o establecer duration=0",
            })

        # 2. Ventana estrecha (ventana < duración)
        elif ws_m is not None and we_m is not None and node_dur is not None:
            window_size = we_m - ws_m
            if 0 < window_size < node_dur:
                issues.append({
                    "type": "narrow_window", "severity": "high",
                    "field": "window_end",
                    "value": f"ventana={window_size} min < duration={node_dur} min",
                    "detail": f"Ventana de {window_size} min es menor que la duración de servicio ({node_dur} min).",
                    "fix":    "Ampliar window_end o reducir duration",
                })
            # 2b. Ventana igual a duración (sin tiempo para traslado)
            elif window_size == node_dur and node_dur > 0:
                issues.append({
                    "type": "tight_window", "severity": "high",
                    "field": "window_end / duration",
                    "value": f"ventana={window_size} min == duration={node_dur} min",
                    "detail": (
                        f"La ventana de tiempo ({window_size} min) es exactamente igual a la duración "
                        f"de servicio ({node_dur} min), dejando 0 minutos para el traslado. "
                        f"El router no puede llegar, atender y salir dentro del mismo intervalo."
                    ),
                    "fix": "Ampliar window_end al menos 15–30 min más allá de la duración de servicio",
                })

        # 3. Duration outlier
        for z in (node_zones if node_zones else ["sin_zona"]):
            zs = findings["zone_stats"].get(z, {})
            if node_dur is not None and node_dur in zs.get("outlier_vals", set()):
                issues.append({
                    "type": "duration_anomaly", "severity": "high",
                    "field": "duration", "value": node_dur,
                    "detail": (
                        f"duration={node_dur} min es un outlier estadístico en zona {z}. "
                        f"Media del resto: ~{zs.get('mean_dur', 0):.0f} min."
                    ),
                    "fix": "Reemplazar duration por el tiempo real de servicio",
                })
                break

        # 4. Desborde de tiempo en zona
        for z in (node_zones if node_zones else ["sin_zona"]):
            zs = findings["zone_stats"].get(z, {})
            if zs.get("overflow"):
                issues.append({
                    "type": "zone_time_overflow", "severity": "high",
                    "field": "duration (zona)",
                    "value": f"{zs['total_dur']} / {zs['avail_min']} min",
                    "detail": (
                        f"Zona {z}: servicio total {format_time_min(zs['total_dur'])} "
                        f"excede ventana total {format_time_min(zs['avail_min'])} "
                        f"({zs['num_vehicles']} vehículo(s) × {format_time_min(zs['avail_min_veh'])})."
                    ),
                    "fix": "Corregir durations outlier y/o agregar vehículo a la zona",
                })
                break

        # 5. Ventana vs turno
        if cand_vehicles:
            any_overlap = any(
                time_overlap(node_ws, node_we,
                             vehicles[vid].get("shift_start", "00:01"),
                             vehicles[vid].get("shift_end",   "23:59"))
                for vid in cand_vehicles if vid in vehicles
            )
            if not any_overlap:
                issues.append({
                    "type": "window_shift_mismatch", "severity": "high",
                    "field": "window_start / window_end",
                    "value": f"{node_ws}–{node_we}",
                    "detail": "La ventana del nodo no se solapa con el turno de ningún vehículo candidato.",
                    "fix":    "Ajustar window_start/window_end o el shift del vehículo",
                })

        # 6. Capacidad
        for vid in cand_vehicles:
            v = vehicles.get(vid, {})
            c1 = v.get("capacity",   0) or 0
            c2 = v.get("capacity_2", 0) or 0
            c3 = v.get("capacity_3", 0) or 0
            l1, l2, l3 = un.get("load", 0), un.get("load_2", 0), un.get("load_3", 0)
            if l1 > c1 or l2 > c2 or (c3 > 0 and l3 > c3):
                issues.append({
                    "type": "capacity_overflow", "severity": "high",
                    "field": "load",
                    "value": f"{l1}/{c1} | {l2}/{c2} | {l3}/{c3}",
                    "detail": f"La carga del nodo excede la capacidad del vehículo {vid}.",
                    "fix":    "Revisar la carga del nodo o la capacidad del vehículo",
                })
                break

        # 7. Zona sin vehículo
        if node_zones and not any(z in zone_to_vehicles for z in node_zones):
            issues.append({
                "type": "zone_mismatch", "severity": "medium",
                "field": "zones", "value": str(node_zones),
                "detail": f"Ningún vehículo tiene asignada la zona {node_zones}.",
                "fix":    "Asignar la zona a un vehículo disponible",
            })

        # 8. Skills
        if node_skills:
            all_skills = set()
            for v in vehicles.values():
                all_skills.update(v.get("skills", []))
            missing = [s for s in node_skills if s not in all_skills]
            if missing:
                issues.append({
                    "type": "skills_mismatch", "severity": "medium",
                    "field": "skills_required", "value": str(missing),
                    "detail": f"Ningún vehículo tiene las skills requeridas: {missing}.",
                    "fix":    "Agregar las skills al vehículo correspondiente",
                })

        # 9. max_visit — vehículos llenos (sin otra causa detectada)
        if not issues and max_visit_global and vehicles_at_max:
            issues.append({
                "type": "max_visit_limit", "severity": "medium",
                "field": "max_visit",
                "value": f"max_visit={max_visit_global}",
                "detail": (
                    f"{len(vehicles_at_max)} de {len(vehicles)} vehículos llegaron al límite "
                    f"de {max_visit_global} visitas y no pueden recibir más nodos. "
                    f"Hay {len(vehicles_idle)} vehículo(s) sin ninguna visita asignada."
                ),
                "fix": f"Aumentar max_visit por encima de {max_visit_global} o reasignar vehículos inactivos",
            })

        # Fallback
        if not issues:
            if cause_code == "EXC_SO-002":
                issues.append({
                    "type": "clustering_preference", "severity": "low",
                    "field": "beauty", "value": cause_code,
                    "detail": (
                        "El optimizador excluyó este nodo para producir rutas más agrupadas "
                        "y con menos cruces. Tiene tiempo y capacidad disponibles."
                    ),
                    "fix": "Probar con beauty=false para priorizar atender todos los nodos",
                })
            elif cause_code == "EXC_SO-001":
                issues.append({
                    "type": "exc_so_001", "severity": "high",
                    "field": "window_start / window_end",
                    "value": cause_code,
                    "detail": (
                        "El nodo fue excluido por restricción de ventana de tiempo antes de la "
                        "optimización. Ningún vehículo puede alcanzarlo dentro de su ventana horaria."
                    ),
                    "fix": "Ampliar la ventana de tiempo o revisar el turno de los vehículos candidatos",
                })
            else:
                issues.append({
                    "type": "capacity_time_general", "severity": "medium",
                    "field": "tiempo / capacidad", "value": cause_code,
                    "detail": (
                        "No se detectó una causa específica. El nodo quedó fuera por "
                        "falta de tiempo o capacidad en los vehículos cercanos."
                    ),
                    "fix": "Revisar carga, duración y ventanas del nodo",
                })

        for iss in issues:
            issue_counts[iss["type"]] += 1

        cause_details.append({
            "ident":        ident,
            "cause_code":   cause_code,
            "address":      req_node.get("address", "N/A")[:60],
            "zones":        node_zones,
            "duration":     node_dur,
            "window":       f"{node_ws}–{node_we}",
            "load":         un.get("load",   0),
            "load_2":       un.get("load_2", 0),
            "load_3":       un.get("load_3", 0),
            "issues":       issues,
            "primary_type": issues[0]["type"] if issues else "unknown",
        })

    findings["unattended"]       = cause_details
    findings["raw_issue_counts"] = issue_counts

    # ── Recomendaciones ───────────────────────────────────────────────────────
    recs = []

    # max_visit — primera prioridad si aplica
    if max_visit_global and vehicles_at_max:
        recs.append({
            "priority": 1, "color": "#ef4444",
            "title": f"Aumentar o eliminar el límite max_visit={max_visit_global}",
            "detail": (
                f"{len(vehicles_at_max)} vehículo(s) llegaron al tope de {max_visit_global} visitas "
                f"y no pueden recibir más nodos aunque tengan tiempo y capacidad disponibles. "
                f"Hay {len(vehicles_idle)} vehículo(s) sin ninguna visita asignada que podrían "
                f"absorber los nodos pendientes."
            ),
            "field": "max_visit",
            "affected": issue_counts.get("max_visit_limit", len(unattended)),
        })

    # Desborde de tiempo por zona
    for z, zs in findings["zone_stats"].items():
        if zs.get("overflow") and zs.get("outlier_count", 0) > 0:
            recs.append({
                "priority": 1, "color": "#ef4444",
                "title": f"Corregir duration outlier en zona {z}",
                "detail": (
                    f"{zs['outlier_count']} nodo(s) tienen duration={zs['outlier_vals']} — "
                    f"valor atípico estadísticamente. Corrigiéndolos el tiempo total bajaría "
                    f"de {format_time_min(zs['total_dur'])} a ~{format_time_min(zs['corrected_dur'])}, "
                    f"dentro de la ventana total disponible de {format_time_min(zs['avail_min'])} "
                    f"({zs['num_vehicles']} vehículo(s) × {format_time_min(zs['avail_min_veh'])})."
                ),
                "field": "duration", "affected": zs["outlier_count"],
            })
        elif zs.get("overflow"):
            recs.append({
                "priority": 1, "color": "#ef4444",
                "title": f"Agregar vehículo o ampliar turno en zona {z}",
                "detail": (
                    f"Zona {z}: la suma de tiempos de servicio ({format_time_min(zs['total_dur'])}) "
                    f"excede el tiempo total disponible entre los {zs['num_vehicles']} vehículo(s) "
                    f"({format_time_min(zs['avail_min'])}). "
                    f"No se detectaron outliers de duration — se necesita más capacidad temporal."
                ),
                "field": "shift_end o nuevo vehículo", "affected": zs["total_nodes"],
            })

    issue_recs = [
        ("tight_window",
         "Ampliar ventanas iguales a la duración de servicio",
         "La ventana de tiempo es exactamente igual a la duración de servicio, "
         "dejando 0 minutos para el traslado. El router no puede completar la visita "
         "dentro del intervalo. Ampliar la ventana al menos 15–30 min adicionales.",
         1, "#ef4444", "window_end"),

        ("exc_so_001",
         "Revisar ventanas de nodos excluidos antes de optimizar (EXC_SO-001)",
         "Estos nodos fueron descartados por el router antes de la optimización "
         "por incompatibilidad de ventana horaria con los vehículos disponibles.",
         2, "#f59e0b", "window_start / window_end"),

        ("zero_window",
         "Corregir nodos con ventana de 0 minutos",
         "Estos nodos tienen window_start igual a window_end con duration > 0. "
         "Es imposible atenderlos en una ventana de 0 minutos.",
         1, "#ef4444", "window_start / window_end"),

        ("window_shift_mismatch",
         "Ajustar ventanas incompatibles con el turno del vehículo",
         "La ventana de tiempo del nodo no se solapa con el horario de ningún vehículo candidato. "
         "El router los descarta aunque haya capacidad disponible.",
         2, "#f59e0b", "window_start / window_end"),

        ("narrow_window",
         "Ampliar ventanas más pequeñas que la duración de servicio",
         "La ventana disponible es menor que el tiempo necesario para completar la visita. "
         "Es físicamente imposible atenderlos.",
         2, "#f59e0b", "window_end"),

        ("inverted_window",
         "Corregir ventanas horarias invertidas (E01006)",
         "window_start es posterior a window_end. El router rechaza estas ventanas.",
         2, "#f59e0b", "window_start / window_end"),

        ("zone_mismatch",
         "Asignar vehículos a las zonas sin cobertura",
         "Estos nodos tienen una zona asignada pero ningún vehículo cubre esa zona. "
         "El router los ignora completamente.",
         2, "#f59e0b", "zones (vehículo)"),

        ("capacity_overflow",
         "Revisar nodos cuya carga excede la capacidad de todos los vehículos",
         "La carga individual supera la capacidad de cualquier vehículo disponible. "
         "Ningún vehículo puede atenderlos independientemente de la ruta.",
         2, "#f59e0b", "load / capacity"),

        ("skills_mismatch",
         "Agregar skills faltantes a vehículos",
         "Estos nodos requieren skills que ningún vehículo tiene asignadas. "
         "El router no puede asignarlos a ningún vehículo.",
         3, "#3b82f6", "skills (vehículo)"),

        ("clustering_preference",
         "Evaluar desactivar el parámetro beauty",
         "El optimizador los excluyó para minimizar cruces y producir rutas más agrupadas. "
         "Tienen tiempo y capacidad disponibles — con beauty=false serían incluidos.",
         4, "#6b7280", "beauty"),

        ("capacity_time_general",
         "Revisar configuración de nodos sin causa específica detectada",
         "No se identificó un problema concreto. Puede ser consecuencia de una combinación "
         "de restricciones globales. Revisar carga, duración y ventanas.",
         4, "#6b7280", "múltiples campos"),
    ]

    for issue_type, title, detail, priority, color, field in issue_recs:
        cnt = issue_counts.get(issue_type, 0)
        if cnt > 0:
            recs.append({
                "priority": priority, "color": color,
                "title":    title,
                "detail":   f"{cnt} nodo(s) afectado(s). {detail}",
                "field":    field, "affected": cnt,
            })

    recs.sort(key=lambda r: r["priority"])
    findings["recommendations"] = recs

    findings["summary"] = {
        "total_nodes":       len(req.get("nodes", [])),
        "total_vehicles":    len(req.get("vehicles", [])),
        "routed_nodes":      len(all_routed_ids),
        "unattended_count":  len(unattended),
        "filtered_count":    len(filtered),
        "vehicles_used":     res.get("num_vehicles_used", 0),
        "so002": sum(1 for u in unattended if u.get("cause", {}).get("code") == "EXC_SO-002"),
        "so003": sum(1 for u in unattended if u.get("cause", {}).get("code") == "EXC_SO-003"),
        "attendance_rate":   round(len(all_routed_ids) / max(len(req.get("nodes", [])), 1) * 100, 1),
        "max_visit_global":  max_visit_global,
        "vehicles_at_max":   len(vehicles_at_max),
        "vehicles_idle":     len(vehicles_idle),
    }
    return findings


# ─────────────────────────────────────────────────────────────────────────────
# UI COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

def render_metric(label, value, color="#1a1d2e", suffix=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val" style="color:{color}">{value}{suffix}</div>
        <div class="metric-lbl">{label}</div>
    </div>""", unsafe_allow_html=True)

def render_recommendation(rec, idx):
    st.markdown(f"""
    <div class="rec-card" style="border-left: 4px solid {rec['color']};">
        <div style="display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:6px;">
            <div class="rec-title">{idx}. {rec['title']}</div>
            <span class="badge badge-gray">{rec['affected']} afectado(s)</span>
        </div>
        <div class="rec-detail">{rec['detail']}</div>
        <div class="rec-field">Campo: <code>{rec['field']}</code></div>
    </div>""", unsafe_allow_html=True)

def render_preflight_issue(issue):
    sev_color = {"critical": "#ef4444", "high": "#f59e0b", "medium": "#3b82f6"}.get(issue["severity"], "#6b7280")
    sev_label = {"critical": "🔴 Crítico", "high": "🟡 Alto", "medium": "🔵 Medio"}.get(issue["severity"], "⚪")

    nodes_html = ""
    node_list  = issue.get("nodes", [])
    if node_list and len(node_list) <= 15:
        rows = ""
        for n in node_list:
            if isinstance(n, dict):
                extras = []
                if "window"      in n: extras.append(f"ventana: {n['window']}")
                if "duration"    in n: extras.append(f"duration: {n['duration']} min")
                if "window_size" in n: extras.append(f"ventana disponible: {n['window_size']} min")
                extra_str = " | ".join(extras)
                rows += (
                    f"<tr>"
                    f"<td style='padding:3px 8px;font-family:monospace;font-size:11px;color:#374151'>{n.get('ident','')}</td>"
                    f"<td style='padding:3px 8px;font-size:11px;color:#6b7280'>{n.get('address','')}</td>"
                    f"<td style='padding:3px 8px;font-size:11px;color:#9ca3af'>{extra_str}</td>"
                    f"</tr>"
                )
            else:
                rows += f"<tr><td style='padding:3px 8px;font-family:monospace;font-size:11px;color:#374151'>{n}</td></tr>"
        if rows:
            nodes_html = (
                f"<table style='margin-top:8px;width:100%;border-collapse:collapse;"
                f"background:#f9fafb;border-radius:6px;overflow:hidden'>{rows}</table>"
            )

    st.markdown(f"""
    <div class="preflight-card" style="border-left: 4px solid {sev_color};">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
            <span class="badge" style="background:{sev_color}20;color:{sev_color}">{sev_label}</span>
            <span class="badge badge-gray"><code style="background:transparent">{issue['code']}</code></span>
            <strong style="font-size:14px;color:#1a1d2e">{issue['title']}</strong>
        </div>
        <div style="font-size:13px;color:#4b5563;line-height:1.55">{issue['detail']}</div>
        <div style="font-size:11px;color:#6b7280;margin-top:6px;">
            Campo: <code>{issue['field']}</code> = <code>{issue['value']}</code>
            &nbsp;|&nbsp; ✏️ {issue['fix']}
        </div>
        {nodes_html}
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:0.25rem;">
        <span style="font-size:32px;">🗺️</span>
        <div>
            <div class="main-header">SimpliRoute — Route Analyzer</div>
            <div class="sub-header">Validación pre-vuelo · Diagnóstico de nodos sin atender · Recomendaciones priorizadas</div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="upload-label">📥 Cargar Request (JSON o cURL)</div>', unsafe_allow_html=True)
        req_file = st.file_uploader("request", type=["json", "txt"], key="req", label_visibility="collapsed")
    with col2:
        st.markdown('<div class="upload-label">📤 Cargar Response (JSON o cURL)</div>', unsafe_allow_html=True)
        res_file = st.file_uploader("response", type=["json", "txt"], key="res", label_visibility="collapsed")

    if not req_file or not res_file:
        st.info("Carga ambos archivos para iniciar el análisis.")
        return

    try:
        req = parse_upload(req_file)
        res = parse_upload(res_file)
    except Exception as e:
        st.error(f"Error al parsear los archivos: {e}")
        return

    # ── PRE-FLIGHT ────────────────────────────────────────────────────────────
    preflight_issues = validate_request(req)
    st.markdown('<div class="section-title">🔍 Validación pre-vuelo del Request</div>', unsafe_allow_html=True)
    if not preflight_issues:
        st.markdown('<div class="preflight-ok">✅ Sin problemas detectados en el request.</div>', unsafe_allow_html=True)
    else:
        critical_count = sum(1 for i in preflight_issues if i["severity"] == "critical")
        if critical_count:
            st.markdown(
                f'<div style="font-size:13px;color:#dc2626;font-weight:600;margin-bottom:8px;">'
                f'⛔ {critical_count} problema(s) crítico(s) — pueden causar error E500</div>',
                unsafe_allow_html=True,
            )
        for issue in preflight_issues:
            render_preflight_issue(issue)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── RESPONSE DE ERROR ─────────────────────────────────────────────────────
    if is_error_response(res):
        for err in res.get("errors", []):
            st.markdown(f"""
            <div class="error-banner">
                <div class="error-title">⛔ Error del Router — {err.get('code','')}</div>
                <div style="font-size:14px;color:#7f1d1d;margin-top:4px;">{err.get('message','')}</div>
                <div style="font-size:12px;color:#991b1b;margin-top:6px;">metadata: {err.get('metadata',{})}</div>
            </div>""", unsafe_allow_html=True)
        st.warning(
            "El router no produjo rutas. Revisa los problemas detectados en la "
            "validación pre-vuelo — uno o más de ellos es la causa probable del error."
        )
        return

    # ── NODOS ─────────────────────────────────────────────────────────────────
    unattended = res.get("unattendedClientsNodes", [])
    filtered   = res.get("filteredClientsNodes", [])
    veh_used   = res.get("num_vehicles_used", 0)

    if veh_used == 0 and (unattended or filtered):
        st.markdown("""
        <div class="error-banner">
            <div class="error-title">⚠️ El router no generó ninguna ruta (num_vehicles_used = 0)</div>
            <div style="font-size:13px;color:#7f1d1d;margin-top:4px;">
                Todos los nodos fueron descartados antes o durante la optimización.
            </div>
        </div>""", unsafe_allow_html=True)

    # ── filteredClientsNodes ───────────────────────────────────────────────────
    if filtered:
        st.markdown(
            f'<div class="section-title">🚫 Nodos filtrados antes de optimizar ({len(filtered)})</div>',
            unsafe_allow_html=True,
        )
        st.caption("Descartados por el router ANTES de generar rutas. No aparecen en unattendedClientsNodes.")

        fa = analyze_filtered_nodes(req, res)

        geo_issues  = [fn for fn in fa if fn["geo_issue"]]
        codes_all   = [c for fn in fa for c in fn["codes"]]
        code_counts = Counter(codes_all)
        dists       = [fn["dist_km"] for fn in fa if fn["dist_km"]]
        avg_dist    = round(sum(dists) / len(dists), 1) if dists else None

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1: render_metric("Total filtrados",      len(filtered),       "#dc2626")
        with mc2: render_metric("Con problema geo",     len(geo_issues),     "#f59e0b" if geo_issues else "#16a34a")
        with mc3: render_metric("Dist. promedio (km)",  avg_dist or "—")
        with mc4: render_metric("Código más frecuente", code_counts.most_common(1)[0][0] if code_counts else "—")

        if len(geo_issues) == len(fa):
            st.error(
                "🗺️ **Todos los nodos tienen incompatibilidad geográfica.** "
                "El vehículo parte de una ubicación demasiado lejana. "
                "Verificar que el punto de inicio corresponda a la misma ciudad/región que las visitas."
            )
        elif geo_issues:
            st.warning(f"🗺️ {len(geo_issues)} de {len(fa)} nodos tienen incompatibilidad geográfica.")

        rows = []
        for fn in fa:
            rows.append({
                "Ident":          fn["ident"],
                "Dirección":      fn["address"],
                "Código":         ", ".join(fn["codes"]),
                "Vehículo ref.":  fn["nearest_v"] or "—",
                "Dist. veh (km)": fn["dist_km"],
                "Problema geo":   "Sí" if fn["geo_issue"] else "No",
                "Diagnóstico":    fn["geo_detail"] if fn["geo_detail"] else (
                    "W00001: revisar coordenadas o configuración del país"
                    if "W00001" in fn["codes"] else "—"
                ),
                "Load 1": fn["load"], "Load 2": fn["load_2"], "Load 3": fn["load_3"],
            })

        df_filtered = pd.DataFrame(rows)
        st.dataframe(df_filtered, use_container_width=True, hide_index=True)

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df_filtered.to_excel(writer, index=False, sheet_name="Nodos Filtrados")
        buffer.seek(0)
        st.download_button(
            label="📥 Descargar detalle en Excel",
            data=buffer,
            file_name="nodos_filtrados.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    if not unattended and not filtered:
        st.success("✅ No hay nodos sin atender ni filtrados en este response.")
        return

    if not unattended:
        return

    # ── ANÁLISIS NODOS SIN ATENDER ────────────────────────────────────────────
    with st.spinner("Analizando..."):
        findings = analyze(req, res)

    s = findings["summary"]
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: render_metric("Total nodos",     s["total_nodes"])
    with c2: render_metric("Enrutados",        s["routed_nodes"],     "#16a34a")
    with c3: render_metric("Sin atender",      s["unattended_count"], "#dc2626")
    with c4: render_metric("Tasa atención",    s["attendance_rate"],  "#1d4ed8", "%")
    with c5: render_metric("Vehículos usados", f"{s['vehicles_used']}/{s['total_vehicles']}")
    with c6: render_metric("SO-002 / SO-003",  f"{s['so002']} / {s['so003']}")

    # Alerta max_visit
    if s.get("max_visit_global") and s.get("vehicles_at_max", 0) > 0:
        st.warning(
            f"🔢 **max_visit={s['max_visit_global']}** — {s['vehicles_at_max']} vehículo(s) llegaron al tope. "
            f"{s['vehicles_idle']} vehículo(s) sin visitas asignadas."
        )

    st.markdown('<div class="section-title">🎯 Recomendaciones priorizadas</div>', unsafe_allow_html=True)
    if not findings["recommendations"]:
        st.info("No se generaron recomendaciones.")
    else:
        for idx, rec in enumerate(findings["recommendations"], 1):
            render_recommendation(rec, idx)

    # ── ANÁLISIS COMPARATIVO ──────────────────────────────────────────────────
    comp = comparative_analysis(req, res)
    if comp:
        st.markdown('<div class="section-title">🔬 Análisis comparativo: enrutados vs sin atender</div>', unsafe_allow_html=True)

        r = comp["routed_stats"]
        u = comp["unatt_stats"]

        # Tabla comparativa
        comp_rows = [
            {"Campo": "Nodos", "Enrutados": r["count"], "Sin atender": u["count"], "Diferencia": "—"},
            {"Campo": "Load promedio",    "Enrutados": r["load_mean"],   "Sin atender": u["load_mean"],   "Diferencia": f"{r['load_mean']/max(u['load_mean'],0.01):.1f}×" if u["load_mean"] > 0 else "—"},
            {"Campo": "Load mediana",     "Enrutados": r["load_median"], "Sin atender": u["load_median"], "Diferencia": "—"},
            {"Campo": "Duration promedio","Enrutados": f"{r['dur_mean']} min",  "Sin atender": f"{u['dur_mean']} min",  "Diferencia": "—"},
            {"Campo": "Ventana promedio", "Enrutados": f"{r['window_mean']} min","Sin atender": f"{u['window_mean']} min","Diferencia": "—"},
        ]
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

        # Diferenciadores detectados
        if comp["differentiators"]:
            for d in comp["differentiators"]:
                sev_color = "#ef4444" if d["severity"] == "high" else "#f59e0b"
                st.markdown(f"""
                <div class="rec-card" style="border-left: 4px solid {sev_color};">
                    <div class="rec-title">🔍 Diferenciador: {d['field']}</div>
                    <div class="rec-detail">
                        <b>Enrutados:</b> {d['routed']} &nbsp;|&nbsp;
                        <b>Sin atender:</b> {d['unattended']} &nbsp;|&nbsp;
                        <b>Ratio:</b> {d['diff']}<br><br>
                        {d['conclusion']}
                    </div>
                    <div class="rec-field">✏️ {d['fix']}</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("No se detectó un diferenciador claro entre ambos grupos. Los grupos son estadísticamente similares en carga, duración y ventana.")

    # ── UTILIZACIÓN DE FLOTA ──────────────────────────────────────────────────
    fleet = fleet_utilization(req, res)
    if fleet:
        st.markdown('<div class="section-title">🚛 Utilización de flota</div>', unsafe_allow_html=True)

        f1, f2, f3, f4, f5 = st.columns(5)
        with f1: render_metric("Vehículos activos",  fleet["active"])
        with f2: render_metric("Vehículos inactivos", fleet["idle"], "#f59e0b" if fleet["alert_idle"] else "#6b7280")
        with f3: render_metric("Visitas promedio/veh", fleet["avg_visits"])
        with f4: render_metric("Máx visitas/veh",    fleet["max_visits"])
        with f5: render_metric("Al 90%+ capacidad",  fleet["at_cap"], "#ef4444" if fleet["alert_at_cap"] else "#6b7280")

        # Alertas sistémicas
        if fleet["alert_single"]:
            st.error(
                f"⚠️ **{fleet['single_visit_ratio']:.0f}% de los vehículos activos hicieron exactamente 1 visita.** "
                f"Esto indica que los vehículos no pueden encadenar visitas — posiblemente por ventanas de tiempo "
                f"demasiado estrechas o ubicaciones de inicio alejadas de los nodos."
            )
        if fleet["alert_idle"]:
            st.warning(
                f"🚛 **{fleet['idle']} vehículos ({fleet['idle']/max(fleet['total'],1)*100:.0f}%) no realizaron ninguna visita.** "
                f"Revisar si sus ubicaciones de inicio o turnos son compatibles con los nodos disponibles."
            )
        if fleet["alert_at_cap"]:
            st.error(
                f"⚖️ **{fleet['at_cap']} vehículos llegaron al 90%+ de su capacidad de carga.** "
                f"La capacidad de la flota puede ser insuficiente para absorber todos los nodos."
            )

        # Tabla por vehículo (solo activos)
        with st.expander("Ver detalle por vehículo", expanded=False):
            veh_rows = []
            for vid, vd in sorted(fleet["veh_data"].items(), key=lambda x: -x[1]["visits"]):
                cap_str = f"{vd['cap_pct']}%" if vd["cap_pct"] is not None else "cap. ilimitada"
                veh_rows.append({
                    "Vehículo":      vid,
                    "Visitas":       vd["visits"],
                    "Carga total":   vd["load1"],
                    "Capacidad":     vd["cap1"] if vd["cap1"] < 1e15 else "ilimitada",
                    "% Capacidad":   cap_str,
                    "Estado":        "🔴 Al límite" if vd["at_cap"] else ("🟢 Activo" if vd["visits"] > 0 else "⚪ Inactivo"),
                })
            st.dataframe(pd.DataFrame(veh_rows), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">📊 Análisis por zona</div>', unsafe_allow_html=True)
    zone_rows = []
    for z, zs in findings["zone_stats"].items():
        zone_rows.append({
            "Zona":                str(z),
            "Nodos":               zs["total_nodes"],
            "Vehículos":           zs["num_vehicles"],
            "Dur. total (min)":    zs["total_dur"],
            "Ventana/veh (min)":   zs["avail_min_veh"]  if zs["avail_min_veh"]  > 0 else "—",
            "Ventana total (min)": zs["avail_min"]       if zs["avail_min"]      > 0 else "—",
            "Turno":               f"{zs['shift_info'][0]}–{zs['shift_info'][1]}",
            "Outliers duration":   zs["outlier_count"],
            "Desborde":            "🔴 Sí" if zs["overflow"] else "🟢 No",
        })
    st.dataframe(pd.DataFrame(zone_rows), use_container_width=True, hide_index=True)

    if findings["raw_issue_counts"]:
        st.markdown('<div class="section-title">🔍 Distribución de causas</div>', unsafe_allow_html=True)
        ic = findings["raw_issue_counts"]
        labels_map = {k: v[1] for k, v in ISSUE_LABELS.items()}
        labels = [labels_map.get(k, k) for k in ic.keys()]
        counts = list(ic.values())
        colors = ["#ef4444","#f59e0b","#3b82f6","#6b7280","#16a34a",
                  "#8b5cf6","#06b6d4","#ec4899","#84cc16","#f97316"]
        fig2 = go.Figure(go.Bar(
            x=counts, y=labels, orientation="h",
            marker_color=colors[:len(labels)],
            text=counts, textposition="outside",
        ))
        fig2.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            height=max(200, len(labels) * 40 + 60),
            margin=dict(t=10, b=40, l=20, r=60),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-title">📋 Tabla de nodos sin atender</div>', unsafe_allow_html=True)
    with st.expander("Ver tabla completa", expanded=False):
        rows = []
        for n in findings["unattended"]:
            icon, label, _ = ISSUE_LABELS.get(n["primary_type"], ("❓", n["primary_type"], "badge-gray"))
            rows.append({
                "Ident":              n["ident"],
                "Dirección":          n["address"],
                "Causa router":       n["cause_code"],
                "Problema principal": f"{icon} {label}",
                "Zona(s)":            str(n["zones"]) if n["zones"] else "sin zona",
                "Duration":           n["duration"],
                "Ventana":            n["window"],
                "Load 1/2/3":         f"{n['load']:.1f} / {n['load_2']:.2f} / {n['load_3']:.1f}",
                "# Problemas":        len(n["issues"]),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">🔎 Diagnóstico por nodo</div>', unsafe_allow_html=True)
    for n in findings["unattended"]:
        icon, label, _ = ISSUE_LABELS.get(n["primary_type"], ("❓", n["primary_type"], "badge-gray"))
        header = f"{icon} **{n['ident']}** — {n['address'][:55]}{'…' if len(n['address'])>55 else ''}"
        with st.expander(header, expanded=False):
            ca, cb = st.columns([3, 2])
            with ca:
                st.markdown(
                    f"**Zona(s):** {n['zones'] or 'sin zona'}  \n"
                    f"**Causa router:** `{n['cause_code']}`  \n"
                    f"**Ventana:** `{n['window']}`  \n"
                    f"**Duration:** `{n['duration']} min`"
                )
            with cb:
                st.markdown(
                    f"**Load 1:** {n['load']:.2f}  \n"
                    f"**Load 2:** {n['load_2']:.3f}  \n"
                    f"**Load 3:** {n['load_3']:.2f}"
                )
            st.markdown("---")
            for iss in n["issues"]:
                sev_icon = {"high": "🔴", "medium": "🟡", "low": "⚪"}.get(iss["severity"], "⚪")
                i_icon, i_label, _ = ISSUE_LABELS.get(iss["type"], ("❓", iss["type"], "badge-gray"))
                st.markdown(
                    f"{sev_icon} **{i_icon} {i_label}** — `{iss['field']}` = `{iss['value']}`  \n"
                    f"&nbsp;&nbsp;&nbsp;{iss['detail']}  \n"
                    f"&nbsp;&nbsp;&nbsp;✏️ *{iss.get('fix', '')}*"
                )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:12px;color:#9ca3af;text-align:center;">'
        'SimpliRoute Route Analyzer v2 · E01/E02/E03/E99 + max_visit + geo distance'
        '</div>', unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()