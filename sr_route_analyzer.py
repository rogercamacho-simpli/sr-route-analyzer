"""
SimpliRoute — Route Analyzer
Analiza request.json + response.json del router y entrega diagnóstico
de nodos sin atender, errores E500 y validaciones pre-vuelo.
"""

import streamlit as st
import json
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

def detect_outliers_iqr(values):
    arr = [v for v in values if v is not None and v > 0]
    if len(arr) < 4:
        return set()
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    return {v for v in arr if v > upper}

def format_time_min(minutes):
    if minutes is None:
        return "N/A"
    return f"{int(minutes // 60)}h {int(minutes % 60):02d}m"

def is_error_response(res):
    return "errors" in res and "vehicles" not in res

VALID_FMV = {1, 2, 3, 1.0, 2.0, 3.0}

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
    "group_invalid":         ("🔗", "Grupo inválido",           "badge-blue"),
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
                f"El campo fmv tiene el valor {fmv}, que no pertenece al conjunto válido {{1, 2, 3}}. "
                f"Puede causar un error E500 en el router ya que el validador no lo detecta previamente."
            ),
            "fix": "Cambiar fmv a 1, 2 o 3",
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

        # Ventana de 0 minutos con duration > 0
        if ws == we and dur and dur > 0:
            zero_window_nodes.append({
                "ident":    ident,
                "address":  node.get("address", "")[:55],
                "window":   f"{ws}–{we}",
                "duration": dur,
            })
        elif ws_m is not None and we_m is not None:
            # Ventana invertida (E01006)
            if not (ws2 == "23:59" and we2 == "23:59") and ws_m > we_m:
                inverted_nodes.append({
                    "ident":   ident,
                    "address": node.get("address", "")[:55],
                    "window":  f"{ws}–{we}",
                })
            # Ventana más pequeña que la duración
            elif dur and 0 < (we_m - ws_m) < dur:
                narrow_nodes.append({
                    "ident":        ident,
                    "address":      node.get("address", "")[:55],
                    "window_size":  we_m - ws_m,
                    "duration":     dur,
                })

        # Coordenadas (0,0) — E02004
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
                f"pero con una duración de servicio mayor a 0. El nodo no puede ser atendido "
                f"dentro de una ventana de 0 minutos."
            ),
            "fix": "Ampliar window_end o establecer duration=0 si no requiere tiempo de servicio",
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
            "detail": f"{len(narrow_nodes)} nodo(s) tienen una ventana de tiempo inferior a su duración de servicio.",
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

    # Turnos de vehículos inválidos
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

    all_routed_ids = set()
    for v in res.get("vehicles", []):
        for tour in v.get("tours", []):
            for n in tour.get("nodes", []):
                if not n["ident"].startswith("vehicle-"):
                    all_routed_ids.add(n["ident"])

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
        durations_valid = [try_int(n.get("duration")) for n in nodes]
        durations_valid = [d for d in durations_valid if d is not None]
        outlier_vals    = detect_outliers_iqr(durations_valid)
        total_dur       = sum(durations_valid)
        outlier_count   = sum(1 for d in durations_valid if d in outlier_vals)
        normal_durs     = [d for d in durations_valid if d not in outlier_vals]
        median_normal   = float(np.median(normal_durs)) if normal_durs else 0
        corrected_dur   = sum(normal_durs) + outlier_count * median_normal

        veh_ids    = zone_to_vehicles.get(zone, vehicles_no_zone if zone == "sin_zona" else [])
        avail_min  = 0
        shift_info = ("00:01", "23:59")
        if veh_ids:
            v  = vehicles.get(veh_ids[0], {})
            ss = v.get("shift_start", "00:01")
            se = v.get("shift_end",   "23:59")
            ps, pe = parse_time(ss), parse_time(se)
            if ps is not None and pe is not None:
                avail_min  = pe - ps
                shift_info = (ss, se)

        findings["zone_stats"][zone] = {
            "total_nodes":   len(nodes),
            "total_dur":     total_dur,
            "avail_min":     avail_min,
            "overflow":      total_dur > avail_min if avail_min > 0 else False,
            "outlier_vals":  outlier_vals,
            "outlier_count": outlier_count,
            "vehicles":      veh_ids,
            "shift_info":    shift_info,
            "mean_dur":      np.mean(durations_valid) if durations_valid else 0,
            "corrected_dur": corrected_dur,
        }

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

        # Ventana de 0 minutos
        if node_ws == node_we and node_dur and node_dur > 0:
            issues.append({
                "type": "zero_window", "severity": "high",
                "field": "window_start / window_end",
                "value": f"{node_ws} == {node_we}, duration={node_dur}",
                "detail": f"Ventana de 0 minutos con duration={node_dur} min. El nodo no puede ser atendido.",
                "fix":    "Ampliar window_end o establecer duration=0",
            })

        # Ventana estrecha
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

        # Duration outlier
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

        # Desborde de tiempo en zona
        for z in (node_zones if node_zones else ["sin_zona"]):
            zs = findings["zone_stats"].get(z, {})
            if zs.get("overflow"):
                issues.append({
                    "type": "zone_time_overflow", "severity": "high",
                    "field": "duration (zona)",
                    "value": f"{zs['total_dur']} / {zs['avail_min']} min",
                    "detail": (
                        f"Zona {z}: servicio total {format_time_min(zs['total_dur'])} "
                        f"excede ventana {format_time_min(zs['avail_min'])}."
                    ),
                    "fix": "Corregir durations outlier y/o agregar vehículo",
                })
                break

        # Ventana vs turno
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

        # Capacidad
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

        # Zona sin vehículo
        if node_zones and not any(z in zone_to_vehicles for z in node_zones):
            issues.append({
                "type": "zone_mismatch", "severity": "medium",
                "field": "zones", "value": str(node_zones),
                "detail": f"Ningún vehículo tiene asignada la zona {node_zones}.",
                "fix":    "Asignar la zona a un vehículo disponible",
            })

        # Skills
        if node_skills:
            all_skills = set()
            for v in vehicles.values():
                all_skills.update(v.get("skills", []))
            missing = [s for s in node_skills if s not in all_skills]
            if missing:
                issues.append({
                    "type": "skills_mismatch", "severity": "medium",
                    "field": "skills_required", "value": str(missing),
                    "detail": f"Ningún vehículo tiene las skills: {missing}.",
                    "fix":    "Agregar las skills al vehículo correspondiente",
                })

        # Grupo inválido
        if len(node_group) > 1:
            missing_group = [g for g in node_group if g not in node_map]
            if missing_group:
                issues.append({
                    "type": "group_invalid", "severity": "medium",
                    "field": "group", "value": str(node_group),
                    "detail": f"Nodos del grupo no encontrados en el request: {missing_group}.",
                    "fix":    "Verificar que todos los idents del grupo existan en el request",
                })

        # Fallback
        if not issues:
            if cause_code == "EXC_SO-002":
                issues.append({
                    "type": "clustering_preference", "severity": "low",
                    "field": "beauty", "value": cause_code,
                    "detail": "Excluido por preferencia de agrupación del optimizador.",
                    "fix":    "Probar con beauty=false",
                })
            else:
                issues.append({
                    "type": "capacity_time_general", "severity": "medium",
                    "field": "tiempo / capacidad", "value": cause_code,
                    "detail": cause_msg or "Nodo excluido por falta de capacidad o tiempo.",
                    "fix":    "Revisar carga, duración y ventanas del nodo",
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

    # Recomendaciones
    recs = []
    for z, zs in findings["zone_stats"].items():
        if zs.get("overflow") and zs.get("outlier_count", 0) > 0:
            recs.append({
                "priority": 1, "color": "#ef4444",
                "title": f"Corregir duration outlier en zona {z}",
                "detail": (
                    f"{zs['outlier_count']} nodo(s) con duration={zs['outlier_vals']}. "
                    f"Corrigiéndolos el tiempo baja de {format_time_min(zs['total_dur'])} "
                    f"a ~{format_time_min(zs['corrected_dur'])}, dentro de {format_time_min(zs['avail_min'])}."
                ),
                "field": "duration", "affected": zs["outlier_count"],
            })
        elif zs.get("overflow"):
            recs.append({
                "priority": 1, "color": "#ef4444",
                "title": f"Agregar vehículo o ampliar turno en zona {z}",
                "detail": (
                    f"Zona {z} requiere {format_time_min(zs['total_dur'])} "
                    f"pero solo hay {format_time_min(zs['avail_min'])}."
                ),
                "field": "shift_end o nuevo vehículo", "affected": zs["total_nodes"],
            })

    issue_recs = [
        ("zero_window",           "Corregir nodos con ventana de 0 minutos",                  1, "#ef4444", "window_start / window_end"),
        ("window_shift_mismatch", "Ajustar ventanas incompatibles con el turno",               2, "#f59e0b", "window_start / window_end"),
        ("narrow_window",         "Ampliar ventanas más pequeñas que la duración de servicio", 2, "#f59e0b", "window_end"),
        ("inverted_window",       "Corregir ventanas horarias invertidas (E01006)",            2, "#f59e0b", "window_start / window_end"),
        ("zone_mismatch",         "Asignar vehículos a zonas sin cobertura",                  2, "#f59e0b", "zones (vehículo)"),
        ("capacity_overflow",     "Revisar nodos que exceden la capacidad del vehículo",       2, "#f59e0b", "load / capacity"),
        ("skills_mismatch",       "Agregar skills faltantes a vehículos",                     3, "#3b82f6", "skills (vehículo)"),
        ("group_invalid",         "Corregir referencias de grupos inválidas",                  3, "#3b82f6", "group"),
        ("clustering_preference", "Evaluar desactivar parámetro beauty",                      4, "#6b7280", "beauty"),
    ]
    for issue_type, label, priority, color, field in issue_recs:
        if issue_counts.get(issue_type, 0) > 0:
            recs.append({
                "priority": priority, "color": color,
                "title":  label,
                "detail": f"{issue_counts[issue_type]} nodo(s) afectado(s).",
                "field":  field, "affected": issue_counts[issue_type],
            })

    if vehicles_no_zone:
        for z, zs in findings["zone_stats"].items():
            if zs.get("overflow") and z != "sin_zona":
                recs.append({
                    "priority": 3, "color": "#3b82f6",
                    "title":  f"Asignar vehículos sin zona a zona {z}",
                    "detail": f"Vehículos {vehicles_no_zone} no tienen zona asignada y pueden absorber visitas.",
                    "field":  "zones (vehículo)", "affected": len(vehicles_no_zone),
                })
                break

    recs.sort(key=lambda r: r["priority"])
    findings["recommendations"] = recs

    findings["summary"] = {
        "total_nodes":      len(req.get("nodes", [])),
        "total_vehicles":   len(req.get("vehicles", [])),
        "routed_nodes":     len(all_routed_ids),
        "unattended_count": len(unattended),
        "filtered_count":   len(filtered),
        "vehicles_used":    res.get("num_vehicles_used", 0),
        "so002": sum(1 for u in unattended if u.get("cause", {}).get("code") == "EXC_SO-002"),
        "so003": sum(1 for u in unattended if u.get("cause", {}).get("code") == "EXC_SO-003"),
        "attendance_rate": round(len(all_routed_ids) / max(len(req.get("nodes", [])), 1) * 100, 1),
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
                if "window" in n:
                    extras.append(f"ventana: {n['window']}")
                if "duration" in n:
                    extras.append(f"duration: {n['duration']} min")
                if "window_size" in n:
                    extras.append(f"ventana: {n['window_size']} min")
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
        st.markdown('<div class="upload-label">📥 Cargar Request JSON</div>', unsafe_allow_html=True)
        req_file = st.file_uploader("request", type="json", key="req", label_visibility="collapsed")
    with col2:
        st.markdown('<div class="upload-label">📤 Cargar Response JSON</div>', unsafe_allow_html=True)
        res_file = st.file_uploader("response", type="json", key="res", label_visibility="collapsed")

    if not req_file or not res_file:
        st.info("Carga ambos archivos para iniciar el análisis.")
        return

    try:
        req = json.load(req_file)
        res = json.load(res_file)
    except Exception as e:
        st.error(f"Error al parsear los archivos JSON: {e}")
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

    # ── NODOS SIN ATENDER ─────────────────────────────────────────────────────
    unattended = res.get("unattendedClientsNodes", [])
    if not unattended:
        st.success("✅ No hay nodos sin atender en este response.")
        return

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

    st.markdown('<div class="section-title">🎯 Recomendaciones priorizadas</div>', unsafe_allow_html=True)
    if not findings["recommendations"]:
        st.info("No se generaron recomendaciones.")
    else:
        for idx, rec in enumerate(findings["recommendations"], 1):
            render_recommendation(rec, idx)

    st.markdown('<div class="section-title">📊 Análisis por zona</div>', unsafe_allow_html=True)
    zone_rows = []
    for z, zs in findings["zone_stats"].items():
        zone_rows.append({
            "Zona":               str(z),
            "Nodos":              zs["total_nodes"],
            "Dur. total (min)":   zs["total_dur"],
            "Ventana (min)":      zs["avail_min"] if zs["avail_min"] > 0 else "—",
            "Turno":              f"{zs['shift_info'][0]}–{zs['shift_info'][1]}",
            "Outliers duration":  zs["outlier_count"],
            "Desborde":           "🔴 Sí" if zs["overflow"] else "🟢 No",
        })
    st.dataframe(pd.DataFrame(zone_rows), use_container_width=True, hide_index=True)

    zones_data = [(z, zs) for z, zs in findings["zone_stats"].items() if zs["avail_min"] > 0]
    if zones_data:
        fig = go.Figure()
        z_labels = [str(z) for z, _ in zones_data]
        fig.add_bar(
            name="Tiempo servicio total", x=z_labels,
            y=[zs["total_dur"] for _, zs in zones_data],
            marker_color=["#ef4444" if zs["overflow"] else "#3b82f6" for _, zs in zones_data],
        )
        fig.add_bar(
            name="Ventana disponible", x=z_labels,
            y=[zs["avail_min"] for _, zs in zones_data],
            marker_color="#d1d5db",
        )
        fig.update_layout(
            barmode="group", title="Tiempo de servicio vs ventana disponible por zona",
            plot_bgcolor="white", paper_bgcolor="white", height=300,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

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
        'SimpliRoute Route Analyzer · E01/E02/E03/E99 + análisis de nodos sin atender'
        '</div>', unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()