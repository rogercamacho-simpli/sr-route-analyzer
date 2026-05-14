"""
SimpliRoute — Route Analyzer
Analiza request.json + response.json del router y entrega diagnóstico
de nodos sin atender con recomendaciones priorizadas.
"""

import streamlit as st
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict, Counter

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SR Route Analyzer",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #f7f8fc; }
    .main-header { font-size: 26px; font-weight: 700; color: #1a1d2e; margin-bottom: 0; }
    .sub-header  { font-size: 14px; color: #6b7280; margin-top: 4px; margin-bottom: 1.5rem; }
    .metric-card {
        background: white; border-radius: 12px; padding: 18px 20px;
        border: 1px solid #e8eaf0; margin-bottom: 0;
    }
    .metric-val { font-size: 28px; font-weight: 700; line-height: 1.1; }
    .metric-lbl { font-size: 12px; color: #6b7280; margin-top: 4px; }
    .rec-card {
        background: white; border-radius: 12px; padding: 18px 20px;
        border-left: 4px solid #ccc; margin-bottom: 12px;
        border: 1px solid #e8eaf0;
    }
    .rec-p1 { border-left-color: #ef4444 !important; }
    .rec-p2 { border-left-color: #f59e0b !important; }
    .rec-p3 { border-left-color: #3b82f6 !important; }
    .rec-p4 { border-left-color: #6b7280 !important; }
    .rec-title { font-size: 15px; font-weight: 600; color: #1a1d2e; margin-bottom: 6px; }
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
    .issue-chip {
        display: inline-block; padding: 2px 8px; border-radius: 6px;
        font-size: 11px; background: #f3f4f6; color: #374151; margin: 2px;
    }
    hr.divider { border: none; border-top: 1px solid #e8eaf0; margin: 1.5rem 0; }
    .upload-box {
        background: white; border-radius: 12px; padding: 1.5rem;
        border: 2px dashed #d1d5db; text-align: center;
    }
    .stFileUploader > div { border-radius: 12px; }
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
    """Convierte 'H:MM' a minutos desde medianoche. None si inválido."""
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
    """Retorna el conjunto de valores considerados outliers (IQR method)."""
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
    h = int(minutes // 60)
    m = int(minutes % 60)
    return f"{h}h {m:02d}m"

ISSUE_LABELS = {
    "duration_anomaly":       ("🕐", "Duration outlier",        "badge-red"),
    "zone_time_overflow":     ("⏱️", "Desborde de tiempo",      "badge-red"),
    "window_shift_mismatch":  ("🚫", "Ventana vs turno",        "badge-red"),
    "narrow_window":          ("📏", "Ventana estrecha",         "badge-red"),
    "capacity_overflow":      ("⚖️", "Capacidad excedida",      "badge-amber"),
    "zone_mismatch":          ("🗺️", "Zona sin vehículo",       "badge-amber"),
    "skills_mismatch":        ("🔧", "Skills faltantes",        "badge-amber"),
    "group_invalid":          ("🔗", "Grupo inválido",          "badge-blue"),
    "clustering_preference":  ("✦",  "Excluido por clustering", "badge-gray"),
    "capacity_time_general":  ("⚠️", "Cap/tiempo general",      "badge-amber"),
    "unknown":                ("❓", "Causa desconocida",        "badge-gray"),
}

PRIORITY_COLORS = {1: "#ef4444", 2: "#f59e0b", 3: "#3b82f6", 4: "#6b7280"}
PRIORITY_LABELS = {1: "🔴 Alta", 2: "🟡 Media", 3: "🔵 Baja", 4: "⚪ Info"}


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def analyze(req: dict, res: dict) -> dict:
    findings = {
        "summary": {},
        "unattended": [],
        "zone_stats": {},
        "recommendations": [],
        "raw_issue_counts": Counter(),
    }

    node_map   = {n["ident"]: n for n in req.get("nodes", [])}
    vehicles   = {v["ident"]: v for v in req.get("vehicles", [])}
    unattended = res.get("unattendedClientsNodes", [])
    filtered   = res.get("filteredClientsNodes", [])

    # ── Routed nodes ──────────────────────────────────────────────────────────
    all_routed_ids = set()
    for v in res.get("vehicles", []):
        for tour in v.get("tours", []):
            for n in tour.get("nodes", []):
                if not n["ident"].startswith("vehicle-"):
                    all_routed_ids.add(n["ident"])

    # ── Zone → vehicles mapping ───────────────────────────────────────────────
    zone_to_vehicles = defaultdict(list)
    for vid, v in vehicles.items():
        for z in v.get("zones", []):
            zone_to_vehicles[z].append(vid)
    vehicles_no_zone = [vid for vid, v in vehicles.items() if not v.get("zones")]

    # ── Zone stats ────────────────────────────────────────────────────────────
    zone_nodes = defaultdict(list)
    for n in req.get("nodes", []):
        zones = n.get("zones", [])
        if zones:
            for z in zones:
                zone_nodes[z].append(n)
        else:
            zone_nodes["sin_zona"].append(n)

    for zone, nodes in zone_nodes.items():
        durations = [try_int(n.get("duration")) for n in nodes]
        durations_valid = [d for d in durations if d is not None]
        outlier_vals    = detect_outliers_iqr(durations_valid)
        total_dur       = sum(durations_valid)
        outlier_count   = sum(1 for d in durations_valid if d in outlier_vals)

        veh_ids    = zone_to_vehicles.get(zone, vehicles_no_zone if zone == "sin_zona" else [])
        avail_min  = 0
        shift_info = ("00:01", "23:59")
        if veh_ids:
            v = vehicles.get(veh_ids[0], {})
            ss, se = v.get("shift_start", "00:01"), v.get("shift_end", "23:59")
            ps, pe = parse_time(ss), parse_time(se)
            if ps is not None and pe is not None:
                avail_min  = pe - ps
                shift_info = (ss, se)

        findings["zone_stats"][zone] = {
            "total_nodes":       len(nodes),
            "total_dur":         total_dur,
            "avail_min":         avail_min,
            "overflow":          total_dur > avail_min if avail_min > 0 else False,
            "outlier_vals":      outlier_vals,
            "outlier_count":     outlier_count,
            "vehicles":          veh_ids,
            "shift_info":        shift_info,
            "mean_dur":          np.mean(durations_valid) if durations_valid else 0,
            "corrected_dur":     total_dur - sum(d for d in durations_valid if d in outlier_vals)
                                 + len([d for d in durations_valid if d in outlier_vals])
                                 * (np.median([d for d in durations_valid if d not in outlier_vals]) if
                                    [d for d in durations_valid if d not in outlier_vals] else 0),
        }

    # ── Per-node analysis ─────────────────────────────────────────────────────
    issue_counts = Counter()
    cause_details = []

    for un in unattended:
        ident      = un["ident"]
        cause_code = un.get("cause", {}).get("code", "")
        cause_msg  = un.get("cause", {}).get("details", "")
        req_node   = node_map.get(ident, {})

        node_zones  = req_node.get("zones", [])
        node_dur    = try_int(req_node.get("duration"))
        node_ws     = req_node.get("window_start", "00:00")
        node_we     = req_node.get("window_end", "23:59")
        node_skills = req_node.get("skills_required", [])
        node_group  = req_node.get("group", [ident])
        node_arr_tw = req_node.get("arrayOfTimeWindows", [])

        issues = []

        # Candidate vehicles for this node
        cand_vehicles = []
        for z in node_zones:
            cand_vehicles.extend(zone_to_vehicles.get(z, []))
        if not cand_vehicles:
            cand_vehicles = vehicles_no_zone[:]

        # 1. Duration outlier
        for z in (node_zones if node_zones else ["sin_zona"]):
            zs = findings["zone_stats"].get(z, {})
            if node_dur is not None and node_dur in zs.get("outlier_vals", set()):
                normal_dur = zs.get("mean_dur", 0)
                issues.append({
                    "type": "duration_anomaly",
                    "severity": "high",
                    "field": "duration",
                    "value": node_dur,
                    "detail": (
                        f"duration={node_dur} min es un outlier estadístico (IQR) en zona {z}. "
                        f"La media del resto de nodos es ~{normal_dur:.0f} min. "
                        f"Corregirlo al valor real reduce el tiempo acumulado de la zona."
                    ),
                    "fix": f"Reemplazar duration={node_dur} por el tiempo real de servicio (~{normal_dur:.0f} min)",
                })
                break

        # 2. Zone time overflow
        for z in (node_zones if node_zones else ["sin_zona"]):
            zs = findings["zone_stats"].get(z, {})
            if zs.get("overflow"):
                issues.append({
                    "type": "zone_time_overflow",
                    "severity": "high",
                    "field": "duration (zona)",
                    "value": f"{zs['total_dur']} / {zs['avail_min']} min",
                    "detail": (
                        f"Zona {z}: tiempo total de servicio = {zs['total_dur']} min "
                        f"({format_time_min(zs['total_dur'])}), ventana disponible = "
                        f"{format_time_min(zs['avail_min'])} "
                        f"(turno {zs['shift_info'][0]}–{zs['shift_info'][1]}). "
                        f"Desborde de {zs['total_dur'] - zs['avail_min']} min."
                    ),
                    "fix": "Corregir durations outlier y/o agregar un vehículo a la zona",
                })
                break

        # 3. Window vs vehicle shift mismatch
        if cand_vehicles:
            any_overlap = any(
                time_overlap(node_ws, node_we,
                             vehicles[vid].get("shift_start", "00:01"),
                             vehicles[vid].get("shift_end", "23:59"))
                for vid in cand_vehicles if vid in vehicles
            )
            if not any_overlap:
                issues.append({
                    "type": "window_shift_mismatch",
                    "severity": "high",
                    "field": "window_start / window_end",
                    "value": f"{node_ws}–{node_we}",
                    "detail": (
                        f"La ventana del nodo ({node_ws}–{node_we}) no se solapa con el turno "
                        f"de ningún vehículo candidato."
                    ),
                    "fix": "Ajustar window_start/window_end del nodo, o el shift del vehículo",
                })

        # 4. Narrow window (window smaller than duration)
        ws_min = parse_time(node_ws)
        we_min = parse_time(node_we)
        if ws_min is not None and we_min is not None and node_dur is not None:
            window_size = we_min - ws_min
            if 0 < window_size < node_dur:
                issues.append({
                    "type": "narrow_window",
                    "severity": "high",
                    "field": "window_end",
                    "value": f"ventana={window_size} min < duration={node_dur} min",
                    "detail": (
                        f"La ventana de tiempo del nodo ({window_size} min) es menor que "
                        f"su duración de servicio ({node_dur} min). El nodo es imposible de atender."
                    ),
                    "fix": "Ampliar window_end o reducir duration",
                })

        # 5. Inverted time window (E01006)
        if ws_min is not None and we_min is not None:
            if node_ws != "23:59" and node_we != "23:59" and ws_min > we_min:
                issues.append({
                    "type": "inverted_window",
                    "severity": "high",
                    "field": "window_start / window_end",
                    "value": f"{node_ws} > {node_we}",
                    "detail": "Ventana invertida: window_start es posterior a window_end (E01006).",
                    "fix": "Intercambiar window_start y window_end",
                })

        # 6. Capacity overflow (per-node vs vehicle)
        for vid in cand_vehicles:
            v = vehicles.get(vid, {})
            cap1 = v.get("capacity", 0) or 0
            cap2 = v.get("capacity_2", 0) or 0
            cap3 = v.get("capacity_3", 0) or 0
            l1, l2, l3 = un.get("load", 0), un.get("load_2", 0), un.get("load_3", 0)
            if l1 > cap1 or l2 > cap2 or l3 > cap3:
                issues.append({
                    "type": "capacity_overflow",
                    "severity": "high",
                    "field": "load",
                    "value": f"{l1}/{cap1} | {l2}/{cap2} | {l3}/{cap3}",
                    "detail": f"La carga del nodo excede la capacidad del vehículo {vid}.",
                    "fix": "Revisar la carga del nodo o la capacidad del vehículo",
                })
                break

        # 7. Zone mismatch (no vehicle covers the node's zone)
        if node_zones:
            covered = any(z in zone_to_vehicles for z in node_zones)
            if not covered:
                issues.append({
                    "type": "zone_mismatch",
                    "severity": "medium",
                    "field": "zones",
                    "value": str(node_zones),
                    "detail": f"Ningún vehículo tiene asignada la zona {node_zones}.",
                    "fix": "Asignar la zona a un vehículo disponible",
                })

        # 8. Skills mismatch (E03003)
        if node_skills:
            all_veh_skills = set()
            for v in vehicles.values():
                all_veh_skills.update(v.get("skills", []))
            missing = [s for s in node_skills if s not in all_veh_skills]
            if missing:
                issues.append({
                    "type": "skills_mismatch",
                    "severity": "medium",
                    "field": "skills_required",
                    "value": str(missing),
                    "detail": f"Ningún vehículo tiene las skills requeridas: {missing}.",
                    "fix": "Agregar las skills al vehículo correspondiente",
                })

        # 9. Invalid group references
        if len(node_group) > 1:
            missing_group = [g for g in node_group if g not in node_map]
            if missing_group:
                issues.append({
                    "type": "group_invalid",
                    "severity": "medium",
                    "field": "group",
                    "value": str(node_group),
                    "detail": f"Nodos del grupo no encontrados en el request: {missing_group}.",
                    "fix": "Verificar que todos los idents del grupo existan en el request",
                })

        # 10. Coordinates (0,0) — E02004
        lat = req_node.get("lat", None)
        lon = req_node.get("lon", None)
        if lat == 0 and lon == 0:
            issues.append({
                "type": "zero_coordinates",
                "severity": "high",
                "field": "lat / lon",
                "value": "(0, 0)",
                "detail": "Las coordenadas del nodo son (0, 0). El nodo no tiene geolocalización válida (E02004).",
                "fix": "Corregir las coordenadas lat/lon del nodo",
            })

        # 11. arrayOfTimeWindows conflict (E01008/E01009)
        if node_arr_tw and len(node_arr_tw) > 1:
            for i in range(len(node_arr_tw) - 1):
                tw1 = node_arr_tw[i]
                tw2 = node_arr_tw[i + 1]
                e1 = parse_time(tw1.get("end") or tw1.get("window_end"))
                s2 = parse_time(tw2.get("start") or tw2.get("window_start"))
                if e1 and s2 and e1 > s2:
                    issues.append({
                        "type": "time_window_conflict",
                        "severity": "medium",
                        "field": "arrayOfTimeWindows",
                        "value": f"TW[{i}] y TW[{i+1}] en conflicto",
                        "detail": f"Conflicto entre ventanas horarias en arrayOfTimeWindows[{i}] y [{i+1}] (E01008/E01009).",
                        "fix": "Reordenar o corregir los intervalos en arrayOfTimeWindows",
                    })
                    break

        # Fallback si no se detectó nada específico
        if not issues:
            if cause_code == "EXC_SO-002":
                issues.append({
                    "type": "clustering_preference",
                    "severity": "low",
                    "field": "beauty / clustering",
                    "value": cause_code,
                    "detail": "Nodo excluido por preferencia de agrupación del optimizador (beauty). Tiene capacidad y tiempo disponibles.",
                    "fix": "Probar con beauty=false o agregar el nodo a una zona específica",
                })
            else:
                issues.append({
                    "type": "capacity_time_general",
                    "severity": "medium",
                    "field": "tiempo / capacidad",
                    "value": cause_code,
                    "detail": cause_msg or "Nodo excluido por falta de capacidad o tiempo en vehículos cercanos.",
                    "fix": "Revisar carga, duración y ventanas de tiempo del nodo",
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
            "load":         un.get("load", 0),
            "load_2":       un.get("load_2", 0),
            "load_3":       un.get("load_3", 0),
            "issues":       issues,
            "primary_type": issues[0]["type"] if issues else "unknown",
        })

    findings["unattended"] = cause_details
    findings["raw_issue_counts"] = issue_counts

    # ── Recommendations ───────────────────────────────────────────────────────
    recs = []

    # R1: Duration outliers with zone overflow
    for z, zs in findings["zone_stats"].items():
        if zs.get("overflow") and zs.get("outlier_count", 0) > 0:
            anom = zs["outlier_vals"]
            corrected = zs.get("corrected_dur", 0)
            recs.append({
                "priority": 1,
                "type": "duration_anomaly",
                "title": f"Corregir duration outlier en zona {z}",
                "detail": (
                    f"{zs['outlier_count']} nodo(s) en zona {z} tienen duration={anom}, "
                    f"valor estadísticamente atípico vs. el resto (media ~{zs['mean_dur']:.0f} min). "
                    f"Corrigiéndolos, el tiempo total bajaría de "
                    f"{format_time_min(zs['total_dur'])} a ~{format_time_min(corrected)}, "
                    f"dentro de la ventana disponible de {format_time_min(zs['avail_min'])}."
                ),
                "field": "duration",
                "affected": zs["outlier_count"],
                "color": "rec-p1",
            })

    # R1b: Zone overflow without duration outlier
    for z, zs in findings["zone_stats"].items():
        if zs.get("overflow") and zs.get("outlier_count", 0) == 0:
            recs.append({
                "priority": 1,
                "type": "time_overflow",
                "title": f"Agregar vehículo o ampliar turno en zona {z}",
                "detail": (
                    f"Zona {z} requiere {format_time_min(zs['total_dur'])} de servicio "
                    f"pero el turno disponible es de {format_time_min(zs['avail_min'])}. "
                    f"No se detectaron outliers de duration — se necesita más capacidad temporal."
                ),
                "field": "shift_end o nuevo vehículo",
                "affected": zs["total_nodes"],
                "color": "rec-p1",
            })

    # R2: Window/shift mismatch
    if issue_counts.get("window_shift_mismatch", 0) > 0:
        recs.append({
            "priority": 2,
            "type": "window_shift_mismatch",
            "title": "Ajustar ventanas de tiempo incompatibles con el turno",
            "detail": (
                f"{issue_counts['window_shift_mismatch']} nodo(s) tienen window_start/window_end "
                f"que no se solapa con el turno de ningún vehículo candidato. "
                f"El router no puede asignarlos aunque haya capacidad disponible."
            ),
            "field": "window_start / window_end",
            "affected": issue_counts["window_shift_mismatch"],
            "color": "rec-p2",
        })

    # R2b: Narrow windows
    if issue_counts.get("narrow_window", 0) > 0:
        recs.append({
            "priority": 2,
            "type": "narrow_window",
            "title": "Ampliar ventanas más pequeñas que la duración de servicio",
            "detail": (
                f"{issue_counts['narrow_window']} nodo(s) tienen una ventana de tiempo "
                f"menor que su propia duración de servicio. Son físicamente imposibles de atender."
            ),
            "field": "window_end",
            "affected": issue_counts["narrow_window"],
            "color": "rec-p2",
        })

    # R2c: Inverted windows
    if issue_counts.get("inverted_window", 0) > 0:
        recs.append({
            "priority": 2,
            "type": "inverted_window",
            "title": "Corregir ventanas horarias invertidas (E01006)",
            "detail": (
                f"{issue_counts['inverted_window']} nodo(s) tienen window_start posterior "
                f"a window_end. Corresponde al error E01006 de la validación de inputs."
            ),
            "field": "window_start / window_end",
            "affected": issue_counts["inverted_window"],
            "color": "rec-p2",
        })

    # R3: Zone mismatch
    if issue_counts.get("zone_mismatch", 0) > 0:
        recs.append({
            "priority": 2,
            "type": "zone_mismatch",
            "title": "Asignar vehículos a zonas sin cobertura",
            "detail": (
                f"{issue_counts['zone_mismatch']} nodo(s) tienen zonas para las que no existe "
                f"ningún vehículo asignado. El router los ignora automáticamente (E03004)."
            ),
            "field": "zones (vehículo)",
            "affected": issue_counts["zone_mismatch"],
            "color": "rec-p2",
        })

    # R3b: Capacity overflow
    if issue_counts.get("capacity_overflow", 0) > 0:
        recs.append({
            "priority": 2,
            "type": "capacity_overflow",
            "title": "Revisar carga de nodos que exceden la capacidad",
            "detail": (
                f"{issue_counts['capacity_overflow']} nodo(s) tienen una carga individual "
                f"superior a la capacidad del vehículo más cercano. "
                f"Ningún vehículo puede atenderlos (E03002)."
            ),
            "field": "load / capacity",
            "affected": issue_counts["capacity_overflow"],
            "color": "rec-p2",
        })

    # R3c: Skills
    if issue_counts.get("skills_mismatch", 0) > 0:
        recs.append({
            "priority": 3,
            "type": "skills_mismatch",
            "title": "Agregar skills faltantes a vehículos",
            "detail": (
                f"{issue_counts['skills_mismatch']} nodo(s) requieren skills que ningún "
                f"vehículo tiene asignadas (E03003)."
            ),
            "field": "skills (vehículo)",
            "affected": issue_counts["skills_mismatch"],
            "color": "rec-p3",
        })

    # R3d: Group
    if issue_counts.get("group_invalid", 0) > 0:
        recs.append({
            "priority": 3,
            "type": "group_invalid",
            "title": "Corregir referencias de grupos (group)",
            "detail": (
                f"{issue_counts['group_invalid']} nodo(s) tienen idents en su grupo "
                f"que no existen en el request. Puede provocar comportamiento inesperado."
            ),
            "field": "group",
            "affected": issue_counts["group_invalid"],
            "color": "rec-p3",
        })

    # R3e: Zero coordinates
    if issue_counts.get("zero_coordinates", 0) > 0:
        recs.append({
            "priority": 1,
            "type": "zero_coordinates",
            "title": "Corregir nodos con coordenadas (0, 0)",
            "detail": (
                f"{issue_counts['zero_coordinates']} nodo(s) tienen lat/lon = (0, 0). "
                f"El router no puede geolocalizar estos nodos (E02004)."
            ),
            "field": "lat / lon",
            "affected": issue_counts["zero_coordinates"],
            "color": "rec-p1",
        })

    # R4: Clustering preference
    if issue_counts.get("clustering_preference", 0) > 0:
        recs.append({
            "priority": 4,
            "type": "clustering_preference",
            "title": "Evaluar desactivar parámetro beauty",
            "detail": (
                f"{issue_counts['clustering_preference']} nodo(s) fueron excluidos por el "
                f"optimizador de agrupación (EXC_SO-002). Tienen capacidad y tiempo disponibles, "
                f"pero el algoritmo los excluye para minimizar cruces. Probar con beauty=false."
            ),
            "field": "beauty",
            "affected": issue_counts["clustering_preference"],
            "color": "rec-p4",
        })

    # R: Free vehicles for overloaded zones
    if vehicles_no_zone:
        for z, zs in findings["zone_stats"].items():
            if zs.get("overflow") and z != "sin_zona":
                recs.append({
                    "priority": 3,
                    "type": "add_vehicle_zone",
                    "title": f"Asignar vehículos sin zona a zona {z}",
                    "detail": (
                        f"Los vehículos {vehicles_no_zone} no tienen zona asignada "
                        f"y terminan su ruta con holgura. Asignarlos a zona {z} podría "
                        f"absorber los nodos sin atender."
                    ),
                    "field": "zones (vehículo)",
                    "affected": len(vehicles_no_zone),
                    "color": "rec-p3",
                })
                break

    recs.sort(key=lambda r: r["priority"])
    findings["recommendations"] = recs

    # ── Summary ───────────────────────────────────────────────────────────────
    findings["summary"] = {
        "total_nodes":     len(req.get("nodes", [])),
        "total_vehicles":  len(req.get("vehicles", [])),
        "routed_nodes":    len(all_routed_ids),
        "unattended_count": len(unattended),
        "filtered_count":  len(filtered),
        "vehicles_used":   res.get("num_vehicles_used", 0),
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
    </div>
    """, unsafe_allow_html=True)

def render_badge(text, cls="badge-gray"):
    return f'<span class="badge {cls}">{text}</span>'

def render_recommendation(rec, idx):
    color_map = {"rec-p1": "#ef4444", "rec-p2": "#f59e0b", "rec-p3": "#3b82f6", "rec-p4": "#6b7280"}
    border_color = color_map.get(rec["color"], "#ccc")
    priority_label = PRIORITY_LABELS.get(rec["priority"], "")
    badge_cls = {"rec-p1": "badge-red", "rec-p2": "badge-amber", "rec-p3": "badge-blue", "rec-p4": "badge-gray"}
    bc = badge_cls.get(rec["color"], "badge-gray")

    st.markdown(f"""
    <div class="rec-card" style="border-left: 4px solid {border_color}; border: 1px solid #e8eaf0; border-left: 4px solid {border_color};">
        <div style="display:flex; align-items:flex-start; justify-content:space-between; margin-bottom:6px;">
            <div class="rec-title">{idx}. {rec['title']}</div>
            <div>{render_badge(priority_label, bc)} {render_badge(f"{rec['affected']} afectado(s)", 'badge-gray')}</div>
        </div>
        <div class="rec-detail">{rec['detail']}</div>
        <div class="rec-field">Campo: <code>{rec['field']}</code></div>
    </div>
    """, unsafe_allow_html=True)

def issue_badge_html(issue_type):
    icon, label, cls = ISSUE_LABELS.get(issue_type, ("❓", issue_type, "badge-gray"))
    return f'<span class="badge {cls}">{icon} {label}</span>'


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown("""
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:0.25rem;">
        <span style="font-size:32px;">🗺️</span>
        <div>
            <div class="main-header">SimpliRoute — Route Analyzer</div>
            <div class="sub-header">Diagnóstico de nodos sin atender · Recomendaciones priorizadas</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── File upload ───────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📥 Request JSON**")
        req_file = st.file_uploader("request.json", type="json", key="req",
                                     label_visibility="collapsed")
    with col2:
        st.markdown("**📤 Response JSON**")
        res_file = st.file_uploader("response.json", type="json", key="res",
                                     label_visibility="collapsed")

    if not req_file or not res_file:
        st.info("Carga ambos archivos para iniciar el análisis.")
        return

    # ── Parse ─────────────────────────────────────────────────────────────────
    try:
        req = json.load(req_file)
        res = json.load(res_file)
    except Exception as e:
        st.error(f"Error al parsear los archivos JSON: {e}")
        return

    unattended = res.get("unattendedClientsNodes", [])
    if not unattended:
        st.success("✅ No hay nodos sin atender en este response.")
        return

    # ── Analyze ───────────────────────────────────────────────────────────────
    with st.spinner("Analizando..."):
        findings = analyze(req, res)

    s = findings["summary"]

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Summary metrics ───────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: render_metric("Total nodos", s["total_nodes"])
    with c2: render_metric("Enrutados", s["routed_nodes"], "#16a34a")
    with c3: render_metric("Sin atender", s["unattended_count"], "#dc2626")
    with c4: render_metric("Tasa de atención", s["attendance_rate"], "#1d4ed8", "%")
    with c5: render_metric("Vehículos usados", f"{s['vehicles_used']}/{s['total_vehicles']}")
    with c6: render_metric("EXC_SO-002 / 003", f"{s['so002']} / {s['so003']}")

    # ── Recommendations ───────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🎯 Recomendaciones priorizadas</div>', unsafe_allow_html=True)

    if not findings["recommendations"]:
        st.info("No se generaron recomendaciones automáticas.")
    else:
        for idx, rec in enumerate(findings["recommendations"], 1):
            render_recommendation(rec, idx)

    # ── Zone stats ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📊 Análisis por zona</div>', unsafe_allow_html=True)

    zone_rows = []
    for z, zs in findings["zone_stats"].items():
        zone_rows.append({
            "Zona": str(z),
            "Nodos": zs["total_nodes"],
            "Dur. total (min)": zs["total_dur"],
            "Ventana disponible (min)": zs["avail_min"] if zs["avail_min"] > 0 else "—",
            "Turno vehículo": f"{zs['shift_info'][0]}–{zs['shift_info'][1]}",
            "Outliers duration": zs["outlier_count"],
            "Valores outlier": str(zs["outlier_vals"]) if zs["outlier_vals"] else "—",
            "Desborde": "🔴 Sí" if zs["overflow"] else "🟢 No",
        })

    zone_df = pd.DataFrame(zone_rows)
    st.dataframe(zone_df, use_container_width=True, hide_index=True)

    # Bar chart: tiempo total vs ventana por zona
    zones_with_data = [(z, zs) for z, zs in findings["zone_stats"].items() if zs["avail_min"] > 0]
    if zones_with_data:
        fig = go.Figure()
        z_labels = [str(z) for z, _ in zones_with_data]
        fig.add_bar(
            name="Tiempo servicio total",
            x=z_labels,
            y=[zs["total_dur"] for _, zs in zones_with_data],
            marker_color=["#ef4444" if zs["overflow"] else "#3b82f6" for _, zs in zones_with_data],
        )
        fig.add_bar(
            name="Ventana disponible",
            x=z_labels,
            y=[zs["avail_min"] for _, zs in zones_with_data],
            marker_color="#d1d5db",
        )
        fig.update_layout(
            barmode="group", title="Tiempo de servicio total vs ventana disponible por zona",
            xaxis_title="Zona", yaxis_title="Minutos",
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=320, margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Issue type distribution ───────────────────────────────────────────────
    if findings["raw_issue_counts"]:
        st.markdown('<div class="section-title">🔍 Distribución de causas detectadas</div>', unsafe_allow_html=True)
        ic = findings["raw_issue_counts"]
        labels_map = {k: v[1] for k, v in ISSUE_LABELS.items()}
        labels = [labels_map.get(k, k) for k in ic.keys()]
        counts = list(ic.values())
        colors = ["#ef4444", "#f59e0b", "#3b82f6", "#6b7280", "#16a34a",
                  "#8b5cf6", "#06b6d4", "#ec4899", "#84cc16", "#f97316"]

        fig2 = go.Figure(go.Bar(
            x=counts, y=labels, orientation="h",
            marker_color=colors[:len(labels)],
            text=counts, textposition="outside",
        ))
        fig2.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis_title="Nodos afectados", yaxis_title="",
            height=max(200, len(labels) * 40 + 60),
            margin=dict(t=10, b=40, l=20, r=40),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Unattended node detail table ──────────────────────────────────────────
    st.markdown('<div class="section-title">📋 Detalle de nodos sin atender</div>', unsafe_allow_html=True)

    with st.expander("Ver tabla completa", expanded=False):
        rows = []
        for n in findings["unattended"]:
            primary = n["primary_type"]
            icon, label, _ = ISSUE_LABELS.get(primary, ("❓", primary, "badge-gray"))
            rows.append({
                "Ident":         n["ident"],
                "Dirección":     n["address"],
                "Causa router":  n["cause_code"],
                "Problema principal": f"{icon} {label}",
                "Zona(s)":       str(n["zones"]) if n["zones"] else "sin zona",
                "Duration":      n["duration"],
                "Ventana":       n["window"],
                "Load 1/2/3":    f"{n['load']:.1f} / {n['load_2']:.2f} / {n['load_3']:.1f}",
                "# Problemas":   len(n["issues"]),
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Per-node detail expandable ─────────────────────────────────────────────
    st.markdown('<div class="section-title">🔎 Diagnóstico por nodo</div>', unsafe_allow_html=True)

    for n in findings["unattended"]:
        primary = n["primary_type"]
        icon, label, badge_cls = ISSUE_LABELS.get(primary, ("❓", primary, "badge-gray"))
        header = f"{icon} **{n['ident']}** — {n['address'][:55]}{'…' if len(n['address'])>55 else ''}"

        with st.expander(header, expanded=False):
            col_a, col_b = st.columns([3, 2])
            with col_a:
                st.markdown(f"**Zona(s):** {n['zones'] or 'sin zona'}  \n"
                            f"**Causa router:** `{n['cause_code']}`  \n"
                            f"**Ventana:** `{n['window']}`  \n"
                            f"**Duration:** `{n['duration']} min`")
            with col_b:
                st.markdown(f"**Load 1:** {n['load']:.2f}  \n"
                            f"**Load 2:** {n['load_2']:.3f}  \n"
                            f"**Load 3:** {n['load_3']:.2f}")

            st.markdown("---")
            for i, iss in enumerate(n["issues"], 1):
                sev_color = {"high": "🔴", "medium": "🟡", "low": "⚪"}.get(iss["severity"], "⚪")
                i_icon, i_label, _ = ISSUE_LABELS.get(iss["type"], ("❓", iss["type"], "badge-gray"))
                st.markdown(
                    f"{sev_color} **{i_icon} {i_label}** — campo: `{iss['field']}` = `{iss['value']}`  \n"
                    f"&nbsp;&nbsp;&nbsp;{iss['detail']}  \n"
                    f"&nbsp;&nbsp;&nbsp;✏️ *{iss.get('fix', '')}*"
                )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:12px;color:#9ca3af;text-align:center;">'
        'SimpliRoute Route Analyzer · Basado en códigos de error E01/E02/E03/E99 de la documentación de validación de inputs'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
