"""
SimpliRoute — Route Analyzer v3
Arquitectura centralizada: Issue Registry como fuente única de verdad.
Mensajes con estructura: Causa → Por qué → Acción para el cliente.
"""

import streamlit as st
import json, re, math, io
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Optional

st.set_page_config(
    page_title="SR Route Analyzer",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: transparent; }
    .stMarkdown div, .stMarkdown p, .stMarkdown span { color: inherit !important; }
    [data-testid="stFileUploader"] section {
        background: transparent !important;
        border: 1.5px dashed rgba(128,128,128,0.4) !important;
        border-radius: 10px !important;
    }
    .badge {
        display: inline-block; padding: 2px 10px; border-radius: 20px;
        font-size: 11px; font-weight: 600; margin-right: 6px;
        border: 1px solid currentColor;
    }
    .badge-red    { background: rgba(220,38,38,0.12);  color: #ef4444; }
    .badge-amber  { background: rgba(245,158,11,0.12); color: #f59e0b; }
    .badge-blue   { background: rgba(59,130,246,0.12); color: #60a5fa; }
    .badge-gray   { background: rgba(107,114,128,0.12);color: #9ca3af; }
    .badge-green  { background: rgba(22,163,74,0.12);  color: #4ade80; }
    .issue-card {
        border-radius: 12px; padding: 16px 20px; margin-bottom: 10px;
        border-left: 4px solid; border-top: 1px solid rgba(128,128,128,0.2);
        border-right: 1px solid rgba(128,128,128,0.2);
        border-bottom: 1px solid rgba(128,128,128,0.2);
    }
    .issue-title  { font-size: 14px; font-weight: 700; margin-bottom: 6px; color: inherit; }
    .issue-why    { font-size: 13px; margin-bottom: 6px; color: inherit; line-height: 1.5; }
    .issue-action { font-size: 12px; font-style: italic; color: inherit; }
    .summary-card {
        border-radius: 12px; padding: 18px 22px; margin-bottom: 1.5rem;
        border: 1px solid rgba(128,128,128,0.25);
    }
    .summary-title { font-size: 18px; font-weight: 700; margin-bottom: 8px; color: inherit; }
    .summary-line  { font-size: 14px; margin-bottom: 4px; color: inherit; }
    .metric-card {
        border-radius: 12px; padding: 16px 18px;
        border: 1px solid rgba(128,128,128,0.25); margin-bottom: 0;
    }
    .metric-val { font-size: 26px; font-weight: 700; line-height: 1.1; }
    .metric-lbl { font-size: 12px; margin-top: 4px; color: inherit; }
    .section-title { font-size: 16px; font-weight: 600; margin: 1.5rem 0 .75rem; color: inherit; }
    .error-banner {
        background: rgba(220,38,38,0.1); border: 1.5px solid rgba(220,38,38,0.4);
        border-radius: 12px; padding: 20px 24px; margin-bottom: 1.5rem;
    }
    .error-title { font-size: 18px; font-weight: 700; color: #ef4444; margin-bottom: 6px; }
    hr.divider { border: none; border-top: 1px solid rgba(128,128,128,0.2); margin: 1.5rem 0; }
    code { background: rgba(128,128,128,0.15); padding: 1px 5px; border-radius: 4px; font-size: 12px; }
    .node-table { width:100%; border-collapse:collapse; margin-top:10px; }
    .node-table td { padding:3px 8px; font-size:12px; border-bottom:1px solid rgba(128,128,128,0.1); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ISSUE MODEL
# ─────────────────────────────────────────────────────────────────────────────

SEV_COLOR    = {"critical":"#ef4444","high":"#f59e0b","medium":"#3b82f6","low":"#6b7280"}
SEV_LABEL    = {"critical":"🔴 Crítico","high":"🟡 Alto","medium":"🔵 Medio","low":"⚪ Bajo"}
SEV_PRIORITY = {"critical":0,"high":1,"medium":2,"low":3}
BADGE_CLASS  = {"critical":"badge-red","high":"badge-amber","medium":"badge-blue","low":"badge-gray"}
SCOPE_ORDER  = {"preflight":0,"filtered":1,"unattended":2,"fleet":3}

@dataclass
class Issue:
    code:     str
    scope:    str
    severity: str
    title:    str
    why:      str
    action:   str
    affected: List[dict] = field(default_factory=list)
    value:    str = ""


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def parse_upload(f) -> dict:
    content = f.read().decode("utf-8", errors="replace").strip()
    try:
        return json.loads(content)
    except Exception:
        pass
    for pat in [r"--data-raw\s+'(.*)'$", r'--data-raw\s+"(.*)"$']:
        m = re.search(pat, content, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
    s, e = content.find("{"), content.rfind("}")
    if s != -1 and e != -1:
        try:
            return json.loads(content[s:e+1])
        except Exception:
            pass
    raise ValueError("No se pudo parsear el archivo. Asegúrate de que sea JSON o formato curl.")

def parse_time(t) -> Optional[int]:
    if not t:
        return None
    try:
        h, m = str(t).split(":")
        return int(h)*60 + int(m)
    except Exception:
        return None

def try_int(v) -> Optional[int]:
    try:
        return int(str(v))
    except Exception:
        return None

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin(math.radians(lat2-lat1)/2)**2 + \
        math.cos(p1)*math.cos(p2)*math.sin(math.radians(lon2-lon1)/2)**2
    return R*2*math.atan2(math.sqrt(a), math.sqrt(1-a))

def detect_outliers_iqr(values) -> set:
    arr = [v for v in values if v is not None and v > 0]
    if len(arr) < 4:
        return set()
    q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
    iqr = q3 - q1
    if iqr == 0:
        return set()
    return {v for v in arr if v > q3 + 3.0*iqr}

def time_overlap(ws1, we1, ws2, we2) -> bool:
    s1,e1 = parse_time(ws1), parse_time(we1)
    s2,e2 = parse_time(ws2), parse_time(we2)
    if None in (s1,e1,s2,e2):
        return True
    return s1 <= e2 and s2 <= e1

def is_error_response(res) -> bool:
    if "errors" in res and "vehicles" not in res:
        return True
    if "description" in res and "cause" in res and "vehicles" not in res:
        return True
    return False

def avg(vals) -> float:
    clean = [v for v in vals if v is not None]
    return round(float(np.mean(clean)), 1) if clean else 0.0

VALID_FMV = {1.0, 1.5, 2.0, 3.0}

ERROR_CODE_MESSAGES = {
    "E01004": ("Duración de descanso incompatible con el turno",
               "La ventana de descanso combinada con su duración no cabe dentro del turno del vehículo.",
               "Ajustar rest_time_start, rest_time_end o rest_time_duration para que quepan dentro del turno."),
    "E01005": ("Ventana de descanso fuera del turno del vehículo",
               "La ventana horaria de descanso no intersecta con el turno. El router no puede programar el descanso.",
               "Corregir rest_time_start y rest_time_end para que estén dentro del shift_start y shift_end."),
    "E03001": ("Carga total de nodos insuficiente para los vehículos",
               "La suma de carga de todos los nodos es menor que la carga mínima requerida por los vehículos.",
               "Revisar el campo min_load de los vehículos o agregar más nodos."),
    "E03002": ("Carga de nodos excede la capacidad de todos los vehículos",
               "Hay nodos cuya carga individual supera la capacidad de cualquier vehículo disponible.",
               "Aumentar la capacidad de los vehículos o dividir la carga de esos nodos."),
    "E03003": ("Ningún vehículo tiene las skills requeridas por los nodos",
               "Todos los nodos piden skills específicas y ningún vehículo las tiene asignadas.",
               "Asignar las skills faltantes a al menos un vehículo."),
    "E03004": ("Ningún vehículo cubre las zonas asignadas a los nodos",
               "Todos los nodos tienen zona asignada pero ningún vehículo cubre esas zonas (autoZone=false).",
               "Asignar las zonas correspondientes a los vehículos, o activar autoZone=true."),
    "E03005": ("Ningún vehículo tiene turno compatible con las ventanas de los nodos",
               "Para todas las visitas, no existe un vehículo con horario disponible.",
               "Revisar que los turnos de los vehículos se solapen con las ventanas de los nodos."),
    "E99001": ("Visita con más de una zona asignada",
               "Una o más visitas tienen más de una zona. Solo se permite una zona por visita.",
               "Corregir el campo zones de los nodos para que tenga máximo un valor."),
    "E99002": ("Configuraciones de optimización incompatibles",
               "Se están usando parámetros que no son compatibles entre sí.",
               "Revisar la combinación de parámetros en el request."),
}


# ─────────────────────────────────────────────────────────────────────────────
# DETECTION ENGINE — fuente única de verdad
# ─────────────────────────────────────────────────────────────────────────────

def build_explained_map(preflight_issues: List[Issue], nodes: list, all_veh_skills: set, vehicles_no_zone: list, all_veh_zones: set, autozone: bool) -> dict:
    """
    Mapea ident de nodo → Issue de pre-vuelo que lo explica.
    Evita que nodos sin atender caigan al fallback cuando ya existe una causa sistémica detectada.
    Agregar nuevas reglas aquí para extender la propagación.
    """
    explained = {}

    for issue in preflight_issues:
        code = issue.code

        # Skills opcionales faltantes → afecta nodos con esa skill
        if code == "W02102":
            try:
                skill = int(issue.value.replace("skill ","").strip())
                for n in nodes:
                    if skill in n.get("skills_optional",[]) and n["ident"] not in explained:
                        explained[n["ident"]] = issue
            except Exception:
                pass

        # min_load imposible → afecta todos los nodos (el vehículo nunca sale)
        elif code == "MIN_LOAD_IMPOSSIBLE":
            for n in nodes:
                if n["ident"] not in explained:
                    explained[n["ident"]] = issue

        # Nodos sin zona con vehículos zonados
        elif code == "NODES_NO_ZONE":
            for n in nodes:
                if not n.get("zones") and n["ident"] not in explained:
                    explained[n["ident"]] = issue

        # Zonas sin cobertura
        elif code == "E03004":
            for n in nodes:
                if n.get("zones") and n["ident"] not in explained:
                    explained[n["ident"]] = issue

        # Capacidad excedida
        elif code == "E03002":
            for a in issue.affected:
                if a["ident"] not in explained:
                    explained[a["ident"]] = issue

        # Turno sin solapamiento
        elif code == "E03005":
            for n in nodes:
                if n["ident"] not in explained:
                    explained[n["ident"]] = issue

    return explained


def collect_all_issues(req: dict, res: dict) -> List[Issue]:
    issues: List[Issue] = []
    nodes    = req.get("nodes", [])
    vehicles = req.get("vehicles", [])
    node_map = {n["ident"]: n for n in nodes}
    veh_map  = {v["ident"]: v for v in vehicles}
    unattended = res.get("unattendedClientsNodes", [])
    filtered   = res.get("filteredClientsNodes",   [])
    autozone   = req.get("autoZone", False)
    beauty_on  = req.get("beauty", True)

    # ── Build routed set + vehicle last state ─────────────────────────────────
    routed_ids       = set()
    routed_per_veh   = {}
    vehicle_last_state = {}

    for v_res in res.get("vehicles", []):
        vid, count = v_res["ident"], 0
        for tour in v_res.get("tours", []):
            real = [n for n in tour.get("nodes",[]) if not n["ident"].startswith("vehicle-")]
            for n in real:
                routed_ids.add(n["ident"])
                count += 1
            if real:
                last  = real[-1]
                v_req = veh_map.get(vid, {})
                se_m  = parse_time(v_req.get("shift_end","23:59"))
                dep_str = last.get("departure","")
                if dep_str:
                    try:
                        h, m, *_ = dep_str.split(":")
                        dep_min = int(h)*60+int(m)
                        rn = node_map.get(last["ident"],{})
                        if rn.get("lat") and se_m:
                            vehicle_last_state[vid] = {
                                "lat": rn["lat"], "lon": rn["lon"],
                                "dep_min": dep_min, "shift_end": se_m,
                                "remaining": se_m - dep_min,
                            }
                    except Exception:
                        pass
        routed_per_veh[vid] = count

    # ── Zone / skill maps ─────────────────────────────────────────────────────
    zone_to_vehicles = defaultdict(list)
    all_veh_zones    = set()
    all_veh_skills   = set()
    vehicles_no_zone = []

    for v in vehicles:
        vid = v["ident"]
        for z in v.get("zones",[]):
            zone_to_vehicles[z].append(vid)
            all_veh_zones.add(z)
        all_veh_skills.update(v.get("skills",[]))
        if not v.get("zones"):
            vehicles_no_zone.append(vid)

    # ── Max visit stats ───────────────────────────────────────────────────────
    max_visit_global = None
    vehicles_at_max  = []
    for v in vehicles:
        mv = v.get("max_visit")
        if mv is not None:
            max_visit_global = mv
            if routed_per_veh.get(v["ident"],0) >= mv:
                vehicles_at_max.append(v["ident"])

    # ─── 1. PRE-FLIGHT ────────────────────────────────────────────────────────

    fmv = req.get("fmv")
    if fmv is not None and fmv not in VALID_FMV:
        issues.append(Issue("E02005","preflight","critical",
            f"Valor fmv={fmv} no permitido — puede causar error E500",
            f"fmv debe ser 1.0 (bajo), 1.5 (medio), 2.0 (alto) o 3.0 (intenso). El valor {fmv} no es válido.",
            "Cambiar fmv a uno de los valores válidos: 1.0, 1.5, 2.0 o 3.0."))

    zero_win, inverted, narrow, zero_coords = [], [], [], []
    for n in nodes:
        ident = n.get("ident","?")
        ws, we = n.get("window_start","00:00"), n.get("window_end","23:59")
        dur    = try_int(n.get("duration"))
        lat, lon = n.get("lat"), n.get("lon")
        ws_m, we_m = parse_time(ws), parse_time(we)
        addr = n.get("address","")[:60]
        extra_node = {"ident":ident,"address":addr,"extra":""}

        if lat == 0 and lon == 0:
            zero_coords.append({**extra_node,"extra":"lat=0, lon=0"})
        if ws == we and dur and dur > 0:
            zero_win.append({**extra_node,"extra":f"ventana:{ws}–{we} | dur:{dur}min"})
        elif ws_m and we_m:
            if ws_m > we_m:
                inverted.append({**extra_node,"extra":f"{ws} > {we}"})
            elif dur and 0 < (we_m-ws_m) < dur:
                narrow.append({**extra_node,"extra":f"ventana:{we_m-ws_m}min < dur:{dur}min"})

    if zero_coords:
        issues.append(Issue("E02004","preflight","critical",
            f"Coordenadas (0,0) en {len(zero_coords)} nodo(s) — el router no puede ubicarlos",
            "Nodos con lat=0 y lon=0 son inválidos. El router los rechaza con error E02004.",
            "Corregir las coordenadas lat/lon antes de enviar el request.",
            zero_coords))
    if zero_win:
        issues.append(Issue("ZERO_WINDOW","preflight","high",
            f"Ventana de 0 minutos en {len(zero_win)} nodo(s) — imposible atenderlos",
            "window_start == window_end con duration > 0. No hay margen temporal para el servicio.",
            "Ampliar window_end o establecer duration=0 en estos nodos.",
            zero_win))
    if inverted:
        issues.append(Issue("E01006","preflight","high",
            f"Ventana horaria invertida en {len(inverted)} nodo(s) — window_start > window_end",
            "El inicio de la ventana es posterior al fin. El router las rechaza con error E01006.",
            "Intercambiar window_start y window_end en los nodos afectados.",
            inverted))
    if narrow:
        issues.append(Issue("NARROW_WINDOW","preflight","high",
            f"Ventana más pequeña que la duración de servicio en {len(narrow)} nodo(s)",
            "La ventana disponible es menor al tiempo necesario para completar la visita.",
            "Ampliar window_end o reducir duration en los nodos afectados.",
            narrow))

    for v in vehicles:
        ss = parse_time(v.get("shift_start","00:01"))
        se = parse_time(v.get("shift_end","23:59"))
        if ss is not None and se is not None and ss >= se:
            issues.append(Issue("E01006_VEH","preflight","high",
                f"Turno inválido en vehículo {v['ident']} — shift_start >= shift_end",
                f"El turno {v.get('shift_start')} – {v.get('shift_end')} no tiene duración. El vehículo nunca puede salir.",
                "Corregir shift_start y shift_end para que el turno tenga duración positiva."))

    if req.get("enable_rest_time"):
        no_ix, exceeds = [], []
        for v in vehicles:
            vid = v.get("ident","?")
            ss  = parse_time(v.get("shift_start","00:01"))
            se  = parse_time(v.get("shift_end","23:59"))
            rs  = parse_time(v.get("rest_time_start"))
            re  = parse_time(v.get("rest_time_end"))
            if None in (ss,se,rs,re):
                continue
            extra = f"shift {v.get('shift_start')}–{v.get('shift_end')} | rest {v.get('rest_time_start')}–{v.get('rest_time_end')}"
            if rs >= se or re <= ss:
                no_ix.append({"ident":vid,"address":extra,"extra":""})
            elif re > se:
                exceeds.append({"ident":vid,"address":extra,"extra":f"excede {re-se} min"})
        if no_ix:
            issues.append(Issue("E01005","preflight","critical",
                f"Ventana de descanso fuera del turno en {len(no_ix)} vehículo(s)",
                "La ventana de descanso no intersecta con el turno. El router no puede programar el descanso (E01005).",
                "Corregir rest_time_start y rest_time_end para que estén dentro del shift del vehículo.",
                no_ix))
        if exceeds:
            issues.append(Issue("E01004","preflight","high",
                f"Ventana de descanso se extiende más allá del turno en {len(exceeds)} vehículo(s)",
                "rest_time_end supera shift_end. El descanso debe caber dentro del turno (E01004).",
                "Ajustar rest_time_end para que no supere shift_end.",
                exceeds))

    total_load   = sum(n.get("load",0)   or 0 for n in nodes)
    total_load_2 = sum(n.get("load_2",0) or 0 for n in nodes)
    total_load_3 = sum(n.get("load_3",0) or 0 for n in nodes)
    for v in vehicles:
        vid, blocked = v.get("ident","?"), []
        ml1 = v.get("min_load",  0) or 0
        ml2 = v.get("min_load_2",0) or 0
        ml3 = v.get("min_load_3",0) or 0
        if ml1>0 and total_load   < ml1: blocked.append(f"load total={total_load:.1f} < min_load={ml1}")
        if ml2>0 and total_load_2 < ml2: blocked.append(f"load_2 total={total_load_2:.1f} < min_load_2={ml2}")
        if ml3>0 and total_load_3 < ml3: blocked.append(f"load_3 total={total_load_3:.1f} < min_load_3={ml3}")
        if blocked:
            issues.append(Issue("MIN_LOAD_IMPOSSIBLE","preflight","critical",
                f"Vehículo {vid} nunca puede cumplir su carga mínima — quedará inactivo",
                f"La carga total disponible en los nodos no alcanza: {' | '.join(blocked)}.",
                "Reducir min_load/min_load_2/min_load_3, o agregar nodos con carga suficiente."))

    missing_opt = defaultdict(list)
    for n in nodes:
        for s in n.get("skills_optional",[]):
            if s not in all_veh_skills:
                missing_opt[s].append({"ident":n.get("ident","?"),"address":n.get("address","")[:55],"extra":f"skill opt: {s}"})
    for skill, aff in missing_opt.items():
        vj = req.get("visit_joiner",{})
        note = " Con visit_joiner activo y match_skills=true, serán filtrados antes de optimizar (W02102)." if vj.get("enable_visit_join") and vj.get("match_skills") else ""
        issues.append(Issue("W02102","preflight","high",
            f"Skill opcional {skill} sin vehículo asignado — {len(aff)} nodo(s) en riesgo de ser filtrados",
            f"Ningún vehículo tiene la skill {skill}.{note}",
            f"Agregar la skill {skill} a al menos un vehículo, o desactivar match_skills en visit_joiner.",
            aff))

    if not autozone:
        nwz = [n for n in nodes if n.get("zones")]
        if nwz and all_veh_zones:
            uncovered = set()
            for n in nwz:
                for z in n.get("zones",[]):
                    if z not in all_veh_zones:
                        uncovered.add(z)
            if uncovered:
                cnt = sum(1 for n in nwz if any(z in uncovered for z in n.get("zones",[])))
                issues.append(Issue("E03004","preflight","critical",
                    f"Zona(s) {sorted(uncovered)} sin vehículo — {cnt} nodo(s) serán ignorados",
                    f"Los nodos tienen zonas {sorted(uncovered)} pero ningún vehículo las cubre. Con autoZone=false el router los ignora.",
                    "Asignar las zonas a al menos un vehículo, o activar autoZone=true."))
        elif nwz and not all_veh_zones:
            issues.append(Issue("E03004","preflight","critical",
                "Todos los nodos tienen zona pero ningún vehículo tiene zona asignada",
                f"{len(nwz)} nodos con zona pero todos los vehículos tienen zones=[]. El router no generará rutas (E03004).",
                "Asignar las zonas correspondientes a los vehículos, o activar autoZone=true."))

        nnozone = [n for n in nodes if not n.get("zones")]
        vnozone_req = [v for v in vehicles if not v.get("zones")]
        if nnozone and all_veh_zones and not vnozone_req:
            issues.append(Issue("NODES_NO_ZONE","preflight","high",
                f"{len(nnozone)} nodo(s) sin zona — todos los vehículos tienen zona, estos quedarán rezagados",
                "Con autoZone=false, el router prioriza nodos de zona. Los sin zona solo se atienden si sobra capacidad.",
                "Asignar la zona correcta a los nodos, o agregar un vehículo con zones=[].",
                [{"ident":n.get("ident","?"),"address":n.get("address","")[:55],"extra":""} for n in nnozone[:15]]))

    max_cap1 = max((v.get("capacity",0) or 0 for v in vehicles), default=0)
    if max_cap1 < 1e15:
        overload = [{"ident":n.get("ident","?"),"address":n.get("address","")[:55],
                     "extra":f"load={n.get('load',0):.1f} > cap_max={max_cap1}"}
                    for n in nodes if n.get("load",0) > max_cap1]
        if overload:
            issues.append(Issue("E03002","preflight","critical",
                f"Carga de {len(overload)} nodo(s) excede la capacidad de todos los vehículos",
                f"La carga individual supera la capacidad del vehículo más grande (cap={max_cap1}). Ningún vehículo puede atenderlos.",
                "Aumentar la capacidad de los vehículos o dividir la carga de estos nodos.",
                overload))

    if not req.get("longRoutes"):
        shifts = [(parse_time(v.get("shift_start","00:01")), parse_time(v.get("shift_end","23:59"))) for v in vehicles]
        shifts = [(s,e) for s,e in shifts if s and e]
        no_ov  = [{"ident":n.get("ident","?"),"address":n.get("address","")[:55],
                   "extra":f"ventana {n.get('window_start')}–{n.get('window_end')}"}
                  for n in nodes
                  if (ws_m:=parse_time(n.get("window_start","00:00"))) is not None
                  and (we_m:=parse_time(n.get("window_end","23:59"))) is not None
                  and not any(ws_m<=se and ss<=we_m for ss,se in shifts)]
        if no_ov and len(no_ov)==len(nodes):
            issues.append(Issue("E03005","preflight","critical",
                "Ningún vehículo tiene turno compatible con las ventanas de los nodos",
                "Las ventanas de todos los nodos están fuera del turno de todos los vehículos. No se generarán rutas (E03005).",
                "Revisar que los turnos de los vehículos se solapen con las ventanas de los nodos.",
                no_ov[:10]))

    # ─── 2. FILTERED ─────────────────────────────────────────────────────────
    w02102_f, w00001_f, other_f = [], [], []
    for fn in filtered:
        ident    = fn["ident"]
        req_node = node_map.get(ident,{})
        cause    = fn.get("cause",{})
        codes    = cause.get("codes",[]) if isinstance(cause,dict) else []
        addr     = req_node.get("address","")[:60]
        if "W02102" in codes:
            w02102_f.append({"ident":ident,"address":addr,"extra":f"skill opt: {req_node.get('skills_optional',[])}"})
        elif "W00001" in codes:
            nlat, nlon = fn.get("lat"), fn.get("lon")
            mn = None
            for v in vehicles:
                vl = v.get("location_start",{})
                if vl.get("lat") and nlat and nlon:
                    d = haversine_km(vl["lat"],vl["lon"],nlat,nlon)
                    if mn is None or d < mn: mn = d
            w00001_f.append({"ident":ident,"address":addr,"extra":f"veh más cercano ~{mn:.0f}km" if mn else ""})
        else:
            other_f.append({"ident":ident,"address":addr,"extra":str(codes)})

    if w02102_f:
        issues.append(Issue("W02102_FILTERED","filtered","high",
            f"{len(w02102_f)} nodo(s) filtrados — skill opcional sin vehículo asignado (W02102)",
            "El router los eliminó antes de optimizar porque ningún vehículo tiene la skill opcional y match_skills=true.",
            "Agregar la skill faltante a un vehículo, o desactivar match_skills en visit_joiner.",
            w02102_f))
    if w00001_f:
        issues.append(Issue("W00001_FILTERED","filtered","high",
            f"{len(w00001_f)} nodo(s) filtrados — incompatibilidad geográfica (W00001)",
            "Los vehículos están demasiado lejos para alcanzar estos nodos dentro del turno disponible.",
            "Verificar que el punto de inicio de los vehículos esté en la misma zona que las visitas.",
            w00001_f))
    # FILTERED_OTHER eliminado — no aporta valor cuando la causa ya está en pre-vuelo

    # ─── 3. UNATTENDED — detección multi-causa por nodo ─────────────────────
    pf_issues_so_far = [i for i in issues if i.scope == "preflight"]
    zone_nodes = defaultdict(list)
    for n in nodes:
        z_list = n.get("zones",[])
        zone_nodes[(z_list[0] if z_list else "sin_zona")].append(n)

    zone_overflow = set()
    zone_outliers = {}
    for z, znodes in zone_nodes.items():
        durs = [d for d in [try_int(n.get("duration")) for n in znodes] if d is not None]
        zone_outliers[z] = detect_outliers_iqr(durs)
        veh_ids = zone_to_vehicles.get(z, vehicles_no_zone if z=="sin_zona" else [])
        if veh_ids:
            v   = veh_map.get(veh_ids[0],{})
            ssm = parse_time(v.get("shift_start","00:01"))
            sem = parse_time(v.get("shift_end","23:59"))
            if ssm and sem and sum(durs) > (sem-ssm)*len(veh_ids):
                zone_overflow.add(z)

    # ctx shared across nodes
    ctx = {
        "veh_map": veh_map, "zone_to_vehicles": zone_to_vehicles,
        "zone_overflow": zone_overflow, "zone_outliers": zone_outliers,
        "vehicles_no_zone": vehicles_no_zone, "all_veh_zones": all_veh_zones,
        "autozone": autozone, "beauty_on": beauty_on,
        "max_visit_global": max_visit_global, "vehicles_at_max": vehicles_at_max,
        "vehicle_last_state": vehicle_last_state,
    }

    CAUSE_DEFS = {
        "zero_window":    ("critical","Ventana de 0 minutos — imposible agendar la visita",
            "window_start == window_end con duration > 0. No hay espacio temporal para el servicio.",
            "Ampliar window_end o establecer duration=0."),
        "narrow_window":  ("high","Ventana más pequeña que la duración — el servicio no cabe",
            "La ventana disponible es menor al tiempo de servicio.",
            "Ampliar window_end o reducir duration."),
        "tight_window":   ("high","Ventana igual a la duración — sin margen para el traslado",
            "La ventana cubre exactamente el servicio pero no deja tiempo para llegar y salir.",
            "Ampliar window_end al menos 15–30 min adicionales."),
        "duration_outlier":("high","Duración atípica — desborda el tiempo disponible de la zona",
            "La duration es estadísticamente anormal y causa desborde temporal en la zona.",
            "Verificar y corregir la duration al tiempo real de servicio."),
        "zone_overflow":  ("high","Tiempo total de servicio supera el turno en la zona",
            "La suma de durations supera el tiempo total de los vehículos asignados.",
            "Corregir durations, agregar vehículo a la zona, o ampliar el turno."),
        "window_shift_mismatch": ("high","Ventana incompatible con el turno del vehículo",
            "La ventana del nodo no se solapa con el turno de ningún vehículo candidato.",
            "Ajustar window_start/window_end o el shift del vehículo."),
        "inactive_vehicles_no_zone": ("high",
            "Vehículo(s) sin zona no pueden atender nodos con zona asignada",
            f"Con autoZone=false, los {len(vehicles_no_zone)} vehículo(s) con zones=[] no pueden atender este nodo.",
            "Asignar las zonas a los vehículos inactivos."),
        "nodes_no_zone":  ("medium","Nodo sin zona — atendido solo si sobra capacidad",
            "Con autoZone=false, el router prioriza nodos con zona. Este compite por el sobrante.",
            "Asignar la zona correcta al nodo, o agregar un vehículo con zones=[]."),
        "max_visit_limit":("medium",
            f"Límite max_visit={max_visit_global} — vehículos sin cupo",
            f"{len(vehicles_at_max)} vehículo(s) llegaron al tope de {max_visit_global} visitas.",
            f"Aumentar max_visit o redistribuir carga."),
        "shift_exhausted":("high","Turno agotado — el vehículo no alcanza a llegar",
            "El vehículo más cercano completó su ruta sin tiempo para llegar, atender y salir.",
            "Ampliar el turno o asignar un vehículo adicional en este sector."),
        "exc_so_002":     ("low" if beauty_on else "medium",
            "Descartado por beauty=true — rutas más agrupadas priorizadas" if beauty_on
            else "Descartado — incluirlo empeoraría la solución global",
            "Con beauty=true el router excluye nodos que generarían cruces." if beauty_on
            else "El nodo tiene tiempo y capacidad pero afectaría la calidad de otras rutas.",
            "Probar con beauty=false." if beauty_on else "Agregar un vehículo o revisar la distribución."),
        "exc_so_001":     ("high","Excluido antes de optimizar — ventana incompatible (EXC_SO-001)",
            "Ningún vehículo puede alcanzar este nodo dentro de su ventana horaria.",
            "Ampliar la ventana o revisar el turno de los vehículos."),
        "cap_time_general":("medium","Sin causa específica identificada",
            "El nodo quedó fuera por una combinación de restricciones de tiempo o capacidad.",
            "Revisar carga, duración y ventanas. Considerar agregar un vehículo."),
    }

    def get_node_primary_cause(ident, req_node, un_node, pf_issues, ctx):
        nz     = req_node.get("zones",[])
        dur    = try_int(req_node.get("duration"))
        ws, we = req_node.get("window_start","00:00"), req_node.get("window_end","23:59")
        ws_m, we_m = parse_time(ws), parse_time(we)
        cause_code = un_node.get("cause",{}).get("code","")
        candidates = []  # (priority, key, pf_or_None, sev, title, why, action)

        # Pre-flight causes — solo si aplican específicamente a ESTE nodo
        for pf in pf_issues:
            if pf.code == "W02102":
                try:
                    skill = int(pf.value.replace("skill","").strip())
                    if skill in req_node.get("skills_optional",[]):
                        candidates.append((0,"pf_w02102",pf,pf.severity,pf.title,pf.why,pf.action))
                except Exception:
                    pass
            elif pf.code == "MIN_LOAD_IMPOSSIBLE":
                vid = pf.value.replace("veh","").strip()
                v   = ctx["veh_map"].get(vid,{})
                vz  = set(v.get("zones",[]))
                if ctx["autozone"] or not set(nz) or (set(nz) & vz):
                    candidates.append((0,"pf_min_load",pf,pf.severity,pf.title,pf.why,pf.action))
            elif pf.code == "E03004":
                try:
                    import ast as _ast
                    uncovered = set(_ast.literal_eval(pf.value))
                    if any(z in uncovered for z in nz):
                        candidates.append((0,"pf_e03004",pf,pf.severity,pf.title,pf.why,pf.action))
                except Exception:
                    pass
            elif pf.code == "NODES_NO_ZONE" and not nz and not ctx["autozone"]:
                candidates.append((0,"pf_nodes_no_zone",pf,pf.severity,pf.title,pf.why,pf.action))
            elif pf.code == "E03005":
                candidates.append((0,"pf_e03005",pf,pf.severity,pf.title,pf.why,pf.action))

        # Causas específicas del nodo
        if ws == we and dur and dur > 0:
            d = CAUSE_DEFS["zero_window"]
            candidates.append((1,"zero_window",None,d[0],d[1],d[2],d[3]))
        elif ws_m and we_m and dur:
            win = we_m - ws_m
            if 0 < win < dur:
                d = CAUSE_DEFS["narrow_window"]
                candidates.append((1,"narrow_window",None,d[0],d[1],d[2],d[3]))
            elif win == dur:
                d = CAUSE_DEFS["tight_window"]
                candidates.append((1,"tight_window",None,d[0],d[1],d[2],d[3]))

        if dur and any(dur in ctx["zone_outliers"].get(z,set()) for z in (nz if nz else ["sin_zona"])):
            d = CAUSE_DEFS["duration_outlier"]
            candidates.append((2,"duration_outlier",None,d[0],d[1],d[2],d[3]))

        if any(z in ctx["zone_overflow"] for z in (nz if nz else ["sin_zona"])):
            d = CAUSE_DEFS["zone_overflow"]
            candidates.append((2,"zone_overflow",None,d[0],d[1],d[2],d[3]))

        if nz and not any(
            time_overlap(ws,we,ctx["veh_map"][v].get("shift_start","00:01"),
                         ctx["veh_map"][v].get("shift_end","23:59"))
            for z in nz for v in ctx["zone_to_vehicles"].get(z,[]) if v in ctx["veh_map"]
        ):
            d = CAUSE_DEFS["window_shift_mismatch"]
            candidates.append((2,"window_shift_mismatch",None,d[0],d[1],d[2],d[3]))

        if nz and not ctx["autozone"] and ctx["vehicles_no_zone"] and ctx["all_veh_zones"]:
            d = CAUSE_DEFS["inactive_vehicles_no_zone"]
            candidates.append((3,"inactive_vehicles_no_zone",None,d[0],d[1],d[2],d[3]))

        if not nz and not ctx["autozone"] and ctx["all_veh_zones"] and not ctx["vehicles_no_zone"]:
            d = CAUSE_DEFS["nodes_no_zone"]
            candidates.append((3,"nodes_no_zone",None,d[0],d[1],d[2],d[3]))

        if ctx["max_visit_global"] and ctx["vehicles_at_max"]:
            d = CAUSE_DEFS["max_visit_limit"]
            candidates.append((3,"max_visit_limit",None,d[0],d[1],d[2],d[3]))

        nlat, nlon = req_node.get("lat"), req_node.get("lon")
        if nlat and nlon and ctx["vehicle_last_state"]:
            for vs in ctx["vehicle_last_state"].values():
                if (haversine_km(vs["lat"],vs["lon"],nlat,nlon)/60*60)+(dur or 0) > vs["remaining"]:
                    d = CAUSE_DEFS["shift_exhausted"]
                    candidates.append((3,"shift_exhausted",None,d[0],d[1],d[2],d[3]))
                    break

        if not candidates:
            key = "exc_so_002" if cause_code=="EXC_SO-002" else "exc_so_001" if cause_code=="EXC_SO-001" else "cap_time_general"
            d = CAUSE_DEFS[key]
            candidates.append((5,key,None,d[0],d[1],d[2],d[3]))

        candidates.sort(key=lambda x: x[0])
        return candidates[0]  # (priority, key, pf, sev, title, why, action)

    # Agrupar nodos por causa primaria
    grouped = defaultdict(lambda: {"items":[], "sev":"low", "title":"", "why":"", "action":""})
    for un in unattended:
        ident    = un["ident"]
        req_node = node_map.get(ident,{})
        addr     = req_node.get("address","")[:60]
        dur      = try_int(req_node.get("duration"))
        ws       = req_node.get("window_start","00:00")
        we       = req_node.get("window_end","23:59")
        info     = {"ident":ident,"address":addr,
                    "extra":f"load={un.get('load',0):.1f} | dur={dur} | {ws}–{we}"}
        _, key, _, sev, title, why, action = get_node_primary_cause(ident, req_node, un, pf_issues_so_far, ctx)
        g = grouped[key]
        g["items"].append(info)
        g["sev"]    = sev
        g["title"]  = title
        g["why"]    = why
        g["action"] = action

    for key, g in grouped.items():
        issues.append(Issue(
            code     = key.upper().replace("PF_",""),
            scope    = "unattended",
            severity = g["sev"],
            title    = f"{g['title']} — {len(g['items'])} nodo(s)",
            why      = g["why"],
            action   = g["action"],
            affected = g["items"],
        ))

    # ─── 4. FLEET — solo si no están ya explicados por pre-vuelo ────────────
    pf_codes = {i.code for i in issues if i.scope == "preflight"}
    active_cnt   = {vid:cnt for vid,cnt in routed_per_veh.items() if cnt>0}
    idle_vehs    = [vid for vid,cnt in routed_per_veh.items() if cnt==0]
    visits_list  = list(active_cnt.values())
    single_ratio = sum(1 for v in visits_list if v==1)/max(len(visits_list),1) if visits_list else 0

    # FLEET_SINGLE_VISIT: solo si no hay causa sistémica que lo explique
    systemic_causes = {"MIN_LOAD_IMPOSSIBLE","W02102","E03004","NODES_NO_ZONE","E03005"}
    if single_ratio > 0.8 and visits_list and not (pf_codes & systemic_causes):
        issues.append(Issue("FLEET_SINGLE_VISIT","fleet","high",
            f"{int(single_ratio*100)}% de los vehículos activos hicieron exactamente 1 visita",
            "Cada vehículo completa una visita y no puede encadenar más. Las ventanas pueden ser demasiado estrechas o el punto de inicio está lejos.",
            "Ampliar las ventanas de tiempo o revisar las ubicaciones de inicio de los vehículos."))

    # FLEET_IDLE: solo si no está ya explicado por pre-vuelo
    if idle_vehs and len(idle_vehs) > len(routed_per_veh)*0.2 and not (pf_codes & systemic_causes):
        issues.append(Issue("FLEET_IDLE","fleet","high",
            f"{len(idle_vehs)} vehículo(s) ({len(idle_vehs)*100//max(len(routed_per_veh),1)}%) sin ninguna visita asignada",
            "Estos vehículos no participaron en ninguna ruta. Puede ser por zonas incompatibles, min_load imposible o turno incompatible.",
            "Revisar zonas, min_load y turno de los vehículos inactivos.",
            [{"ident":vid,"address":"","extra":""} for vid in idle_vehs]))

    issues.sort(key=lambda i: (SEV_PRIORITY.get(i.severity,4), SCOPE_ORDER.get(i.scope,5)))
    return issues


# ─────────────────────────────────────────────────────────────────────────────
# RENDERING
# ─────────────────────────────────────────────────────────────────────────────

def render_issue_card(issue: Issue, idx: Optional[int]=None):
    color  = SEV_COLOR.get(issue.severity,"#6b7280")
    badge  = f'<span class="badge {BADGE_CLASS.get(issue.severity,"badge-gray")}">{issue.code}</span>'
    prefix = f"{idx}. " if idx else ""
    count  = f'<span style="float:right;font-size:11px;opacity:0.55">{len(issue.affected)} nodo(s)</span>' if issue.affected else ""

    rows = ""
    for a in issue.affected[:12]:
        rows += f"<tr><td style='font-family:monospace;font-size:11px;padding:3px 8px;width:200px'>{a.get('ident','')[:36]}</td><td style='font-size:11px;padding:3px 8px'>{a.get('address','')}</td><td style='font-size:11px;padding:3px 8px;opacity:0.55'>{a.get('extra','')}</td></tr>"
    if len(issue.affected)>12:
        rows += f"<tr><td colspan='3' style='font-size:11px;padding:4px 8px;opacity:0.4'>… y {len(issue.affected)-12} más</td></tr>"
    node_html = f"<table class='node-table'>{rows}</table>" if rows else ""

    st.markdown(f"""
    <div class="issue-card" style="border-left-color:{color}">
        <div style="margin-bottom:6px">{badge}{count}</div>
        <div class="issue-title">{prefix}{issue.title}</div>
        <div class="issue-why">{issue.why}</div>
        <div class="issue-action">→ {issue.action}</div>
        {node_html}
    </div>""", unsafe_allow_html=True)


def render_metric(label, value, color="inherit", suffix=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val" style="color:{color}">{value}{suffix}</div>
        <div class="metric-lbl">{label}</div>
    </div>""", unsafe_allow_html=True)


def render_executive_summary(issues, routed, total, n_unat, n_filt):
    critical = [i for i in issues if i.severity=="critical"]
    pct      = round(routed/max(total,1)*100,1)
    if not issues:
        st.markdown("""<div class="summary-card" style="border-color:rgba(22,163,74,0.4);background:rgba(22,163,74,0.08)">
            <div class="summary-title">✅ Sin problemas detectados</div>
            <div class="summary-line">Todos los nodos fueron enrutados correctamente.</div>
        </div>""", unsafe_allow_html=True)
        return
    bg  = "rgba(220,38,38,0.08)"  if critical else "rgba(245,158,11,0.08)"
    brd = "rgba(220,38,38,0.4)"   if critical else "rgba(245,158,11,0.4)"
    ico = "🔴" if critical else "🟡"
    main_issue = issues[0]
    extras = []
    if len(issues)>1:
        extras.append(f'<div class="summary-line">📋 {len(issues)-1} problema(s) adicional(es) — ver detalle abajo.</div>')
    if n_unat or n_filt:
        extras.append(f'<div class="summary-line">📊 Cobertura actual: <strong>{pct}%</strong> ({routed}/{total} nodos) · {n_unat} sin atender · {n_filt} filtrados.</div>')
    st.markdown(f"""<div class="summary-card" style="border-color:{brd};background:{bg}">
        <div class="summary-title">{ico} {len(issues)} problema(s) detectado(s)</div>
        <div class="summary-line">📌 <strong>Principal:</strong> {main_issue.title}</div>
        {"".join(extras)}
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.markdown("# 🗺️ SimpliRoute — Route Analyzer")
    st.caption("Validación pre-vuelo · Nodos filtrados · Nodos sin atender · Flota")
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**📥 Cargar Request (JSON o cURL)**")
        req_file = st.file_uploader("request", type=["json","txt"], key="req", label_visibility="collapsed")
    with c2:
        st.markdown("**📤 Cargar Response (JSON o cURL)**")
        res_file = st.file_uploader("response", type=["json","txt"], key="res", label_visibility="collapsed")

    if not req_file or not res_file:
        st.info("Carga ambos archivos para iniciar el análisis.")
        return

    try:
        req = parse_upload(req_file)
        res = parse_upload(res_file)
    except Exception as e:
        st.error(f"Error al parsear los archivos: {e}")
        return

    # ── Error response ────────────────────────────────────────────────────────
    if is_error_response(res):
        err_code, err_desc = "", ""
        if "errors" in res:
            for err in res.get("errors",[]):
                err_code = err.get("code","")
                err_desc = err.get("message","")
        elif "cause" in res:
            err_code = res.get("cause",{}).get("code","")
            err_desc = res.get("description","")
        id_veh = res.get("cause",{}).get("id_vehicle","") if "cause" in res else ""
        veh_str = f" (Vehículo: {id_veh})" if id_veh else ""
        friendly = ERROR_CODE_MESSAGES.get(err_code)
        if friendly:
            t, w, a = friendly
            st.markdown(f"""<div class="error-banner">
                <div class="error-title">⛔ {err_code} — {t}{veh_str}</div>
                <div style="font-size:14px;margin-top:6px">{w}</div>
                <div style="font-size:12px;margin-top:8px;font-style:italic">→ {a}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="error-banner">
                <div class="error-title">⛔ Error del Router — {err_code}{veh_str}</div>
                <div style="font-size:14px;margin-top:4px">{err_desc}</div>
            </div>""", unsafe_allow_html=True)

        pf = [i for i in collect_all_issues(req, res) if i.scope=="preflight"]
        if pf:
            st.markdown('<div class="section-title">🔍 Problemas detectados en el Request</div>', unsafe_allow_html=True)
            for idx, issue in enumerate(pf, 1):
                render_issue_card(issue, idx)
        return

    # ── Collect ───────────────────────────────────────────────────────────────
    node_map   = {n["ident"]: n for n in req.get("nodes",[])}
    unattended = res.get("unattendedClientsNodes",[])
    filtered   = res.get("filteredClientsNodes",[])
    veh_used   = res.get("num_vehicles_used",0)
    total_veh  = len(req.get("vehicles",[]))
    routed_ids = set()
    for v in res.get("vehicles",[]):
        for tour in v.get("tours",[]):
            for n in tour.get("nodes",[]):
                if not n["ident"].startswith("vehicle-"):
                    routed_ids.add(n["ident"])
    total  = len(req.get("nodes",[]))
    routed = len(routed_ids)
    n_unat = len(unattended)
    n_filt = len(filtered)
    pct    = round(routed/max(total,1)*100,1)

    with st.spinner("Analizando..."):
        all_issues = collect_all_issues(req, res)

    pf_issues   = [i for i in all_issues if i.scope=="preflight"]
    flt_issues  = [i for i in all_issues if i.scope=="filtered"]
    un_issues   = [i for i in all_issues if i.scope=="unattended"]
    flt_issues2 = [i for i in all_issues if i.scope=="fleet"]

    render_executive_summary(all_issues, routed, total, n_unat, n_filt)

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: render_metric("Total nodos",   total)
    with c2: render_metric("Enrutados",     routed, "#16a34a")
    with c3: render_metric("Sin atender",   n_unat, "#ef4444" if n_unat else "inherit")
    with c4: render_metric("Filtrados",     n_filt, "#f59e0b" if n_filt else "inherit")
    with c5: render_metric("Tasa atención", pct,    "#1d4ed8", "%")
    with c6: render_metric("Vehículos",     f"{veh_used}/{total_veh}")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Pre-flight ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🔍 Validación pre-vuelo del Request</div>', unsafe_allow_html=True)
    if pf_issues:
        crit = sum(1 for i in pf_issues if i.severity=="critical")
        if crit:
            st.markdown(f'<div style="font-size:13px;color:#ef4444;font-weight:600;margin-bottom:8px">⛔ {crit} problema(s) crítico(s) detectado(s)</div>', unsafe_allow_html=True)
        for idx, issue in enumerate(pf_issues, 1):
            render_issue_card(issue, idx)
    else:
        st.markdown('<div style="background:rgba(22,163,74,0.1);border:1px solid rgba(22,163,74,0.3);border-radius:10px;padding:12px 16px;font-size:13px;font-weight:500">✅ Sin problemas detectados en el request.</div>', unsafe_allow_html=True)

    if not unattended and not filtered:
        st.success("✅ Todos los nodos fueron enrutados correctamente.")
        return

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Filtered ──────────────────────────────────────────────────────────────
    if filtered:
        st.markdown(f'<div class="section-title">🚫 Nodos descartados antes de optimizar ({n_filt})</div>', unsafe_allow_html=True)
        for issue in flt_issues:
            render_issue_card(issue)
        rows = []
        for fn in filtered:
            ident    = fn["ident"]
            req_node = node_map.get(ident,{})
            cause    = fn.get("cause",{})
            codes    = cause.get("codes",[]) if isinstance(cause,dict) else []
            rows.append({"Ident":ident,"Dirección":req_node.get("address","")[:60],
                         "Código":", ".join(codes) if codes else "—",
                         "Load":fn.get("load",0),"Skill opt.":str(req_node.get("skills_optional",[]))})
        with st.expander("Ver tabla completa de nodos filtrados"):
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as w:
                df.to_excel(w, index=False)
            buf.seek(0)
            st.download_button("📥 Descargar Excel", buf, "filtrados.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    if not unattended:
        return

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Unattended ────────────────────────────────────────────────────────────
    st.markdown(f'<div class="section-title">⚠️ Nodos sin atender ({n_unat})</div>', unsafe_allow_html=True)
    if un_issues:
        for idx, issue in enumerate(un_issues, 1):
            render_issue_card(issue, idx)
    else:
        st.info("No se detectaron causas específicas. Revisar configuración general.")
    rows = [{"Ident":un["ident"],"Dirección":node_map.get(un["ident"],{}).get("address","")[:60],
             "Causa":un.get("cause",{}).get("code",""),
             "Zona(s)":str(node_map.get(un["ident"],{}).get("zones",[])),
             "Duration":node_map.get(un["ident"],{}).get("duration"),
             "Ventana":f"{node_map.get(un['ident'],{}).get('window_start')}–{node_map.get(un['ident'],{}).get('window_end')}",
             "Load":f"{un.get('load',0):.1f}"} for un in unattended]
    with st.expander("Ver tabla completa de nodos sin atender"):
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Fleet ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🚛 Utilización de flota</div>', unsafe_allow_html=True)
    if flt_issues2:
        for issue in flt_issues2:
            render_issue_card(issue)
    vrows = []
    for v_res in res.get("vehicles",[]):
        vid   = v_res["ident"]
        v_req = next((v for v in req.get("vehicles",[]) if v["ident"]==vid), {})
        visits= sum(1 for t in v_res.get("tours",[]) for n in t.get("nodes",[]) if not n["ident"].startswith("vehicle-"))
        load1 = sum(node_map.get(n["ident"],{}).get("load",0) or 0 for t in v_res.get("tours",[]) for n in t.get("nodes",[]) if not n["ident"].startswith("vehicle-"))
        cap1  = v_req.get("capacity",0) or 0
        pct_c = round(load1/cap1*100,1) if cap1 and cap1<1e15 else None
        vrows.append({"Vehículo":vid,"Visitas":visits,"Carga":round(load1,1),
                      "Capacidad":cap1 if cap1<1e15 else "ilimitada",
                      "% Cap.":f"{pct_c}%" if pct_c is not None else "—",
                      "Estado":"🟢 Activo" if visits>0 else "⚪ Inactivo"})
    st.dataframe(pd.DataFrame(vrows), use_container_width=True, hide_index=True)

    # ── Comparative — solo si hay diferenciador real ──────────────────────────
    r_nodes = [node_map[i] for i in routed_ids if i in node_map]
    u_nodes = [node_map[u["ident"]] for u in unattended if u["ident"] in node_map]
    if r_nodes and u_nodes:
        def win_min(n):
            try:
                ws,we = n.get("window_start",""), n.get("window_end","")
                h1,m1 = int(ws.split(":")[0]),int(ws.split(":")[1])
                h2,m2 = int(we.split(":")[0]),int(we.split(":")[1])
                return (h2*60+m2)-(h1*60+m1)
            except: return None

        r_l = [n.get("load",0) for n in r_nodes]
        u_l = [n.get("load",0) for n in u_nodes]
        r_w = [w for n in r_nodes if (w:=win_min(n)) is not None]
        u_w = [w for n in u_nodes if (w:=win_min(n)) is not None]

        load_ratio = avg(r_l)/max(avg(u_l),0.01) if avg(u_l)>0 else 1
        win_ratio  = avg(r_w)/max(avg(u_w),0.01)  if avg(u_w)>0 else 1
        has_differentiator = load_ratio > 2.0 or win_ratio > 1.5

        if has_differentiator:
            r_d = [d for n in r_nodes if (d:=try_int(n.get("duration"))) is not None]
            u_d = [d for n in u_nodes if (d:=try_int(n.get("duration"))) is not None]
            st.markdown('<div class="section-title">🔬 Análisis comparativo</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame([
                {"Campo":"Nodos","Enrutados":len(r_nodes),"Sin atender":len(u_nodes),"Diferencia":"—"},
                {"Campo":"Load promedio","Enrutados":avg(r_l),"Sin atender":avg(u_l),
                 "Diferencia":f"{load_ratio:.1f}×"},
                {"Campo":"Duration promedio","Enrutados":f"{avg(r_d)} min","Sin atender":f"{avg(u_d)} min","Diferencia":"—"},
                {"Campo":"Ventana promedio","Enrutados":f"{avg(r_w)} min","Sin atender":f"{avg(u_w)} min","Diferencia":"—"},
            ]), use_container_width=True, hide_index=True)
            if load_ratio > 2.0:
                st.info(f"🔍 **Diferenciador — Carga:** Los nodos enrutados tienen {load_ratio:.1f}× más carga ({avg(r_l):.0f} vs {avg(u_l):.0f}). El router priorizó los de mayor carga.")
            elif win_ratio > 1.5:
                st.info(f"🔍 **Diferenciador — Ventana:** Los nodos enrutados tienen ventanas {win_ratio:.1f}× más amplias ({avg(r_w):.0f} vs {avg(u_w):.0f} min).")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:11px;opacity:0.4;text-align:center">SimpliRoute Route Analyzer v3 · Issue Registry centralizado</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()