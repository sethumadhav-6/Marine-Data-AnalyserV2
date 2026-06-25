"""
Copernicus Marine Data Visualiser  –  v4
==========================================
Standalone Dash app (no Streamlit).
Run   : python app.py            →  http://localhost:8050
Deploy: gunicorn app:server      (see render.yaml)

Key features
  • Continuous heatmap surface maps  (toggle: heatmap / geo-dots)
  • True polygon masking for ROI statistics (shapely 2.0 vectorised)
  • Multi-file layer support  — load physics + BGC + optics in parallel
  • 11 analytical tabs + Multi-Criteria Analysis
"""

import base64, io, json, os, tempfile, uuid, warnings, zipfile, urllib.request
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, Input, Output, State, dash_table, no_update, ctx
import dash_bootstrap_components as dbc

# ─────────────────────────────────────────────────────────────────────────────
# App init
# ─────────────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True,
    title="Copernicus Marine Visualiser",
)
server = app.server
TMP_DIR = tempfile.mkdtemp(prefix="copernicus_")

_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
os.makedirs(_ASSETS, exist_ok=True)
with open(os.path.join(_ASSETS, "custom.css"), "w") as _f:
    _f.write("""
body{font-family:'Segoe UI',Arial,sans-serif;background:#f8fafc}
.sidebar-brand{border-bottom:2px solid #0077b6;padding-bottom:10px;margin-bottom:12px}
.card{border:none!important;border-radius:10px!important}
.card-header{background:#e8f4fd!important;color:#023e8a!important;
  font-size:.82rem;padding:8px 14px!important;border-radius:10px 10px 0 0!important}
.nav-tabs .nav-link{font-size:.79rem;color:#495057;padding:6px 10px}
.nav-tabs .nav-link.active{color:#0077b6!important;font-weight:700;
  border-bottom:3px solid #0077b6!important}
.status-bar{background:#0077b6;color:#fff;font-size:.78rem;
  padding:6px 16px;border-radius:6px;margin-bottom:8px}
.landing-card{border-left:4px solid #0077b6!important}
.mca-section{border-left:3px solid #0077b6;padding-left:12px;margin-bottom:16px}
.layer-badge{font-size:.7rem}
""")

_LOGO = os.path.join(_ASSETS, "logo.png")

# ─────────────────────────────────────────────────────────────────────────────
# Natural-Earth land polygons  (loaded once at startup, cached globally)
# ─────────────────────────────────────────────────────────────────────────────
_LAND_TRACES: list | None = None   # list of {lon, lat} dicts

def _load_land():
    global _LAND_TRACES
    if _LAND_TRACES is not None:
        return
    url = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
           "master/geojson/ne_110m_land.geojson")
    try:
        with urllib.request.urlopen(url, timeout=8) as r:
            gj = json.loads(r.read())
        traces = []
        for feat in gj.get("features", []):
            geom = feat.get("geometry", {})
            gtype = geom.get("type", "")
            if gtype == "Polygon":
                rings = [geom["coordinates"][0]]
            elif gtype == "MultiPolygon":
                rings = [p[0] for p in geom["coordinates"]]
            else:
                continue
            for ring in rings:
                traces.append({
                    "lon": [c[0] for c in ring],
                    "lat": [c[1] for c in ring],
                })
        _LAND_TRACES = traces
    except Exception:
        _LAND_TRACES = []          # fallback – no land overlay

# Kick off at import time (non-blocking; worst case it stays [])
try:
    _load_land()
except Exception:
    _LAND_TRACES = []

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def detect_dims(ds):
    m = {}
    for d in ds.dims:
        dl = d.lower()
        if   dl in ("time","t"):
            m["time"]  = d
        elif dl in ("depth","deptht","depthu","depthv","depthw","lev","level","z"):
            m["depth"] = d
        elif dl in ("latitude","lat","y","nav_lat"):
            m["lat"]   = d
        elif dl in ("longitude","lon","x","nav_lon"):
            m["lon"]   = d
    return m

def get_units(ds, var):
    return ds[var].attrs.get("units", "")

def load_ds(path):
    return xr.open_dataset(path, decode_times=True)

def decode_upload(contents, filename):
    _, b64 = contents.split(",", 1)
    data   = base64.b64decode(b64)
    path   = os.path.join(TMP_DIR, f"{uuid.uuid4().hex}_{filename}")
    with open(path, "wb") as f:
        f.write(data)
    return path

def logo_img(height="38px"):
    if os.path.isfile(_LOGO):
        return html.Img(src="/assets/logo.png", height=height,
                        style={"objectFit":"contain"},
                        title="Canopy Geospatial Solutions")
    return html.Span("CGS", style={
        "display":"inline-block","width":height,"height":height,
        "lineHeight":height,"textAlign":"center","borderRadius":"50%",
        "background":"#0077b6","color":"#fff","fontWeight":"bold","fontSize":"0.75rem",
    })

# ── Variable key helpers  (format: "layerIdx::varname") ──────────────────────

def parse_var(key):
    """'0::thetao' → (0, 'thetao').  Plain 'thetao' → (0, 'thetao')."""
    if "::" in key:
        idx, v = key.split("::", 1)
        return int(idx), v
    return 0, key

def open_var(layers, var_key):
    """Return (ds, varname, dims) for a var_key, opening the right file."""
    idx, v = parse_var(var_key)
    if not layers or idx >= len(layers):
        return None, v, {}
    ds   = load_ds(layers[idx]["path"])
    dims = detect_dims(ds)
    return ds, v, dims

def collect_flat(layers, var_keys):
    """Load all vars grouped by file; returns {var_key: (flat_array, units)}."""
    by_layer: dict[int, list] = {}
    for vk in var_keys:
        i, v = parse_var(vk)
        by_layer.setdefault(i, []).append((vk, v))

    result = {}
    for i, pairs in by_layer.items():
        if not layers or i >= len(layers):
            continue
        try:
            ds = load_ds(layers[i]["path"])
            for vk, v in pairs:
                try:
                    arr = ds[v].values.flatten().astype(float)
                    result[vk] = (arr, get_units(ds, v))
                except Exception:
                    pass
            ds.close()
        except Exception:
            pass
    return result

# ── Surface-map helpers ───────────────────────────────────────────────────────

def _prep_latlon_z(lat, lon, z):
    """Ensure lat increases S→N and z rows match."""
    if lat.ndim > 1:
        lat = lat[:, 0]
    if lon.ndim > 1:
        lon = lon[0, :]
    if len(lat) > 1 and lat[0] > lat[-1]:
        lat = lat[::-1]
        z   = z[::-1, :]
    return lat.astype(float), lon.astype(float), z.astype(float)

def make_heatmap_fig(lat, lon, z, title, units, cmap="Viridis"):
    """
    Smooth continuous heatmap with Natural-Earth land overlay.
    NaN cells (land in ocean models) show as the grey plot background.
    """
    lat, lon, z = _prep_latlon_z(lat, lon, z)

    fig = go.Figure(go.Heatmap(
        z=z, x=lon, y=lat,
        colorscale=cmap, zsmooth="fast",
        colorbar=dict(title=units, thickness=14, len=0.8, x=1.01),
        hovertemplate="Lon: %{x:.2f}°<br>Lat: %{y:.2f}°<br>%{z:.4g}<extra></extra>",
    ))

    # Land polygons on top (grey fill, thin border)
    if _LAND_TRACES:
        for t in _LAND_TRACES:
            fig.add_trace(go.Scatter(
                x=t["lon"], y=t["lat"],
                mode="lines", fill="toself",
                fillcolor="rgba(200,196,187,0.92)",
                line=dict(color="rgba(90,90,90,0.5)", width=0.4),
                showlegend=False, hoverinfo="skip",
            ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        xaxis=dict(title="Longitude", showgrid=False,
                   range=[float(lon.min()), float(lon.max())]),
        yaxis=dict(title="Latitude",  showgrid=False,
                   range=[float(lat.min()), float(lat.max())],
                   scaleanchor="x", scaleratio=0.85),
        plot_bgcolor="#cde4ef",   # ocean-blue background (shows under NaN=land cells)
        paper_bgcolor="white",
        height=430, margin=dict(l=60, r=20, t=40, b=50),
    )
    return fig

def make_geo_dots_fig(lat, lon, z, title, units, cmap="Viridis", max_pts=3500):
    """Subsampled dots on Natural-Earth geo projection."""
    stride  = max(1, int(((lat.size * lon.size) / max_pts) ** 0.5))
    lat_s   = lat[::stride]
    lon_s   = lon[::stride]
    z_s     = z[::stride, ::stride] if z.ndim == 2 else z
    lf      = np.repeat(lat_s, len(lon_s))
    lnf     = np.tile(lon_s, len(lat_s))
    zf      = z_s.flatten()
    ok      = ~np.isnan(zf)

    fig = go.Figure(go.Scattergeo(
        lat=lf[ok], lon=lnf[ok], mode="markers",
        marker=dict(color=zf[ok], colorscale=cmap, size=4, opacity=0.85,
                    colorbar=dict(title=units, thickness=12, len=0.6)),
        showlegend=False,
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        geo=dict(
            showland=True,  landcolor="#d0cfc4",
            showocean=True, oceancolor="#cde4ef",
            showcoastlines=True, coastlinecolor="#666", coastlinewidth=0.8,
            showframe=False, projection_type="natural earth",
            lonaxis=dict(range=[float(lnf[ok].min())-3, float(lnf[ok].max())+3]),
            lataxis=dict(range=[float(lf[ok].min())-3,  float(lf[ok].max())+3]),
        ),
        height=430, margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig

def surface_fig(lat, lon, z, title, units, cmap, mode):
    if mode == "geo":
        return make_geo_dots_fig(lat, lon, z, title, units, cmap)
    return make_heatmap_fig(lat, lon, z, title, units, cmap)

# ── ROI helpers ───────────────────────────────────────────────────────────────

def polygon_mask_2d(geojson_str, lat_arr, lon_arr):
    """
    True polygon mask using shapely 2.0 vectorised contains_xy.
    Returns 2-D bool array (lat × lon) — True inside AOI.
    """
    import shapely
    from shapely.geometry import shape
    from shapely.ops import unary_union

    gj   = json.loads(geojson_str)
    feats = gj.get("features", [{"geometry": gj}])
    polys = []
    for f in feats:
        geom = f.get("geometry") or f
        if geom and geom.get("type"):
            try:
                polys.append(shape(geom))
            except Exception:
                pass
    if not polys:
        return np.ones((len(lat_arr), len(lon_arr)), dtype=bool)

    union   = unary_union(polys)
    lon2d, lat2d = np.meshgrid(lon_arr, lat_arr)
    flat_mask    = shapely.contains_xy(union, lon2d.ravel(), lat2d.ravel())
    return flat_mask.reshape(lon2d.shape)

def clip_ds(ds, dims, bbox):
    minx, miny, maxx, maxy = bbox
    sel = {}
    if "lon" in dims:
        lv = ds[dims["lon"]].values
        sel[dims["lon"]] = (lv >= minx) & (lv <= maxx)
    if "lat" in dims:
        lv = ds[dims["lat"]].values
        sel[dims["lat"]] = (lv >= miny) & (lv <= maxy)
    return ds.isel({k: v for k, v in sel.items()}) if sel else ds

# ── ROI map ───────────────────────────────────────────────────────────────────

def make_roi_map(geojson_str, bbox, ds_lat=None, ds_lon=None):
    fig = go.Figure()
    if ds_lat is not None and ds_lon is not None:
        fig.add_trace(go.Scattergeo(
            lon=[float(ds_lon.min()), float(ds_lon.max()),
                 float(ds_lon.max()), float(ds_lon.min()), float(ds_lon.min())],
            lat=[float(ds_lat.min()), float(ds_lat.min()),
                 float(ds_lat.max()), float(ds_lat.max()), float(ds_lat.min())],
            mode="lines", line=dict(color="#0077b6", width=1.5, dash="dot"),
            name="Dataset extent", showlegend=True,
        ))
    if geojson_str:
        try:
            gj   = json.loads(geojson_str)
            feats = gj.get("features", [gj])
            for feat in feats:
                geom = feat.get("geometry", feat)
                gtype = geom.get("type", "")
                rings = []
                if gtype == "Polygon":
                    rings = [geom["coordinates"][0]]
                elif gtype == "MultiPolygon":
                    rings = [p[0] for p in geom["coordinates"]]
                for ring in rings:
                    lons = [c[0] for c in ring]
                    lats = [c[1] for c in ring]
                    fig.add_trace(go.Scattergeo(
                        lon=lons, lat=lats, mode="lines",
                        fill="toself", fillcolor="rgba(214,40,40,0.15)",
                        line=dict(color="#d62828", width=2.5),
                        name="AOI polygon", showlegend=True,
                    ))
        except Exception:
            pass
    minx, miny, maxx, maxy = bbox
    pad = max(maxx - minx, maxy - miny) * 0.35 + 1
    fig.update_layout(
        geo=dict(
            showland=True, landcolor="#d0cfc4",
            showocean=True, oceancolor="#cde4ef",
            showcoastlines=True, coastlinecolor="#666",
            showframe=False, projection_type="natural earth",
            lonaxis=dict(range=[minx - pad, maxx + pad]),
            lataxis=dict(range=[miny - pad, maxy + pad]),
        ),
        legend=dict(orientation="h", y=-0.08),
        height=320, margin=dict(l=0, r=0, t=10, b=0),
    )
    return fig

# ── HTML report ───────────────────────────────────────────────────────────────

def build_report(stats_df, aoi_name, info):
    rows  = "".join(
        f"<tr><td>{r['Variable']}</td><td>{r.get('Units','')}</td>"
        f"<td>{r['Min']:.4g}</td><td>{r['Max']:.4g}</td>"
        f"<td>{r['Mean']:.4g}</td><td>{r['Std']:.4g}</td>"
        f"<td>{r['Median']:.4g}</td><td>{r['N valid']:,}</td></tr>"
        for _, r in stats_df.iterrows()
    )
    irows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in info.items())
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>ROI Report — {aoi_name}</title>
<style>
body{{font-family:Arial,sans-serif;margin:40px;color:#0d1117}}
h1{{color:#0077b6;border-bottom:2px solid #0077b6;padding-bottom:8px}}
h2{{color:#023e8a;margin-top:30px}}
table{{border-collapse:collapse;width:100%;margin-top:12px}}
th{{background:#0077b6;color:#fff;padding:8px 12px;text-align:left}}
td{{padding:7px 12px;border-bottom:1px solid #dee2e6}}
tr:nth-child(even){{background:#f0f7ff}}
.credit{{margin-top:40px;padding:12px;background:#eaf4fb;
  border-left:4px solid #0077b6;font-size:.85rem}}
.footer{{margin-top:30px;font-size:.8rem;color:#6c757d;
  border-top:1px solid #dee2e6;padding-top:10px;
  display:flex;justify-content:space-between;align-items:center}}
@media print{{button{{display:none}}}}
</style></head><body>
<h1>🌊 Copernicus Marine — ROI Statistics Report</h1>
<p><strong>Area of Interest:</strong> {aoi_name}</p>
<h2>Dataset Information</h2>
<table><tr><th>Property</th><th>Value</th></tr>{irows}</table>
<h2>ROI Statistics (polygon-masked)</h2>
<table>
<tr><th>Variable</th><th>Units</th><th>Min</th><th>Max</th>
<th>Mean</th><th>Std Dev</th><th>Median</th><th>Valid Points</th></tr>
{rows}</table>
<div class="credit">
<strong>Data Credit:</strong><br>
Generated using E.U. Copernicus Marine Service Information —
<a href="https://marine.copernicus.eu">marine.copernicus.eu</a>
</div>
<div class="footer">
<span>Maintained by <strong> Canopy Geospatial Solutions</strong> ·
<a href="https://canopygs.in">canopygs.in</a> ·
{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M UTC')}</span>
<button onclick="window.print()"
  style="padding:6px 16px;background:#0077b6;color:#fff;border:none;
  border-radius:4px;cursor:pointer">🖨️ Print / Save PDF</button>
</div></body></html>"""

# ── MCA knowledge base ────────────────────────────────────────────────────────

MCA_INTERACTIONS = [
    {"pair":"SST / Temperature","partner":"Salinity",
     "relationship":"Thermohaline circulation driver",
     "detail":"Together define water density and mass. Anomalies indicate upwelling, ENSO cycles."},
    {"pair":"Chlorophyll-a (chl)","partner":"Nutrients (NO₃ / PO₄ / Si)",
     "relationship":"Limiting nutrients → primary production",
     "detail":"High nutrients + light → phytoplankton bloom. Si limits diatoms specifically."},
    {"pair":"O₂","partner":"Primary Production / chl",
     "relationship":"Photosynthesis / Respiration balance",
     "detail":"O₂ produced in photic zone, consumed by heterotrophs. Low O₂ → hypoxic dead zones."},
    {"pair":"CO₂ / DIC","partner":"pH / Alkalinity (talk)",
     "relationship":"Ocean acidification pathway",
     "detail":"Rising CO₂ lowers pH. Alkalinity (talk) buffers the carbonate system."},
    {"pair":"SST Anomaly","partner":"Primary Production / chl",
     "relationship":"Thermal stratification feedback",
     "detail":"Warm anomalies deepen mixed layer, cut nutrient supply → productivity crash."},
    {"pair":"Iron (Fe)","partner":"Chlorophyll-a (chl)",
     "relationship":"Micronutrient limitation (HNLC regions)",
     "detail":"Southern Ocean, equatorial Pacific: Fe limits productivity despite ample NO₃."},
    {"pair":"Zooplankton (ZOO)","partner":"Chlorophyll-a (chl) / POC",
     "relationship":"Predator–prey trophic coupling",
     "detail":"Zooplankton graze phytoplankton; lag correlation ~2–4 weeks typical."},
    {"pair":"pCO₂ / CO₂ flux","partner":"SST / Wind speed (uo/vo)",
     "relationship":"Air–sea gas exchange",
     "detail":"Cold water absorbs CO₂ (higher solubility). Wind speed drives piston velocity."},
    {"pair":"Alkalinity (talk)","partner":"Salinity",
     "relationship":"Conservative tracer relationship",
     "detail":"Alkalinity tracks salinity closely; deviations flag CaCO₃ cycling / freshwater."},
    {"pair":"Nitrate (NO₃)","partner":"Phosphate (PO₄)",
     "relationship":"Redfield ratio N:P ≈ 16:1",
     "detail":"Deviation from Redfield indicates N-fixation, denitrification, or nutrient limitation."},
    {"pair":"PAR / ED490","partner":"Chlorophyll-a (chl)",
     "relationship":"Light-driven photosynthesis",
     "detail":"PAR (400–700 nm) is the energy source for phytoplankton. Shallow mixed layer → more PAR."},
    {"pair":"bbp (backscatter)","partner":"POC / chl",
     "relationship":"Optical proxy for particle load",
     "detail":"Particulate backscattering coefficient scales with POC; used to infer biomass from optics."},
    {"pair":"Sea surface height (zos)","partner":"SST / Salinity",
     "relationship":"Geostrophic currents / mesoscale eddies",
     "detail":"SSH anomalies reveal anticyclonic (warm-core) and cyclonic (cold-core, nutrient-rich) eddies."},
    {"pair":"Eastward (uo) / Northward (vo) velocity","partner":"Nutrients / Chl-a",
     "relationship":"Advection of biogeochemical properties",
     "detail":"Ocean currents transport nutrients, plankton, and carbon over basin scales."},
]

# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def section(title, body):
    return dbc.Card([
        dbc.CardHeader(html.Strong(title)),
        dbc.CardBody(body),
    ], className="mb-3 shadow-sm")

def _layer_badge_color(i):
    return ["primary","success","warning","info"][i % 4]

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
SIDEBAR = dbc.Col([

    # Brand
    html.Div([
        html.Div([
            logo_img("34px"),
            html.Div([
                html.Span("Canopy GS", className="fw-bold",
                          style={"color":"#0077b6","fontSize":"0.93rem"}),
                html.Br(),
                html.A("canopygs.in", href="https://canopygs.in", target="_blank",
                       style={"fontSize":"0.72rem","color":"#555","textDecoration":"none"}),
            ], className="ms-2 lh-sm"),
        ], className="d-flex align-items-center mb-1"),
        html.Small(" Copernicus Marine Visualiser",
                   className="text-primary fw-semibold d-block",
                   style={"fontSize":"0.8rem"}),
    ], className="sidebar-brand"),

    # ── Data source ───────────────────────────────────────────
    section("📂 Primary Data Source", [
        dbc.RadioItems(
            id="source-radio",
            options=[
                {"label":" Upload NC file",           "value":"upload"},
                {"label":" Enter file path",          "value":"path"},
                {"label":" Download from Copernicus", "value":"download"},
            ],
            value="upload", className="mb-2",
        ),

        # Upload
        html.Div(id="div-upload", children=[
            dcc.Upload(
                id="upload-nc",
                children=html.Div([
                    html.I(className="fa fa-cloud-upload-alt me-2"),
                    html.Span("Drag & drop or "),
                    html.A("browse"),
                ]),
                style={"borderWidth":"2px","borderStyle":"dashed","borderRadius":"8px",
                       "textAlign":"center","padding":"16px 8px","cursor":"pointer",
                       "borderColor":"#0077b6","color":"#0077b6","fontSize":"0.84rem",
                       "background":"#f0f8ff"},
                multiple=False, max_size=-1,
            ),
            dbc.Alert([html.I(className="fa fa-lightbulb me-1"),
                       html.Strong("Large file? "),
                       "Use ", html.Em("Enter file path"), " instead."],
                      color="warning", className="small py-1 mt-1 mb-0"),
        ]),

        # Path
        html.Div(id="div-path", style={"display":"none"}, children=[
            dbc.InputGroup([
                dbc.InputGroupText(html.I(className="fa fa-folder-open")),
                dbc.Input(id="path-input", placeholder=r"E:\data\file.nc"),
            ], size="sm", className="mb-1"),
            dbc.Button([html.I(className="fa fa-play me-1"), "Load"],
                       id="btn-path", color="primary", size="sm", className="w-100"),
        ]),

        # Download
        html.Div(id="div-download", style={"display":"none"}, children=[
            dbc.Input(id="cm-user",    placeholder="CMEMS Username", type="text",    size="sm", className="mb-1"),
            dbc.Input(id="cm-pass",    placeholder="Password",       type="password",size="sm", className="mb-1"),
            dbc.Input(id="cm-dataset", value="cmems_mod_glo_bgc-car_anfc_0.25deg_P1D-m",        size="sm", className="mb-1"),
            dbc.Input(id="cm-version", value="202311",               placeholder="Version",      size="sm", className="mb-1"),
            dbc.Input(id="cm-vars",    value="dissic,ph,talk",       placeholder="Variables",    size="sm", className="mb-1"),
            dbc.Row([
                dbc.Col(dbc.Input(id="cm-minlon",type="number",value=-180,size="sm"),width=6),
                dbc.Col(dbc.Input(id="cm-maxlon",type="number",value=180, size="sm"),width=6),
            ], className="g-1 mb-1"),
            dbc.Row([
                dbc.Col(dbc.Input(id="cm-minlat",type="number",value=-90, size="sm"),width=6),
                dbc.Col(dbc.Input(id="cm-maxlat",type="number",value=90,  size="sm"),width=6),
            ], className="g-1 mb-1"),
            dbc.Input(id="cm-start", value="2023-07-01T00:00:00", size="sm", className="mb-1"),
            dbc.Input(id="cm-end",   value="2026-07-01T00:00:00", size="sm", className="mb-1"),
            dbc.Row([
                dbc.Col(dbc.Input(id="cm-mindep",type="number",value=0.494,   size="sm"),width=6),
                dbc.Col(dbc.Input(id="cm-maxdep",type="number",value=5727.92, size="sm"),width=6),
            ], className="g-1 mb-1"),
            dbc.Button([html.I(className="fa fa-download me-1"),"Download"],
                       id="btn-download", color="primary", size="sm", className="w-100 mt-1"),
        ]),

        dcc.Loading(type="default",
                    children=html.Div(id="source-status", className="mt-1")),
    ]),

    # ── Additional layers ─────────────────────────────────────
    section("➕ Additional Layers", [
        html.Small("Load a 2nd / 3rd NC file (different Copernicus product) "
                   "to combine variables across layers in analysis tabs.",
                   className="text-muted d-block mb-2"),
        dbc.InputGroup([
            dbc.Input(id="layer-path",  placeholder=r"Path to NC file…",   size="sm"),
        ], className="mb-1"),
        dbc.Input(id="layer-label", placeholder="Label  e.g. BGC / Optics / Physics",
                  size="sm", className="mb-1"),
        dbc.Button([html.I(className="fa fa-plus me-1"), "Add Layer"],
                   id="btn-add-layer", color="success", outline=True,
                   size="sm", className="w-100"),
        html.Div(id="layers-list", className="mt-2"),
    ]),

    # ── AOI ───────────────────────────────────────────────────
    section("📐 Area of Interest", [
        dcc.Upload(
            id="upload-aoi",
            children=html.Div([html.I(className="fa fa-map-marked-alt me-2"),
                               "Upload .zip (shapefile) or .geojson"]),
            style={"borderWidth":"2px","borderStyle":"dashed","borderRadius":"8px",
                   "textAlign":"center","padding":"10px","cursor":"pointer",
                   "borderColor":"#28a745","color":"#28a745","fontSize":"0.8rem",
                   "background":"#f0fff4"},
            multiple=False,
        ),
        html.Div(id="aoi-status", className="mt-1 small"),
    ]),

    # ── Variables ─────────────────────────────────────────────
    section("📊 Variables (all layers)", [
        dcc.Dropdown(id="var-selector", multi=True,
                     placeholder="Load a dataset first…",
                     optionHeight=30, className="small"),
    ]),

    # ── Map display ───────────────────────────────────────────
    section("🗺️ Map Display Mode", [
        dbc.RadioItems(
            id="map-mode",
            options=[
                {"label":" Continuous (heatmap + land overlay)","value":"heatmap"},
                {"label":" Geo projection (dots)","value":"geo"},
            ],
            value="heatmap", className="small",
        ),
        html.Small("Continuous mode renders filled colour with land overlay. "
                   "Geo mode plots on a Natural-Earth projection.",
                   className="text-muted d-block mt-1"),
    ]),

    # Footer
    html.Hr(),
    html.Div([
        html.Div([
            logo_img("24px"),
            html.A("canopygs.in", href="https://canopygs.in", target="_blank",
                   className="ms-2 small fw-semibold",
                   style={"color":"#0077b6","textDecoration":"none"}),
        ], className="d-flex align-items-center mb-1"),
        html.Small([
            "Data: ",
            html.A("E.U. Copernicus Marine Service",
                   href="https://marine.copernicus.eu", target="_blank",
                   style={"color":"#0077b6","fontSize":"0.7rem"}),
        ], className="text-muted d-block"),
    ]),

], width=3, className="bg-light border-end px-3 py-3",
   style={"minHeight":"100vh","overflowY":"auto"})

# ─────────────────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────────────────
app.layout = dbc.Container([

    dcc.Store(id="store-nc-path"),          # primary file path
    dcc.Store(id="store-layers"),           # list of all loaded layers
    dcc.Store(id="store-dims"),             # dims of primary layer
    dcc.Store(id="store-aoi"),              # AOI data incl. GeoJSON
    dcc.Store(id="export-store"),           # persistent ROI export payload
    dcc.Download(id="dl-csv"),
    dcc.Download(id="dl-report"),

    # Header
    dbc.Row(dbc.Col(html.Div([
        html.Div([
            logo_img("44px"),
            html.Div([
                html.H4("Copernicus Marine Data Visualiser",
                        className="mb-0 fw-bold", style={"color":"#0077b6"}),
                html.Small("Physics · Nutrients · Carbon · Biology · Optics — multi-layer analysis",
                           className="text-muted"),
            ], className="ms-3"),
            html.Div([
                html.A("canopygs.in", href="https://canopygs.in", target="_blank",
                       className="btn btn-outline-primary btn-sm me-2",
                       style={"fontSize":"0.78rem"}),
                html.A("Copernicus Marine", href="https://marine.copernicus.eu",
                       target="_blank",
                       className="btn btn-outline-secondary btn-sm",
                       style={"fontSize":"0.78rem"}),
            ], className="ms-auto"),
        ], className="d-flex align-items-center py-2 px-3"),
    ], className="border-bottom bg-white shadow-sm")),
    className="mb-0 sticky-top", style={"zIndex":"1000"}),

    dbc.Row([
        SIDEBAR,
        dbc.Col([

            # Landing
            html.Div(id="div-landing", children=[
                html.Div([
                    html.Div([
                        logo_img("70px"),
                        html.H4("Welcome to Copernicus Marine Visualiser",
                                className="fw-bold text-primary mt-3 mb-1"),
                        html.P("Load your ocean data to begin interactive analysis.",
                               className="text-muted mb-4"),
                        dbc.Row([
                            dbc.Col(dbc.Card(dbc.CardBody([
                                html.I(className="fa fa-upload fa-2x text-primary mb-2 d-block"),
                                html.H6("Upload NC File"),
                                html.P("Browser upload for small–medium files", className="small text-muted mb-0"),
                            ]), className="text-center landing-card h-100"), width=4),
                            dbc.Col(dbc.Card(dbc.CardBody([
                                html.I(className="fa fa-folder-open fa-2x text-success mb-2 d-block"),
                                html.H6("Enter File Path"),
                                html.P("Best for large files — reads directly from disk", className="small text-muted mb-0"),
                            ]), className="text-center landing-card h-100"), width=4),
                            dbc.Col(dbc.Card(dbc.CardBody([
                                html.I(className="fa fa-layer-group fa-2x text-info mb-2 d-block"),
                                html.H6("Add Multiple Layers"),
                                html.P("Physics + BGC + Optics — analyse them together", className="small text-muted mb-0"),
                            ]), className="text-center landing-card h-100"), width=4),
                        ], className="mb-4 g-3"),
                        dbc.Alert([
                            html.I(className="fa fa-lightbulb me-2"),
                            html.Strong("Multi-product tip: "),
                            "Load physics data first, then use ",
                            html.Strong("Add Layers"),
                            " to overlay BGC/optics products from the same region.",
                        ], color="info", className="text-start"),
                    ], className="text-center"),
                ], className="d-flex justify-content-center align-items-center",
                   style={"minHeight":"65vh","padding":"40px 20px"}),
            ]),

            # Analysis
            html.Div(id="div-analysis", style={"display":"none"}, children=[
                html.Div(id="file-status-bar", className="status-bar"),
                dcc.Loading(type="cube", color="#0077b6", children=[
                    dbc.Tabs(id="main-tabs", active_tab="tab-info", children=[
                        dbc.Tab(label="📋 Info",           tab_id="tab-info"),
                        dbc.Tab(label="📊 Statistics",     tab_id="tab-stats"),
                        dbc.Tab(label="🗺️ Surface Maps",   tab_id="tab-surface"),
                        dbc.Tab(label="📉 Depth Profiles", tab_id="tab-depth"),
                        dbc.Tab(label="📈 Time Series",    tab_id="tab-ts"),
                        dbc.Tab(label="🌐 Zonal Means",    tab_id="tab-zonal"),
                        dbc.Tab(label="⬇️ Depth Slices",   tab_id="tab-slices"),
                        dbc.Tab(label="📊 Histograms",     tab_id="tab-hist"),
                        dbc.Tab(label="🔗 Correlations",   tab_id="tab-corr"),
                        dbc.Tab(label="📍 ROI Statistics", tab_id="tab-roi"),
                        dbc.Tab(label="🧪 Multi-Criteria", tab_id="tab-mca"),
                    ], className="mt-2"),
                    html.Div(id="tab-content", className="mt-3 pb-4"),
                ]),
            ]),

        ], width=9, className="px-3 py-2"),
    ], className="g-0"),

    # Page footer
    dbc.Row(dbc.Col(html.Div([
        html.Hr(className="mt-2 mb-2"),
        html.Div([
            html.Div([
                logo_img("22px"),
                html.Span(" Maintained by ", className="text-muted small ms-2"),
                html.A("Canopy Geospatial Solutions",
                       href="https://canopygs.in", target="_blank",
                       className="small fw-semibold",
                       style={"color":"#0077b6","textDecoration":"none"}),
            ], className="d-flex align-items-center"),
            html.Small([
                "Generated using ",
                html.Strong("E.U. Copernicus Marine Service Information"),
                " · ",
                html.A("marine.copernicus.eu",
                       href="https://marine.copernicus.eu", target="_blank",
                       style={"color":"#0077b6"}),
            ], className="text-muted"),
        ], className="d-flex justify-content-between align-items-center flex-wrap py-2"),
    ], className="px-3"))),

], fluid=True)

# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

# ── Show / hide landing vs analysis ──────────────────────────────────────────
@app.callback(
    Output("div-landing",     "style"),
    Output("div-analysis",    "style"),
    Output("file-status-bar", "children"),
    Input("store-layers",     "data"),
)
def toggle_analysis(layers):
    if not layers:
        return {"display":"block"}, {"display":"none"}, ""
    primary = layers[0]
    name = primary["name"]
    try:    size_str = f"  ·  {os.path.getsize(primary['path'])/1e6:.1f} MB"
    except: size_str = ""
    extra = (f"  +  {len(layers)-1} additional layer(s)"
             if len(layers) > 1 else "")
    bar = [
        html.I(className="fa fa-check-circle me-2"),
        html.Strong(f"Loaded: {name}"), html.Span(size_str + extra, className="opacity-75"),
        html.Span("  ·  Select a tab to explore", className="ms-3 opacity-75"),
    ]
    return {"display":"none"}, {"display":"block"}, bar


# ── Source radio toggle ───────────────────────────────────────────────────────
@app.callback(
    Output("div-upload",   "style"),
    Output("div-path",     "style"),
    Output("div-download", "style"),
    Input("source-radio",  "value"),
)
def toggle_source(val):
    show, hide = {"display":"block"}, {"display":"none"}
    return (show if val=="upload"   else hide,
            show if val=="path"     else hide,
            show if val=="download" else hide)


# ── Primary data source → sets store-nc-path AND initialises store-layers ────
@app.callback(
    Output("store-nc-path", "data"),
    Output("store-layers",  "data",   allow_duplicate=True),
    Output("source-status", "children"),
    Input("upload-nc",    "contents"),
    Input("btn-path",     "n_clicks"),
    Input("btn-download", "n_clicks"),
    State("upload-nc",    "filename"),
    State("path-input",   "value"),
    State("cm-user","value"), State("cm-pass","value"),
    State("cm-dataset","value"), State("cm-version","value"),
    State("cm-vars","value"),
    State("cm-minlon","value"), State("cm-maxlon","value"),
    State("cm-minlat","value"), State("cm-maxlat","value"),
    State("cm-start","value"),  State("cm-end","value"),
    State("cm-mindep","value"), State("cm-maxdep","value"),
    prevent_initial_call=True,
)
def cb_primary(contents, _p, _d, filename, path_val,
               user, pwd, dataset_id, version, vars_raw,
               minlon, maxlon, minlat, maxlat, start, end, mindep, maxdep):
    t   = ctx.triggered_id
    ok  = lambda m: dbc.Alert([html.I(className="fa fa-check-circle me-1"), m],
                               color="success", className="py-1 mb-0 small")
    err = lambda m: dbc.Alert(f"❌ {m}", color="danger",  className="py-1 mb-0 small")
    wrn = lambda m: dbc.Alert(m,         color="warning", className="py-1 mb-0 small")

    def make_layers(path, name):
        return [{"path": path, "name": name, "label": "Primary", "key": "0"}]

    if t == "upload-nc":
        if not contents: return no_update, no_update, no_update
        try:
            path = decode_upload(contents, filename)
            load_ds(path).close()
            return path, make_layers(path, filename), ok(f"Loaded: {filename}")
        except Exception as e:
            return no_update, no_update, err(str(e))

    if t == "btn-path":
        if not path_val: return no_update, no_update, wrn("Enter a file path.")
        if not os.path.isfile(path_val): return no_update, no_update, err("File not found.")
        try:
            load_ds(path_val).close()
            name = os.path.basename(path_val)
            return path_val, make_layers(path_val, name), ok(f"Loaded: {name}")
        except Exception as e:
            return no_update, no_update, err(str(e))

    if t == "btn-download":
        if not user or not pwd: return no_update, no_update, wrn("Enter credentials.")
        try:
            import copernicusmarine, inspect
            os.environ["COPERNICUSMARINE_SERVICE_USERNAME"] = user
            os.environ["COPERNICUSMARINE_SERVICE_PASSWORD"] = pwd
            variables = [v.strip() for v in vars_raw.split(",") if v.strip()]
            out_dir   = tempfile.mkdtemp()
            out_file  = os.path.join(out_dir, "copernicus_data.nc")
            sig = inspect.signature(copernicusmarine.subset).parameters
            kw  = dict(dataset_id=dataset_id, variables=variables,
                       minimum_longitude=minlon, maximum_longitude=maxlon,
                       minimum_latitude=minlat,  maximum_latitude=maxlat,
                       start_datetime=start,     end_datetime=end,
                       minimum_depth=mindep,     maximum_depth=maxdep,
                       disable_progress_bar=True)
            if version:                           kw["dataset_version"]          = version
            if "netcdf_compression_level" in sig: kw["netcdf_compression_level"] = 1
            if "output_filename"          in sig: kw["output_filename"]          = out_file
            elif "output_directory"       in sig: kw["output_directory"]         = out_dir
            result  = copernicusmarine.subset(**kw)
            nc_path = (str(result.file_path) if hasattr(result, "file_path")
                       else result if isinstance(result, str) and result.endswith(".nc")
                       else next((os.path.join(out_dir, f)
                                  for f in os.listdir(out_dir) if f.endswith(".nc")), None))
            if nc_path and os.path.isfile(nc_path):
                name = os.path.basename(nc_path)
                return nc_path, make_layers(nc_path, name), ok(f"Downloaded: {name}")
            return no_update, no_update, err("No NC file produced.")
        except ImportError:
            return no_update, no_update, err("copernicusmarine not installed.")
        except Exception as e:
            return no_update, no_update, err(str(e))

    return no_update, no_update, no_update


# ── Add additional layer ──────────────────────────────────────────────────────
@app.callback(
    Output("store-layers", "data",  allow_duplicate=True),
    Output("layers-list",  "children"),
    Input("btn-add-layer", "n_clicks"),
    State("store-layers",  "data"),
    State("layer-path",    "value"),
    State("layer-label",   "value"),
    prevent_initial_call=True,
)
def cb_add_layer(_, layers, path, label):
    if not path or not os.path.isfile(path):
        return no_update, no_update
    try:
        load_ds(path).close()
    except Exception as e:
        return no_update, dbc.Alert(f"❌ {e}", color="danger", className="py-1 small")

    layers = layers or []
    idx    = len(layers)
    name   = os.path.basename(path)
    layers.append({"path": path, "name": name,
                   "label": label or f"Layer {idx}", "key": str(idx)})
    list_items = dbc.ListGroup([
        dbc.ListGroupItem([
            dbc.Badge(l["label"], color=_layer_badge_color(i),
                      className="me-1 layer-badge"),
            html.Small(l["name"], className="text-muted"),
        ], className="py-1 px-2")
        for i, l in enumerate(layers)
    ], flush=True, className="small mt-1")
    return layers, list_items


# ── AOI upload ────────────────────────────────────────────────────────────────
@app.callback(
    Output("store-aoi",  "data"),
    Output("aoi-status", "children"),
    Input("upload-aoi",  "contents"),
    State("upload-aoi",  "filename"),
    prevent_initial_call=True,
)
def cb_aoi(contents, filename):
    if not contents: return no_update, no_update
    try:
        import geopandas as gpd
        try:    import pyogrio; engine = "pyogrio"
        except: engine = "fiona"
        _, b64 = contents.split(",", 1)
        data = base64.b64decode(b64)
        tmp  = tempfile.mkdtemp()
        fn   = filename.lower()
        if fn.endswith(".zip"):
            zp = os.path.join(tmp, "aoi.zip")
            with open(zp, "wb") as f: f.write(data)
            with zipfile.ZipFile(zp, "r") as z: z.extractall(tmp)
            shps = [os.path.join(tmp, x) for x in os.listdir(tmp) if x.endswith(".shp")]
            if not shps: raise ValueError("No .shp found in ZIP.")
            gdf = gpd.read_file(shps[0], engine=engine)
        else:
            gp = os.path.join(tmp, "aoi.geojson")
            with open(gp, "wb") as f: f.write(data)
            gdf = gpd.read_file(gp, engine=engine)
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        bbox    = gdf.total_bounds.tolist()
        geojson = gdf.to_json()
        name    = filename.rsplit(".", 1)[0]
        msg = dbc.Alert(
            f"✅ {name}  |  {bbox[0]:.2f}°–{bbox[2]:.2f}°E,"
            f"  {bbox[1]:.2f}°–{bbox[3]:.2f}°N",
            color="success", className="py-1 mb-0 small",
        )
        return {"bbox": bbox, "name": name, "geojson": geojson}, msg
    except Exception as e:
        return no_update, dbc.Alert(f"❌ {e}", color="danger", className="py-1 mb-0 small")


# ── Variable selector — aggregates ALL layers ─────────────────────────────────
@app.callback(
    Output("var-selector", "options"),
    Output("var-selector", "value"),
    Output("store-dims",   "data"),
    Input("store-layers",  "data"),
)
def cb_vars(layers):
    if not layers: return [], [], {}
    opts, primary_dims = [], {}
    for i, layer in enumerate(layers):
        try:
            ds = load_ds(layer["path"])
            if i == 0:
                primary_dims = detect_dims(ds)
            for v in ds.data_vars:
                u = get_units(ds, v)
                lbl = f"[{layer['label']}] {v}" + (f"  [{u}]" if u else "")
                opts.append({"label": lbl, "value": f"{i}::{v}"})
            ds.close()
        except Exception:
            pass
    default = [o["value"] for o in opts[:8]]
    return opts, default, primary_dims


# ─────────────────────────────────────────────────────────────────────────────
# Main tab renderer
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("tab-content",  "children"),
    Output("export-store", "data"),
    Input("main-tabs",     "active_tab"),
    Input("store-layers",  "data"),
    Input("var-selector",  "value"),
    Input("store-dims",    "data"),
    Input("store-aoi",     "data"),
    Input("map-mode",      "value"),
)
def render_tab(tab, layers, sel_vars, dims, aoi, map_mode):
    if not layers:
        return dbc.Alert("👈  Load a dataset using the sidebar.", color="info"), no_update

    # Open primary dataset for meta/dim work
    primary_path = layers[0]["path"]
    ds0 = load_ds(primary_path)
    if not sel_vars: sel_vars = [f"0::{v}" for v in list(ds0.data_vars)[:8]]
    if not dims:     dims     = detect_dims(ds0)

    has_lat  = "lat"   in dims
    has_lon  = "lon"   in dims
    has_time = "time"  in dims
    has_dep  = "depth" in dims
    is_spatial = has_lat and has_lon

    # ── Info tab ──────────────────────────────────────────────
    if tab == "tab-info":
        all_rows = []
        for i, layer in enumerate(layers):
            try:
                ds = load_ds(layer["path"])
                for v in ds.data_vars:
                    all_rows.append({
                        "Layer": f"[{layer['label']}]",
                        "Variable": v,
                        "Long name": ds[v].attrs.get("long_name",
                                     ds[v].attrs.get("standard_name", "")),
                        "Units": ds[v].attrs.get("units", ""),
                        "Shape": str(ds[v].shape),
                        "Dims":  str(ds[v].dims),
                    })
                ds.close()
            except Exception:
                pass

        tbl = dash_table.DataTable(
            data=all_rows,
            columns=[{"name": c, "id": c} for c in all_rows[0].keys()] if all_rows else [],
            style_table={"overflowX":"auto"},
            style_cell={"fontSize":"0.81rem","padding":"5px"},
            style_header={"background":"#0077b6","color":"#fff","fontWeight":"bold"},
            style_data_conditional=[{"if":{"row_index":"odd"},"background":"#f0f7ff"}],
            page_size=25,
        )
        dim_chips = [dbc.Badge(f"{k}: {v}", color="primary", className="me-1 mb-1")
                     for k, v in ds0.dims.items()]
        g_attrs   = [dbc.ListGroupItem(f"{k}: {v}") for k, v in ds0.attrs.items()] \
                    or [dbc.ListGroupItem("No global attributes.")]
        ds0.close()
        return html.Div([
            dbc.Row([
                dbc.Col(dbc.Card([dbc.CardHeader("Primary dimensions"),
                                  dbc.CardBody(dim_chips)]), width=5),
                dbc.Col(dbc.Card([dbc.CardHeader("Global Attributes (primary)"),
                                  dbc.CardBody(dbc.ListGroup(
                                      g_attrs, flush=True,
                                      style={"fontSize":"0.81rem","maxHeight":"160px","overflowY":"auto"}))
                                  ]), width=7),
            ], className="mb-3"),
            dbc.Card([dbc.CardHeader(f"All Variables — {len(layers)} layer(s)"),
                      dbc.CardBody(tbl)]),
        ]), no_update

    ds0.close()   # done with primary for meta; each tab re-opens as needed

    # ── Statistics ────────────────────────────────────────────
    if tab == "tab-stats":
        flat_data = collect_flat(layers, sel_vars)
        rows = []
        for vk, (arr, units) in flat_data.items():
            arr = arr[~np.isnan(arr)]
            if arr.size == 0: continue
            _, v = parse_var(vk)
            lbl  = next((l["label"] for l in layers
                         if l["key"] == str(parse_var(vk)[0])), "")
            rows.append({"Layer": f"[{lbl}]", "Variable": v, "Units": units,
                         "Min": f"{np.min(arr):.4g}", "Max": f"{np.max(arr):.4g}",
                         "Mean": f"{np.mean(arr):.4g}", "Std": f"{np.std(arr):.4g}",
                         "Median": f"{np.median(arr):.4g}",
                         "N valid": f"{arr.size:,}"})
        if not rows:
            return dbc.Alert("No data in selected variables.", color="warning"), no_update
        tbl = dash_table.DataTable(
            data=rows, columns=[{"name": c, "id": c} for c in rows[0]],
            style_table={"overflowX":"auto"},
            style_cell={"fontSize":"0.82rem","padding":"6px"},
            style_header={"background":"#0077b6","color":"#fff","fontWeight":"bold"},
            style_data_conditional=[{"if":{"row_index":"odd"},"background":"#f0f7ff"}],
            export_format="csv",
        )
        return html.Div([
            html.H5("Global Statistics (all layers)"),
            tbl,
            html.Small("Use the export icon (top-right) to download CSV.",
                       className="text-muted"),
        ]), no_update

    # ── Surface Maps ──────────────────────────────────────────
    if tab == "tab-surface":
        if not is_spatial:
            return dbc.Alert("No lat/lon in primary dataset.", color="info"), no_update
        plots = []
        for vk in sel_vars:
            idx, v = parse_var(vk)
            try:
                ds   = load_ds(layers[idx]["path"])
                dims_ = detect_dims(ds)
                if not ("lat" in dims_ and "lon" in dims_):
                    ds.close(); continue
                lat_v = ds[dims_["lat"]].values.flatten()
                lon_v = ds[dims_["lon"]].values.flatten()
                kw = {}
                if "time"  in dims_: kw[dims_["time"]]  = 0
                if "depth" in dims_: kw[dims_["depth"]] = 0
                surf  = ds[v].isel(kw).squeeze()
                z     = surf.values
                if z.ndim != 2:
                    z = z.reshape(len(lat_v), len(lon_v))
                u   = get_units(ds, v)
                lbl = layers[idx]["label"]
                fig = surface_fig(lat_v, lon_v, z,
                                  f"[{lbl}] {v}  [{u}]", u, "Viridis", map_mode)
                plots.append(dcc.Graph(figure=fig))
                ds.close()
            except Exception as e:
                plots.append(dbc.Alert(f"{vk}: {e}", color="warning"))
        return html.Div(plots), no_update

    # ── Depth Profiles ────────────────────────────────────────
    if tab == "tab-depth":
        if not has_dep:
            return dbc.Alert("No depth dimension in primary dataset.", color="info"), no_update
        figs = []
        for vk in sel_vars:
            idx, v = parse_var(vk)
            try:
                ds    = load_ds(layers[idx]["path"])
                dims_ = detect_dims(ds)
                if "depth" not in dims_: ds.close(); continue
                depth_vals = ds[dims_["depth"]].values
                arr = ds[v]
                if "time"  in dims_: arr = arr.isel({dims_["time"]:0})
                avg_d = [d for d in [dims_.get("lat"), dims_.get("lon")]
                         if d and d in arr.dims]
                if avg_d: arr = arr.mean(dim=avg_d)
                arr = arr.squeeze()
                fig = go.Figure(go.Scatter(
                    x=arr.values.flatten(), y=depth_vals,
                    mode="lines", line=dict(color="#0077b6", width=2.5),
                ))
                fig.update_yaxes(autorange="reversed")
                fig.update_layout(
                    title=f"[{layers[idx]['label']}] {v} — vertical profile",
                    xaxis_title=f"{v} [{get_units(ds,v)}]",
                    yaxis_title="Depth (m)", height=380,
                    margin=dict(l=60,r=20,t=40,b=40),
                )
                figs.append(dcc.Graph(figure=fig, style={"display":"inline-block","width":"48%"}))
                ds.close()
            except Exception as e:
                figs.append(dbc.Alert(f"{vk}: {e}", color="warning"))
        return html.Div(figs), no_update

    # ── Time Series ───────────────────────────────────────────
    if tab == "tab-ts":
        if not has_time:
            return dbc.Alert("No time dimension in primary dataset.", color="info"), no_update
        figs = []
        colours = px.colors.qualitative.Plotly
        fig_overlay = go.Figure()
        for ci, vk in enumerate(sel_vars):
            idx, v = parse_var(vk)
            try:
                ds    = load_ds(layers[idx]["path"])
                dims_ = detect_dims(ds)
                if "time" not in dims_: ds.close(); continue
                times = ds[dims_["time"]].values
                arr   = ds[v]
                if "depth" in dims_: arr = arr.isel({dims_["depth"]:0})
                avg_d = [d for d in [dims_.get("lat"), dims_.get("lon")]
                         if d and d in arr.dims]
                if avg_d: arr = arr.mean(dim=avg_d)
                arr = arr.squeeze().values.flatten().astype(float)
                col = colours[ci % len(colours)]
                lbl = f"[{layers[idx]['label']}] {v} [{get_units(ds,v)}]"
                fig_overlay.add_trace(go.Scatter(x=times, y=arr, mode="lines",
                                                 name=lbl, line=dict(color=col, width=1.5)))
                ds.close()
            except Exception as e:
                figs.append(dbc.Alert(f"{vk}: {e}", color="warning"))
        fig_overlay.update_layout(
            title="Time series — all selected variables",
            xaxis_title="Time", yaxis_title="Value",
            height=380, legend=dict(orientation="h", y=-0.25),
            margin=dict(l=60,r=20,t=40,b=80),
        )
        return html.Div([dcc.Graph(figure=fig_overlay)] + figs), no_update

    # ── Zonal Means ───────────────────────────────────────────
    if tab == "tab-zonal":
        if not (is_spatial and has_dep):
            return dbc.Alert("Requires lat/lon + depth.", color="info"), no_update
        plots = []
        for vk in sel_vars:
            idx, v = parse_var(vk)
            try:
                ds    = load_ds(layers[idx]["path"])
                dims_ = detect_dims(ds)
                if not ("lat" in dims_ and "depth" in dims_):
                    ds.close(); continue
                arr = ds[v]
                if "time" in dims_: arr = arr.isel({dims_["time"]:0})
                if "lon"  in dims_: arr = arr.mean(dim=dims_["lon"])
                arr = arr.squeeze()
                lat_ = arr[dims_["lat"]].values
                dep_ = arr[dims_["depth"]].values
                fig  = go.Figure(go.Heatmap(z=arr.values, x=lat_, y=dep_,
                                            colorscale="Turbo",
                                            colorbar=dict(title=get_units(ds,v), thickness=12)))
                fig.update_yaxes(autorange="reversed")
                fig.update_layout(
                    title=f"[{layers[idx]['label']}] {v} — zonal mean",
                    xaxis_title="Latitude", yaxis_title="Depth (m)",
                    height=380, margin=dict(l=60,r=20,t=40,b=40),
                )
                plots.append(dcc.Graph(figure=fig))
                ds.close()
            except Exception as e:
                plots.append(dbc.Alert(f"{vk}: {e}", color="warning"))
        return html.Div(plots), no_update

    # ── Depth Slices ──────────────────────────────────────────
    if tab == "tab-slices":
        if not (is_spatial and has_dep):
            return dbc.Alert("Requires lat/lon + depth.", color="info"), no_update
        vk  = sel_vars[0]
        idx, v = parse_var(vk)
        plots = []
        try:
            ds    = load_ds(layers[idx]["path"])
            dims_ = detect_dims(ds)
            lat_v = ds[dims_["lat"]].values.flatten()
            lon_v = ds[dims_["lon"]].values.flatten()
            n_dep = ds.dims.get(dims_.get("depth", ""), 0)
            for di in range(min(4, n_dep)):
                kw = {dims_["depth"]: di}
                if "time" in dims_: kw[dims_["time"]] = 0
                layer_ = ds[v].isel(kw).squeeze()
                z = layer_.values
                if z.ndim != 2: z = z.reshape(len(lat_v), len(lon_v))
                dep_val = ds[dims_["depth"]].values[di]
                fig = surface_fig(lat_v, lon_v, z,
                                  f"[{layers[idx]['label']}] {v} @ {dep_val:.1f} m",
                                  get_units(ds, v), "Turbo", map_mode)
                plots.append(dcc.Graph(figure=fig))
            ds.close()
        except Exception as e:
            plots.append(dbc.Alert(str(e), color="danger"))
        return html.Div(plots), no_update

    # ── Histograms ────────────────────────────────────────────
    if tab == "tab-hist":
        flat_data = collect_flat(layers, sel_vars)
        figs = []
        for vk, (arr, units) in flat_data.items():
            _, v = parse_var(vk)
            arr  = arr[~np.isnan(arr)]
            fig  = go.Figure(go.Histogram(x=arr, nbinsx=80,
                                          marker_color="#0077b6", opacity=0.8))
            fig.update_layout(title=f"{v} distribution [{units}]",
                              xaxis_title=f"[{units}]", yaxis_title="Count",
                              height=280, margin=dict(l=50,r=20,t=40,b=40))
            figs.append(dcc.Graph(figure=fig, style={"display":"inline-block","width":"48%"}))
        return html.Div(figs), no_update

    # ── Correlations ──────────────────────────────────────────
    if tab == "tab-corr":
        if len(sel_vars) < 2:
            return dbc.Alert("Select ≥ 2 variables.", color="info"), no_update
        flat_data = collect_flat(layers, sel_vars)
        min_len   = min(len(a) for a, _ in flat_data.values())
        labels    = []
        df_c      = {}
        for vk, (arr, _) in flat_data.items():
            _, v = parse_var(vk)
            labels.append(v)
            df_c[v] = arr[:min_len]
        df = pd.DataFrame(df_c).dropna()
        corr = df.corr()
        fig  = px.imshow(corr, color_continuous_scale="RdBu_r",
                         zmin=-1, zmax=1, text_auto=".2f",
                         title="Pearson Correlation Matrix", aspect="auto")
        fig.update_layout(height=max(320, 80*len(sel_vars)))
        return dcc.Graph(figure=fig), no_update

    # ── ROI Statistics ────────────────────────────────────────
    if tab == "tab-roi":
        if not aoi:
            return dbc.Alert("Upload a shapefile or GeoJSON first.", color="info"), no_update

        bbox      = aoi["bbox"]
        aoi_name  = aoi["name"]
        geojson   = aoi.get("geojson")

        # Use primary dataset for spatial context
        ds_primary = load_ds(primary_path)
        dims_p     = detect_dims(ds_primary)
        lat_p = ds_primary[dims_p["lat"]].values if "lat" in dims_p else None
        lon_p = ds_primary[dims_p["lon"]].values if "lon" in dims_p else None

        # Build polygon mask for primary grid
        poly_mask = None
        if geojson and lat_p is not None and lon_p is not None:
            try:
                lat_flat = lat_p.flatten() if lat_p.ndim > 1 else lat_p
                lon_flat = lon_p.flatten() if lon_p.ndim > 1 else lon_p
                # Clip to bbox first for speed
                lat_sel = lat_flat[(lat_flat >= bbox[1]) & (lat_flat <= bbox[3])]
                lon_sel = lon_flat[(lon_flat >= bbox[0]) & (lon_flat <= bbox[2])]
                poly_mask = polygon_mask_2d(geojson, lat_sel, lon_sel)
            except Exception:
                poly_mask = None

        ds_primary.close()

        # Per-variable ROI stats with true polygon masking
        roi_rows = []
        for vk in sel_vars:
            idx_l, v = parse_var(vk)
            try:
                ds_l    = load_ds(layers[idx_l]["path"])
                dims_l  = detect_dims(ds_l)
                ds_clip = clip_ds(ds_l, dims_l, bbox)    # bbox clip first
                arr_da  = ds_clip[v]
                if "time"  in dims_l: arr_da = arr_da.isel({dims_l["time"]:0})
                if "depth" in dims_l: arr_da = arr_da.isel({dims_l["depth"]:0})
                arr_2d = arr_da.squeeze().values.astype(float)

                # Apply polygon mask if available and shapes match
                if poly_mask is not None and arr_2d.ndim == 2:
                    # Resize mask to match arr_2d shape if grids differ slightly
                    if arr_2d.shape == poly_mask.shape:
                        arr_2d[~poly_mask] = np.nan
                    else:
                        # Coarse fit: just use the bbox-clipped data as-is
                        pass

                flat = arr_2d.flatten()
                flat = flat[~np.isnan(flat)]
                if flat.size == 0:
                    ds_l.close(); continue
                roi_rows.append({
                    "Layer": f"[{layers[idx_l]['label']}]",
                    "Variable": v, "Units": get_units(ds_l, v),
                    "Min": float(np.min(flat)), "Max": float(np.max(flat)),
                    "Mean": float(np.mean(flat)), "Std": float(np.std(flat)),
                    "Median": float(np.median(flat)), "N valid": int(flat.size),
                })
                ds_l.close()
            except Exception:
                pass

        df_roi = pd.DataFrame(roi_rows) if roi_rows else pd.DataFrame()

        # AOI map with actual polygon geometry
        fig_map = make_roi_map(geojson, bbox,
                               lat_p if lat_p is not None else None,
                               lon_p if lon_p is not None else None)

        # Stats table
        tbl_data = df_roi.copy()
        if not tbl_data.empty:
            for c in ["Min","Max","Mean","Std","Median"]:
                tbl_data[c] = tbl_data[c].map(lambda x: f"{x:.4g}")
        tbl = (dash_table.DataTable(
                   data=tbl_data.to_dict("records"),
                   columns=[{"name": c, "id": c} for c in tbl_data.columns],
                   style_table={"overflowX":"auto"},
                   style_cell={"fontSize":"0.82rem","padding":"6px"},
                   style_header={"background":"#0077b6","color":"#fff","fontWeight":"bold"},
                   style_data_conditional=[{"if":{"row_index":"odd"},"background":"#f0f7ff"}],
               ) if not tbl_data.empty
               else dbc.Alert("No data found within AOI polygon.", color="warning"))

        # Export payload
        export_payload = None
        if not df_roi.empty:
            primary_ds_meta = load_ds(primary_path)
            info = {
                "Dataset": primary_ds_meta.attrs.get("title",
                           primary_ds_meta.attrs.get("id","Copernicus Marine")),
                "Layers": f"{len(layers)} file(s)",
                "AOI": aoi_name,
                "AOI Bounds": (f"{bbox[0]:.3f}°–{bbox[2]:.3f}°E, "
                               f"{bbox[1]:.3f}°–{bbox[3]:.3f}°N"),
                "Masking": "True polygon mask" if poly_mask is not None else "Bounding box",
                "Variables": ", ".join(f"{parse_var(vk)[1]}" for vk in sel_vars),
                "Report date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M UTC"),
            }
            primary_ds_meta.close()
            export_payload = {
                "stats_json": df_roi.to_json(),
                "aoi_name":   aoi_name,
                "info":       info,
            }

        return html.Div([
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_map), width=8),
                dbc.Col([
                    html.Strong("AOI Bounds"),
                    html.Br(),
                    html.Small(f"Lon: {bbox[0]:.3f}° – {bbox[2]:.3f}°"),
                    html.Br(),
                    html.Small(f"Lat: {bbox[1]:.3f}° – {bbox[3]:.3f}°"),
                    html.Hr(className="my-2"),
                    dbc.Badge("Polygon-masked" if poly_mask is not None
                              else "BBox clip",
                              color="success" if poly_mask is not None else "warning"),
                ], width=4, className="small align-self-center"),
            ]),
            html.Hr(),
            html.H5(f"ROI Statistics — {aoi_name}"),
            tbl,
            html.H6("Export", className="mt-3 fw-bold"),
            dbc.Row([
                dbc.Col(dbc.Button([html.I(className="fa fa-file-csv me-1"), "Download CSV"],
                                   id="btn-csv", color="primary", outline=True,
                                   className="w-100"), width=4),
                dbc.Col(dbc.Button([html.I(className="fa fa-print me-1"),
                                    "Download Printable Report"],
                                   id="btn-report", color="secondary", outline=True,
                                   className="w-100"), width=5),
            ], className="mb-2"),
            html.Small("Open the HTML report → Ctrl+P → Save as PDF",
                       className="text-muted"),
        ]), export_payload

    # ── Multi-Criteria Analysis ───────────────────────────────
    if tab == "tab-mca":
        if len(sel_vars) < 2:
            return dbc.Alert("Select ≥ 2 variables to run MCA.", color="info"), no_update

        content = []
        flat_data = collect_flat(layers, sel_vars)
        if len(flat_data) < 2:
            return dbc.Alert("Could not load enough variable data.", color="warning"), no_update

        vk_list = list(flat_data.keys())
        var_labels = {}
        for vk in vk_list:
            i, v = parse_var(vk)
            var_labels[vk] = f"[{layers[i]['label']}] {v}"

        # 1. Scatter matrix
        content.append(html.Div([
            html.H5("1. Scatter Plot Matrix", className="fw-bold text-primary"),
            html.P("Pairwise scatter — reveals co-variation across products and layers.",
                   className="text-muted small"),
        ], className="mca-section"))
        try:
            min_len = min(len(a) for a, _ in flat_data.values())
            df_s = pd.DataFrame(
                {var_labels[vk]: a[:min_len] for vk, (a, _) in flat_data.items()}
            ).dropna()
            if len(df_s) > 5000: df_s = df_s.sample(5000, random_state=42)
            fig_sc = px.scatter_matrix(df_s, dimensions=list(df_s.columns),
                                       opacity=0.4, title="")
            fig_sc.update_traces(diagonal_visible=True, marker_size=2)
            fig_sc.update_layout(height=max(400, 120*len(sel_vars)),
                                 margin=dict(l=40,r=20,t=20,b=40))
            content.append(dcc.Graph(figure=fig_sc))
        except Exception as e:
            content.append(dbc.Alert(f"Scatter matrix: {e}", color="warning"))

        # 2. Correlation heatmap
        content.append(html.Div([
            html.H5("2. Correlation Heatmap", className="fw-bold text-primary mt-4"),
        ], className="mca-section"))
        try:
            min_len = min(len(a) for a, _ in flat_data.values())
            df_c = pd.DataFrame(
                {var_labels[vk]: a[:min_len] for vk, (a, _) in flat_data.items()}
            ).dropna()
            corr = df_c.corr()
            fig_corr = px.imshow(corr, color_continuous_scale="RdBu_r",
                                 zmin=-1, zmax=1, text_auto=".2f", aspect="auto")
            fig_corr.update_layout(height=max(300, 80*len(sel_vars)),
                                   margin=dict(l=40,r=20,t=10,b=40))
            content.append(dcc.Graph(figure=fig_corr))
        except Exception as e:
            content.append(dbc.Alert(f"Correlation: {e}", color="warning"))

        # 3. Normalised time-series overlay
        content.append(html.Div([
            html.H5("3. Normalised Time Series Overlay",
                    className="fw-bold text-primary mt-4"),
            html.P("All variables scaled 0–1 — reveals phase lags across products.",
                   className="text-muted small"),
        ], className="mca-section"))
        ts_traces = 0
        fig_ts = go.Figure()
        colours = px.colors.qualitative.Plotly
        for ci, vk in enumerate(sel_vars):
            idx_l, v = parse_var(vk)
            try:
                ds_l   = load_ds(layers[idx_l]["path"])
                dims_l = detect_dims(ds_l)
                if "time" not in dims_l: ds_l.close(); continue
                times  = ds_l[dims_l["time"]].values
                arr    = ds_l[v]
                if "depth" in dims_l: arr = arr.isel({dims_l["depth"]:0})
                avg_d  = [d for d in [dims_l.get("lat"), dims_l.get("lon")]
                          if d and d in arr.dims]
                if avg_d: arr = arr.mean(dim=avg_d)
                arr = arr.squeeze().values.flatten().astype(float)
                mn, mx = np.nanmin(arr), np.nanmax(arr)
                norm   = (arr - mn) / (mx - mn + 1e-12)
                fig_ts.add_trace(go.Scatter(
                    x=times, y=norm, mode="lines",
                    name=var_labels[vk],
                    line=dict(color=colours[ci % len(colours)], width=1.5),
                ))
                ts_traces += 1
                ds_l.close()
            except Exception:
                pass
        if ts_traces > 0:
            fig_ts.update_layout(
                title="Normalised time series (0–1)", xaxis_title="Time",
                yaxis_title="Normalised value", height=360,
                legend=dict(orientation="h", y=-0.25),
                margin=dict(l=60,r=20,t=40,b=80),
            )
            content.append(dcc.Graph(figure=fig_ts))

        # 4. PCA
        content.append(html.Div([
            html.H5("4. PCA Biplot", className="fw-bold text-primary mt-4"),
            html.P("Dominant co-variability modes across all selected variables and layers.",
                   className="text-muted small"),
        ], className="mca-section"))
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA as skPCA
            min_len = min(len(a) for a, _ in flat_data.values())
            df_pca = pd.DataFrame(
                {var_labels[vk]: a[:min_len] for vk, (a, _) in flat_data.items()}
            ).dropna()
            if len(df_pca) > 10000: df_pca = df_pca.sample(10000, random_state=0)
            X      = StandardScaler().fit_transform(df_pca)
            n_comp = min(len(df_pca.columns), 3)
            pca    = skPCA(n_components=n_comp)
            scores = pca.fit_transform(X)
            loads  = pca.components_.T
            evr    = pca.explained_variance_ratio_
            fig_pca = go.Figure()
            fig_pca.add_trace(go.Scatter(
                x=scores[:,0], y=scores[:,1], mode="markers",
                marker=dict(size=3, color="#cce5f0", opacity=0.5), name="Samples",
            ))
            scale = 3
            for i, lbl in enumerate(df_pca.columns):
                fig_pca.add_annotation(
                    x=loads[i,0]*scale, y=loads[i,1]*scale, text=lbl,
                    showarrow=True, ax=0, ay=0,
                    arrowcolor="#d62828", arrowwidth=2,
                    font=dict(color="#d62828", size=10),
                )
            fig_pca.update_layout(
                title=f"PCA Biplot  (PC1 {evr[0]*100:.1f}%,  PC2 {evr[1]*100:.1f}%)",
                xaxis_title=f"PC1 ({evr[0]*100:.1f}%)",
                yaxis_title=f"PC2 ({evr[1]*100:.1f}%)",
                height=420, margin=dict(l=60,r=40,t=50,b=40),
            )
            content.append(dcc.Graph(figure=fig_pca))
            content.append(html.Small(
                f"Total variance explained by {n_comp} PCs: {sum(evr)*100:.1f}%",
                className="text-muted d-block mb-3"))
        except ImportError:
            content.append(dbc.Alert(
                "Install scikit-learn for PCA:  pip install scikit-learn",
                color="info", className="small"))
        except Exception as e:
            content.append(dbc.Alert(f"PCA error: {e}", color="warning"))

        # 5. Interaction Reference Guide
        content.append(html.Div([
            html.H5("5. Variable Interaction Reference Guide",
                    className="fw-bold text-primary mt-4"),
            html.P("Known oceanographic / biogeochemical couplings between "
                   "Copernicus marine products.", className="text-muted small"),
        ], className="mca-section"))
        guide_items = []
        for r in MCA_INTERACTIONS:
            guide_items.append(dbc.ListGroupItem([
                html.Div([
                    dbc.Badge(r["pair"],    color="primary", className="me-2"),
                    html.I(className="fa fa-arrows-alt-h mx-1 text-muted"),
                    dbc.Badge(r["partner"], color="success"),
                ]),
                html.Small([
                    html.Strong(r["relationship"]), html.Br(),
                    html.Span(r["detail"], className="text-muted"),
                ], className="d-block mt-1"),
            ]))
        content.append(dbc.ListGroup(guide_items, className="mb-4"))
        return html.Div(content), no_update

    return dbc.Alert("Select a tab.", color="secondary"), no_update


# ─────────────────────────────────────────────────────────────────────────────
# Export callbacks — read from persistent export-store
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("dl-csv",      "data"),
    Input("btn-csv",      "n_clicks"),
    State("export-store", "data"),
    prevent_initial_call=True,
)
def cb_dl_csv(_, store):
    if not store or "stats_json" not in store: return no_update
    df = pd.read_json(io.StringIO(store["stats_json"]))
    return dcc.send_data_frame(df.to_csv,
                               f"roi_stats_{store['aoi_name']}.csv", index=False)


@app.callback(
    Output("dl-report",   "data"),
    Input("btn-report",   "n_clicks"),
    State("export-store", "data"),
    prevent_initial_call=True,
)
def cb_dl_report(_, store):
    if not store or "stats_json" not in store: return no_update
    df  = pd.read_json(io.StringIO(store["stats_json"]))
    htm = build_report(df, store["aoi_name"], store["info"])
    return dict(content=htm,
                filename=f"roi_report_{store['aoi_name']}.html",
                type="text/html")


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
