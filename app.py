"""
Copernicus Marine Data Visualiser
==================================
Standalone Dash app — NO Streamlit dependency
Run locally : python app.py
Deploy Render: start command → python app.py
"""

import base64, io, json, os, shutil, tempfile, uuid, warnings, zipfile
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

# ──────────────────────────────────────────────────────────────
# App init
# ──────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True,
    title="Copernicus Marine Visualiser",
)
server = app.server   # expose Flask for Gunicorn on Render
TMP_DIR = tempfile.mkdtemp(prefix="copernicus_")

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def detect_dims(ds):
    m = {}
    for d in ds.dims:
        dl = d.lower()
        if   dl in ("time","t"):                                              m["time"]  = d
        elif dl in ("depth","deptht","depthu","depthv","depthw","lev","level","z"): m["depth"] = d
        elif dl in ("latitude","lat","y","nav_lat"):                          m["lat"]   = d
        elif dl in ("longitude","lon","x","nav_lon"):                         m["lon"]   = d
    return m

def get_units(ds, var): return ds[var].attrs.get("units","")

def load_ds(path): return xr.open_dataset(path, decode_times=True)

def decode_upload(contents, filename):
    _, b64 = contents.split(",")
    data = base64.b64decode(b64)
    path = os.path.join(TMP_DIR, f"{uuid.uuid4().hex}_{filename}")
    with open(path,"wb") as f: f.write(data)
    return path

def clip_ds(ds, dims, bbox):
    minx, miny, maxx, maxy = bbox
    sel = {}
    if "lon" in dims:
        lv = ds[dims["lon"]].values
        sel[dims["lon"]] = (lv >= minx) & (lv <= maxx)
    if "lat" in dims:
        lv = ds[dims["lat"]].values
        sel[dims["lat"]] = (lv >= miny) & (lv <= maxy)
    return ds.isel({k:v for k,v in sel.items()}) if sel else ds

def roi_stats(ds, dims, selected_vars, bbox):
    ds_c = clip_ds(ds, dims, bbox)
    rows = []
    for var in selected_vars:
        try:
            arr = ds_c[var].values.flatten().astype(float)
            arr = arr[~np.isnan(arr)]
            if arr.size == 0: continue
            rows.append({"Variable":var,"Units":get_units(ds,var),
                         "Min":float(np.min(arr)),"Max":float(np.max(arr)),
                         "Mean":float(np.mean(arr)),"Std":float(np.std(arr)),
                         "Median":float(np.median(arr)),"N valid":int(arr.size)})
        except: pass
    return pd.DataFrame(rows), ds_c

def build_report(stats_df, aoi_name, info):
    rows = "".join(
        f"<tr><td>{r['Variable']}</td><td>{r.get('Units','')}</td>"
        f"<td>{r['Min']:.4g}</td><td>{r['Max']:.4g}</td>"
        f"<td>{r['Mean']:.4g}</td><td>{r['Std']:.4g}</td>"
        f"<td>{r['Median']:.4g}</td><td>{r['N valid']:,}</td></tr>"
        for _,r in stats_df.iterrows()
    )
    irows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k,v in info.items())
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
.credit{{margin-top:40px;padding:12px;background:#eaf4fb;border-left:4px solid #0077b6;font-size:.85rem}}
.footer{{margin-top:30px;font-size:.8rem;color:#6c757d;border-top:1px solid #dee2e6;padding-top:10px}}
@media print{{button{{display:none}}}}
</style></head><body>
<h1>🌊 Copernicus Marine — ROI Statistics Report</h1>
<p><strong>Area of Interest:</strong> {aoi_name}</p>
<h2>Dataset Information</h2>
<table><tr><th>Property</th><th>Value</th></tr>{irows}</table>
<h2>ROI Statistics</h2>
<table><tr><th>Variable</th><th>Units</th><th>Min</th><th>Max</th>
<th>Mean</th><th>Std Dev</th><th>Median</th><th>Valid Points</th></tr>{rows}</table>
<div class="credit"><strong>Data Credit:</strong><br>
Generated using E.U. Copernicus Marine Service Information;
<a href="https://marine.copernicus.eu">https://marine.copernicus.eu</a></div>
<div class="footer">Maintained by <strong>Canopy Geospatial Solutions</strong> ·
Report generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M UTC')}
<button onclick="window.print()" style="float:right;padding:6px 16px;
background:#0077b6;color:#fff;border:none;border-radius:4px;cursor:pointer">
🖨️ Print / Save as PDF</button></div></body></html>"""

# ──────────────────────────────────────────────────────────────
# Reusable UI components
# ──────────────────────────────────────────────────────────────

def section(title, body):
    return dbc.Card([
        dbc.CardHeader(html.Strong(title)),
        dbc.CardBody(body),
    ], className="mb-3 shadow-sm")

SIDEBAR = dbc.Col([
    html.H5("🌊 Copernicus Marine", className="fw-bold text-primary mt-2 mb-3"),

    # ── Data source ──
    section("📂 Data Source", [
        dbc.RadioItems(
            id="source-radio",
            options=[
                {"label": " Upload NC file",              "value": "upload"},
                {"label": " Enter file path",             "value": "path"},
                {"label": " Download from Copernicus",    "value": "download"},
            ],
            value="upload", className="mb-2",
        ),

        # Upload
        html.Div(id="div-upload", children=[
            dcc.Upload(
                id="upload-nc",
                children=html.Div(["Drag & drop or ", html.A("browse")]),
                style={"borderWidth":"2px","borderStyle":"dashed","borderRadius":"6px",
                       "textAlign":"center","padding":"18px","cursor":"pointer",
                       "borderColor":"#0077b6","color":"#0077b6","fontSize":"0.85rem"},
                multiple=False,
                max_size=-1,     # no size limit
            ),
            html.Small("No upload size limit on Render", className="text-muted"),
        ]),

        # Path
        html.Div(id="div-path", style={"display":"none"}, children=[
            dbc.Input(id="path-input", placeholder=r"D:\data\file.nc", type="text"),
            dbc.Button("Load", id="btn-path", color="primary", size="sm", className="mt-2 w-100"),
        ]),

        # Download
        html.Div(id="div-download", style={"display":"none"}, children=[
            dbc.Input(id="cm-user",    placeholder="Username",  type="text",     className="mb-1"),
            dbc.Input(id="cm-pass",    placeholder="Password",  type="password", className="mb-1"),
            dbc.Input(id="cm-dataset", value="cmems_mod_glo_bgc-car_anfc_0.25deg_P1D-m",
                      placeholder="Dataset ID", className="mb-1"),
            dbc.Input(id="cm-version", value="202311",   placeholder="Version (optional)", className="mb-1"),
            dbc.Input(id="cm-vars",    value="dissic,ph,talk", placeholder="Variables (comma-sep)", className="mb-1"),
            dbc.Row([
                dbc.Col(dbc.Input(id="cm-minlon", type="number", value=73.036,  placeholder="Min Lon"), width=6),
                dbc.Col(dbc.Input(id="cm-maxlon", type="number", value=73.036,  placeholder="Max Lon"), width=6),
            ], className="mb-1"),
            dbc.Row([
                dbc.Col(dbc.Input(id="cm-minlat", type="number", value=11.968,  placeholder="Min Lat"), width=6),
                dbc.Col(dbc.Input(id="cm-maxlat", type="number", value=11.968,  placeholder="Max Lat"), width=6),
            ], className="mb-1"),
            dbc.Input(id="cm-start", value="2023-07-01T00:00:00", className="mb-1"),
            dbc.Input(id="cm-end",   value="2026-07-01T00:00:00", className="mb-1"),
            dbc.Row([
                dbc.Col(dbc.Input(id="cm-mindep", type="number", value=0.494,    placeholder="Min Depth"), width=6),
                dbc.Col(dbc.Input(id="cm-maxdep", type="number", value=5727.92,  placeholder="Max Depth"), width=6),
            ], className="mb-1"),
            dbc.Button("⬇️ Download", id="btn-download", color="primary", className="w-100 mt-1"),
        ]),

        html.Div(id="source-status", className="mt-2"),
    ]),

    # ── AOI ──
    section("📐 Area of Interest", [
        dcc.Upload(
            id="upload-aoi",
            children=html.Div(["Upload .zip (shapefile) or .geojson"]),
            style={"borderWidth":"2px","borderStyle":"dashed","borderRadius":"6px",
                   "textAlign":"center","padding":"12px","cursor":"pointer",
                   "borderColor":"#28a745","color":"#28a745","fontSize":"0.8rem"},
            multiple=False,
        ),
        html.Div(id="aoi-status", className="mt-1 small"),
    ]),

    # ── Variable selector ──
    section("📊 Variables", [
        dcc.Dropdown(id="var-selector", multi=True, placeholder="Load a dataset first…"),
    ]),

    # ── Footer ──
    html.Hr(),
    html.Div([
        html.Small(" "),
        html.Strong("Canopy Geospatial Solutions"),
        html.Br(),
        html.Small("Generated using "),
        html.Small(html.B("E.U. Copernicus Marine Service Information")),
        html.Br(),
        html.A("marine.copernicus.eu", href="https://marine.copernicus.eu",
               target="_blank", className="small"),
    ], className="text-muted small lh-lg"),

], width=3, className="bg-light border-end px-3 py-3", style={"minHeight":"100vh"})

# ──────────────────────────────────────────────────────────────
# Layout
# ──────────────────────────────────────────────────────────────
app.layout = dbc.Container([

    # Hidden stores
    dcc.Store(id="store-nc-path"),
    dcc.Store(id="store-dims"),
    dcc.Store(id="store-aoi"),
    dcc.Download(id="dl-csv"),
    dcc.Download(id="dl-report"),
    dcc.Interval(id="interval", interval=999999, n_intervals=0),

    # Header
    dbc.Row(dbc.Col(html.Div([
        html.H3("🌊 Copernicus Marine Data Visualiser",
                className="mb-0 fw-bold text-primary"),
        html.Small("Nutrients · Carbon · Physics · Any Copernicus NC file",
                   className="text-muted"),
    ], className="py-3 border-bottom")), className="mb-0"),

    # Body
    dbc.Row([
        SIDEBAR,
        dbc.Col([
            dbc.Tabs(id="main-tabs", active_tab="tab-info", children=[
                dbc.Tab(label="📋 Dataset Info",    tab_id="tab-info"),
                dbc.Tab(label="📊 Statistics",      tab_id="tab-stats"),
                dbc.Tab(label="🗺️ Surface Maps",    tab_id="tab-surface"),
                dbc.Tab(label="📉 Depth Profiles",  tab_id="tab-depth"),
                dbc.Tab(label="📈 Time Series",     tab_id="tab-ts"),
                dbc.Tab(label="🌐 Zonal Means",     tab_id="tab-zonal"),
                dbc.Tab(label="⬇️ Depth Slices",    tab_id="tab-slices"),
                dbc.Tab(label="📊 Histograms",      tab_id="tab-hist"),
                dbc.Tab(label="🔗 Correlations",    tab_id="tab-corr"),
                dbc.Tab(label="📍 ROI Statistics",  tab_id="tab-roi"),
            ], className="mt-2"),
            html.Div(id="tab-content", className="mt-3"),
        ], width=9, className="px-3 py-2"),
    ], className="g-0"),

    # Page footer
    dbc.Row(dbc.Col(html.Div([
        html.Hr(className="mt-4"),
        html.P([
            " ", html.Strong("Maintained and Hosted by Canopy Geospatial Solutions"),
            " | Generated using ",
            html.Strong("E.U. Copernicus Marine Service Information"),
            " · ",
            html.A("marine.copernicus.eu", href="https://marine.copernicus.eu", target="_blank"),
        ], className="text-center text-muted small py-2 mb-0"),
    ]))),

], fluid=True)

# ──────────────────────────────────────────────────────────────
# Callbacks — sidebar controls
# ──────────────────────────────────────────────────────────────

@app.callback(
    Output("div-upload",   "style"),
    Output("div-path",     "style"),
    Output("div-download", "style"),
    Input("source-radio",  "value"),
)
def toggle_source(val):
    show = {"display": "block"}
    hide = {"display": "none"}
    return (
        show if val == "upload"   else hide,
        show if val == "path"     else hide,
        show if val == "download" else hide,
    )


@app.callback(
    Output("store-nc-path", "data",  allow_duplicate=True),
    Output("source-status", "children", allow_duplicate=True),
    Input("upload-nc", "contents"),
    State("upload-nc", "filename"),
    prevent_initial_call=True,
)
def cb_upload(contents, filename):
    if not contents:
        return no_update, no_update
    try:
        path = decode_upload(contents, filename)
        ds = load_ds(path); ds.close()
        return path, dbc.Alert(f"✅ Loaded: {filename}", color="success", className="py-1 mb-0 small")
    except Exception as e:
        return no_update, dbc.Alert(f"❌ {e}", color="danger", className="py-1 mb-0 small")


@app.callback(
    Output("store-nc-path", "data",  allow_duplicate=True),
    Output("source-status", "children", allow_duplicate=True),
    Input("btn-path", "n_clicks"),
    State("path-input", "value"),
    prevent_initial_call=True,
)
def cb_path(_, path):
    if not path: return no_update, no_update
    if not os.path.isfile(path):
        return no_update, dbc.Alert("❌ File not found.", color="danger", className="py-1 mb-0 small")
    try:
        ds = load_ds(path); ds.close()
        return path, dbc.Alert(f"✅ Loaded: {os.path.basename(path)}", color="success", className="py-1 mb-0 small")
    except Exception as e:
        return no_update, dbc.Alert(f"❌ {e}", color="danger", className="py-1 mb-0 small")


@app.callback(
    Output("store-nc-path", "data",  allow_duplicate=True),
    Output("source-status", "children", allow_duplicate=True),
    Input("btn-download", "n_clicks"),
    State("cm-user","value"), State("cm-pass","value"),
    State("cm-dataset","value"), State("cm-version","value"),
    State("cm-vars","value"),
    State("cm-minlon","value"), State("cm-maxlon","value"),
    State("cm-minlat","value"), State("cm-maxlat","value"),
    State("cm-start","value"),  State("cm-end","value"),
    State("cm-mindep","value"), State("cm-maxdep","value"),
    prevent_initial_call=True,
)
def cb_cm_download(_, user, pwd, dataset_id, version, vars_raw,
                   minlon, maxlon, minlat, maxlat, start, end, mindep, maxdep):
    if not user or not pwd:
        return no_update, dbc.Alert("Enter Copernicus credentials.", color="warning", className="py-1 mb-0 small")
    try:
        import copernicusmarine, inspect
        os.environ["COPERNICUSMARINE_SERVICE_USERNAME"] = user
        os.environ["COPERNICUSMARINE_SERVICE_PASSWORD"] = pwd
        variables = [v.strip() for v in vars_raw.split(",") if v.strip()]
        out_dir  = tempfile.mkdtemp()
        out_file = os.path.join(out_dir, "copernicus_data.nc")
        sig = inspect.signature(copernicusmarine.subset).parameters
        kw  = dict(dataset_id=dataset_id, variables=variables,
                   minimum_longitude=minlon, maximum_longitude=maxlon,
                   minimum_latitude=minlat,  maximum_latitude=maxlat,
                   start_datetime=start, end_datetime=end,
                   minimum_depth=mindep,  maximum_depth=maxdep,
                   disable_progress_bar=True)
        if version: kw["dataset_version"] = version
        if "netcdf_compression_level" in sig: kw["netcdf_compression_level"] = 1
        if "output_filename"          in sig: kw["output_filename"] = out_file
        elif "output_directory"       in sig: kw["output_directory"] = out_dir
        result = copernicusmarine.subset(**kw)
        # Resolve output path
        nc_path = None
        if isinstance(result, str) and result.endswith(".nc"):
            nc_path = result
        elif hasattr(result, "file_path"):
            nc_path = str(result.file_path)
        else:
            for f in os.listdir(out_dir):
                if f.endswith(".nc"):
                    nc_path = os.path.join(out_dir, f); break
        if nc_path and os.path.isfile(nc_path):
            return nc_path, dbc.Alert(f"✅ Downloaded: {os.path.basename(nc_path)}", color="success", className="py-1 mb-0 small")
        return no_update, dbc.Alert("Download finished but no NC file found.", color="warning", className="py-1 mb-0 small")
    except ImportError:
        return no_update, dbc.Alert("copernicusmarine not installed. Run: pip install copernicusmarine", color="danger", className="py-1 mb-0 small")
    except Exception as e:
        return no_update, dbc.Alert(f"❌ {e}", color="danger", className="py-1 mb-0 small")


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
        try: import pyogrio; engine = "pyogrio"
        except ImportError: engine = "fiona"
        _, b64 = contents.split(",")
        data = base64.b64decode(b64)
        tmp = tempfile.mkdtemp()
        fn = filename.lower()
        if fn.endswith(".zip"):
            zp = os.path.join(tmp,"aoi.zip")
            with open(zp,"wb") as f: f.write(data)
            with zipfile.ZipFile(zp,"r") as z: z.extractall(tmp)
            shps = [os.path.join(tmp,x) for x in os.listdir(tmp) if x.endswith(".shp")]
            if not shps: raise ValueError("No .shp inside ZIP.")
            gdf = gpd.read_file(shps[0], engine=engine)
        else:
            gp = os.path.join(tmp,"aoi.geojson")
            with open(gp,"wb") as f: f.write(data)
            gdf = gpd.read_file(gp, engine=engine)
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        b = gdf.total_bounds.tolist()
        name = filename.rsplit(".",1)[0]
        aoi_data = {"bbox": b, "name": name}
        msg = dbc.Alert(
            f"✅ {name} | {b[0]:.2f}°–{b[2]:.2f}°E, {b[1]:.2f}°–{b[3]:.2f}°N",
            color="success", className="py-1 mb-0"
        )
        return aoi_data, msg
    except Exception as e:
        return no_update, dbc.Alert(f"❌ {e}", color="danger", className="py-1 mb-0")


@app.callback(
    Output("var-selector", "options"),
    Output("var-selector", "value"),
    Output("store-dims",   "data"),
    Input("store-nc-path", "data"),
)
def cb_vars(path):
    if not path: return [], [], {}
    ds = load_ds(path)
    dims  = detect_dims(ds)
    vars_ = list(ds.data_vars)
    ds.close()
    opts = [{"label": v, "value": v} for v in vars_]
    return opts, vars_[:min(6,len(vars_))], dims

# ──────────────────────────────────────────────────────────────
# Main tab-content callback
# ──────────────────────────────────────────────────────────────

@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs",    "active_tab"),
    Input("store-nc-path","data"),
    Input("var-selector", "value"),
    Input("store-dims",   "data"),
    Input("store-aoi",    "data"),
)
def render_tab(tab, path, sel_vars, dims, aoi):
    if not path:
        return dbc.Alert("👈  Load a dataset using the sidebar.", color="info")
    ds = load_ds(path)
    if not sel_vars: sel_vars = list(ds.data_vars)[:6]
    if not dims: dims = detect_dims(ds)

    has_lat  = "lat"   in dims
    has_lon  = "lon"   in dims
    has_time = "time"  in dims
    has_dep  = "depth" in dims
    is_spatial = has_lat and has_lon
    is_point   = not is_spatial

    # ── Dataset Info ──────────────────────────────────────────
    if tab == "tab-info":
        rows = [{"Variable": v,
                 "Long name": ds[v].attrs.get("long_name", ds[v].attrs.get("standard_name","")),
                 "Units":     ds[v].attrs.get("units",""),
                 "Shape":     str(ds[v].shape),
                 "Dims":      str(ds[v].dims)}
                for v in ds.data_vars]
        tbl = dash_table.DataTable(
            data=rows, columns=[{"name":c,"id":c} for c in rows[0].keys()],
            style_table={"overflowX":"auto"},
            style_cell={"fontSize":"0.82rem","padding":"6px"},
            style_header={"background":"#0077b6","color":"#fff","fontWeight":"bold"},
            style_data_conditional=[{"if":{"row_index":"odd"},"background":"#f0f7ff"}],
            page_size=20,
        )
        dim_cards = [dbc.Badge(f"{k}: {v}", color="primary", className="me-1 mb-1")
                     for k,v in ds.dims.items()]
        globs = [dbc.ListGroupItem(f"{k}: {v}") for k,v in ds.attrs.items()] or [dbc.ListGroupItem("No global attributes.")]
        ds.close()
        return html.Div([
            dbc.Row([
                dbc.Col(dbc.Card([dbc.CardHeader("Dimensions"), dbc.CardBody(dim_cards)]), width=6),
                dbc.Col(dbc.Card([dbc.CardHeader("Global Attributes"),
                                  dbc.CardBody(dbc.ListGroup(globs, flush=True, style={"fontSize":"0.82rem","maxHeight":"180px","overflowY":"auto"}))]), width=6),
            ], className="mb-3"),
            dbc.Card([dbc.CardHeader("Variable Metadata"), dbc.CardBody(tbl)]),
        ])

    # ── Statistics ────────────────────────────────────────────
    if tab == "tab-stats":
        stats = []
        for var in sel_vars:
            arr = ds[var].values.flatten().astype(float)
            arr = arr[~np.isnan(arr)]
            if arr.size == 0: continue
            stats.append({"Variable":var,"Units":get_units(ds,var),
                           "Min":f"{np.min(arr):.4g}","Max":f"{np.max(arr):.4g}",
                           "Mean":f"{np.mean(arr):.4g}","Std":f"{np.std(arr):.4g}",
                           "Median":f"{np.median(arr):.4g}","N valid":f"{arr.size:,}"})
        ds.close()
        if not stats: return dbc.Alert("No data.", color="warning")
        tbl = dash_table.DataTable(
            data=stats, columns=[{"name":c,"id":c} for c in stats[0].keys()],
            style_table={"overflowX":"auto"},
            style_cell={"fontSize":"0.83rem","padding":"6px"},
            style_header={"background":"#0077b6","color":"#fff","fontWeight":"bold"},
            style_data_conditional=[{"if":{"row_index":"odd"},"background":"#f0f7ff"}],
            export_format="csv",
        )
        return html.Div([html.H5("Global Statistics"), tbl,
                         html.Small("Click the export button (top-right of table) to download CSV.", className="text-muted")])

    # ── Surface Maps ──────────────────────────────────────────
    if tab == "tab-surface":
        if not is_spatial:
            ds.close(); return dbc.Alert("No lat/lon dimensions — use Time Series tab.", color="info")
        plots = []
        for var in sel_vars:
            try:
                arr = ds[var]
                kw  = {}
                if has_time:  kw[dims["time"]]  = 0
                if has_dep:   kw[dims["depth"]] = 0
                surf = arr.isel(kw).squeeze()
                lat  = surf[dims["lat"]].values
                lon  = surf[dims["lon"]].values
                z    = surf.values
                fig  = go.Figure(go.Heatmap(z=z, x=lon, y=lat, colorscale="Viridis",
                                            colorbar={"title":get_units(ds,var)}))
                fig.update_layout(title=f"{var} — surface", height=350,
                                  xaxis_title="Longitude", yaxis_title="Latitude",
                                  margin=dict(l=50,r=20,t=40,b=40))
                plots.append(dcc.Graph(figure=fig))
            except Exception as e:
                plots.append(dbc.Alert(f"{var}: {e}", color="warning"))
        ds.close()
        return html.Div(plots)

    # ── Depth Profiles ────────────────────────────────────────
    if tab == "tab-depth":
        if not has_dep:
            ds.close(); return dbc.Alert("No depth dimension found.", color="info")
        depth_dim  = dims["depth"]
        depth_vals = ds[depth_dim].values
        figs = []
        for var in sel_vars:
            try:
                arr = ds[var]
                kw  = {}
                if has_time: kw[dims["time"]] = 0
                if is_spatial:
                    lat_d = [d for d in [dims.get("lat"),dims.get("lon")] if d and d in arr.dims]
                    arr = arr.mean(dim=lat_d)
                arr = arr.isel(kw).squeeze() if kw else arr.squeeze()
                fig = go.Figure(go.Scatter(x=arr.values.flatten(), y=depth_vals,
                                           mode="lines", line={"color":"teal","width":2}))
                fig.update_yaxes(autorange="reversed")
                fig.update_layout(title=f"{var} — vertical profile", height=350,
                                  xaxis_title=f"{var} [{get_units(ds,var)}]",
                                  yaxis_title="Depth (m)",
                                  margin=dict(l=60,r=20,t=40,b=40))
                figs.append(dcc.Graph(figure=fig, style={"display":"inline-block","width":"48%"}))
            except Exception as e:
                figs.append(dbc.Alert(f"{var}: {e}", color="warning"))
        ds.close()
        return html.Div(figs)

    # ── Time Series ───────────────────────────────────────────
    if tab == "tab-ts":
        if not has_time:
            ds.close(); return dbc.Alert("No time dimension found.", color="info")
        figs = []
        for var in sel_vars:
            try:
                arr = ds[var]
                if has_dep:
                    arr = arr.isel({dims["depth"]: 0})
                arr = arr.squeeze()
                times = ds[dims["time"]].values
                fig = go.Figure(go.Scatter(x=times, y=arr.values.flatten(),
                                           mode="lines", line={"color":"steelblue","width":1.5}))
                fig.update_layout(title=f"{var} — time series", height=300,
                                  xaxis_title="Time",
                                  yaxis_title=f"{var} [{get_units(ds,var)}]",
                                  margin=dict(l=60,r=20,t=40,b=40))
                figs.append(dcc.Graph(figure=fig))
            except Exception as e:
                figs.append(dbc.Alert(f"{var}: {e}", color="warning"))
        ds.close()
        return html.Div(figs)

    # ── Zonal Means ───────────────────────────────────────────
    if tab == "tab-zonal":
        if not (is_spatial and has_dep):
            ds.close(); return dbc.Alert("Zonal means require lat/lon and depth dimensions.", color="info")
        plots = []
        for var in sel_vars:
            try:
                arr = ds[var]
                if has_time: arr = arr.isel({dims["time"]: 0})
                zonal = arr.mean(dim=dims["lon"]).squeeze()
                lat   = zonal[dims["lat"]].values
                dep   = zonal[dims["depth"]].values
                fig   = go.Figure(go.Heatmap(z=zonal.values, x=lat, y=dep,
                                             colorscale="Turbo",
                                             colorbar={"title":get_units(ds,var)}))
                fig.update_yaxes(autorange="reversed")
                fig.update_layout(title=f"{var} — zonal mean", height=380,
                                  xaxis_title="Latitude", yaxis_title="Depth (m)",
                                  margin=dict(l=60,r=20,t=40,b=40))
                plots.append(dcc.Graph(figure=fig))
            except Exception as e:
                plots.append(dbc.Alert(f"{var}: {e}", color="warning"))
        ds.close()
        return html.Div(plots)

    # ── Depth Slices ──────────────────────────────────────────
    if tab == "tab-slices":
        if not (is_spatial and has_dep):
            ds.close(); return dbc.Alert("Depth slices require lat/lon and depth dimensions.", color="info")
        depth_dim  = dims["depth"]
        depth_vals = ds[depth_dim].values
        n_dep = len(depth_vals)
        plots = []
        for idx in range(min(5, n_dep)):
            try:
                kw = {depth_dim: idx}
                if has_time: kw[dims["time"]] = 0
                var = sel_vars[0]
                layer = ds[var].isel(kw).squeeze()
                lat   = layer[dims["lat"]].values
                lon   = layer[dims["lon"]].values
                fig   = go.Figure(go.Heatmap(z=layer.values, x=lon, y=lat,
                                             colorscale="Turbo",
                                             colorbar={"title":get_units(ds,var)}))
                fig.update_layout(title=f"{var} @ {depth_vals[idx]:.1f} m", height=320,
                                  xaxis_title="Longitude", yaxis_title="Latitude",
                                  margin=dict(l=50,r=20,t=40,b=40))
                plots.append(dcc.Graph(figure=fig))
            except Exception as e:
                plots.append(dbc.Alert(f"Depth {idx}: {e}", color="warning"))
        ds.close()
        return html.Div(plots)

    # ── Histograms ────────────────────────────────────────────
    if tab == "tab-hist":
        figs = []
        for var in sel_vars:
            try:
                vals = ds[var].values.flatten().astype(float)
                vals = vals[~np.isnan(vals)]
                fig  = go.Figure(go.Histogram(x=vals, nbinsx=80,
                                              marker_color="steelblue", opacity=0.85))
                fig.update_layout(title=f"{var} distribution", height=280,
                                  xaxis_title=f"[{get_units(ds,var)}]",
                                  yaxis_title="Count",
                                  margin=dict(l=50,r=20,t=40,b=40))
                figs.append(dcc.Graph(figure=fig, style={"display":"inline-block","width":"48%"}))
            except Exception as e:
                figs.append(dbc.Alert(f"{var}: {e}", color="warning"))
        ds.close()
        return html.Div(figs)

    # ── Correlations ──────────────────────────────────────────
    if tab == "tab-corr":
        if len(sel_vars) < 2:
            ds.close(); return dbc.Alert("Select at least 2 variables.", color="info")
        try:
            sample = {v: ds[v].values.flatten().astype(float) for v in sel_vars}
            min_len = min(len(v) for v in sample.values())
            df_c = pd.DataFrame({k: v[:min_len] for k,v in sample.items()}).dropna()
            corr = df_c.corr()
            fig  = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                             text_auto=".2f", title="Pearson Correlation Matrix",
                             aspect="auto")
            fig.update_layout(height=max(300, 80*len(sel_vars)))
            ds.close()
            return dcc.Graph(figure=fig)
        except Exception as e:
            ds.close(); return dbc.Alert(f"Correlation error: {e}", color="danger")

    # ── ROI Statistics ────────────────────────────────────────
    if tab == "tab-roi":
        if not aoi:
            ds.close(); return dbc.Alert("Upload a shapefile or GeoJSON in the sidebar first.", color="info")
        bbox     = aoi["bbox"]
        aoi_name = aoi["name"]
        df_stats, ds_c = roi_stats(ds, dims, sel_vars, bbox)
        if df_stats.empty:
            ds.close(); return dbc.Alert("No data within the AOI extent.", color="warning")

        # AOI map
        try:
            fig_map = go.Figure()
            if is_spatial:
                lon_v = ds[dims["lon"]].values
                lat_v = ds[dims["lat"]].values
                fig_map.add_shape(type="rect",
                    x0=float(lon_v.min()), x1=float(lon_v.max()),
                    y0=float(lat_v.min()), y1=float(lat_v.max()),
                    line={"color":"#0077b6","width":2}, fillcolor="rgba(0,119,182,0.1)")
            fig_map.add_shape(type="rect",
                x0=bbox[0], x1=bbox[2], y0=bbox[1], y1=bbox[3],
                line={"color":"#d62828","width":2}, fillcolor="rgba(214,40,40,0.15)")
            pad = max(bbox[2]-bbox[0], bbox[3]-bbox[1]) * 0.4 + 1
            fig_map.update_layout(
                title=f"Study Area: {aoi_name}",
                xaxis={"range":[bbox[0]-pad, bbox[2]+pad], "title":"Longitude"},
                yaxis={"range":[bbox[1]-pad, bbox[3]+pad], "title":"Latitude"},
                height=300, margin=dict(l=50,r=20,t=40,b=40),
            )
        except Exception: fig_map = go.Figure()

        # Stats table
        tbl_data = df_stats.copy()
        for c in ["Min","Max","Mean","Std","Median"]:
            tbl_data[c] = tbl_data[c].map(lambda x: f"{x:.4g}")
        tbl = dash_table.DataTable(
            data=tbl_data.to_dict("records"),
            columns=[{"name":c,"id":c} for c in tbl_data.columns],
            style_table={"overflowX":"auto"},
            style_cell={"fontSize":"0.83rem","padding":"6px"},
            style_header={"background":"#0077b6","color":"#fff","fontWeight":"bold"},
            style_data_conditional=[{"if":{"row_index":"odd"},"background":"#f0f7ff"}],
        )

        # ROI time series if applicable
        ts_plots = []
        if has_time:
            for var in sel_vars[:4]:
                try:
                    arr = ds_c[var]
                    if has_dep: arr = arr.isel({dims["depth"]: 0})
                    arr = arr.squeeze()
                    times = ds_c[dims["time"]].values
                    fig_ts = go.Figure(go.Scatter(x=times, y=arr.values.flatten(),
                                                  mode="lines", line={"color":"#d62828","width":1.5}))
                    fig_ts.update_layout(title=f"{var} — ROI time series", height=260,
                                         xaxis_title="Time",
                                         yaxis_title=f"{var} [{get_units(ds,var)}]",
                                         margin=dict(l=60,r=20,t=40,b=40))
                    ts_plots.append(dcc.Graph(figure=fig_ts))
                except: pass

        # Export section (CSV + HTML report)
        ds_info = {
            "Dataset":   ds.attrs.get("title", ds.attrs.get("id","Copernicus Marine")),
            "AOI":       aoi_name,
            "AOI Bounds":f"{bbox[0]:.3f}° – {bbox[2]:.3f}°E, {bbox[1]:.3f}° – {bbox[3]:.3f}°N",
            "Variables": ", ".join(sel_vars),
            "Date":      pd.Timestamp.now().strftime("%Y-%m-%d %H:%M UTC"),
        }

        ds.close()
        return html.Div([
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_map), width=7),
                dbc.Col([
                    html.Strong("AOI Bounds"),
                    html.Br(),
                    html.Small(f"Lon: {bbox[0]:.3f}° – {bbox[2]:.3f}°"),
                    html.Br(),
                    html.Small(f"Lat: {bbox[1]:.3f}° – {bbox[3]:.3f}°"),
                    html.Hr(className="my-2"),
                    html.Strong("Clipped dataset dims"),
                    html.Br(),
                    html.Small(str(dict(ds_c.dims))),
                ], width=5, className="small"),
            ]),
            html.Hr(),
            html.H5(f"ROI Statistics — {aoi_name}"),
            tbl,
            html.Hr(),
            *ts_plots,
            html.H6("Export", className="mt-3"),
            dbc.Row([
                dbc.Col(dbc.Button("⬇️ Download CSV",            id="btn-csv",    color="primary",   outline=True, className="w-100"), width=4),
                dbc.Col(dbc.Button("🖨️ Download Printable Report", id="btn-report", color="secondary", outline=True, className="w-100"), width=5),
            ]),
            html.Small("Open the HTML report in a browser → Ctrl+P → Save as PDF", className="text-muted"),
            # hidden store for export data
            dcc.Store(id="export-store", data={
                "stats_json": df_stats.to_json(),
                "aoi_name":   aoi_name,
                "info":       ds_info,
            }),
        ])

    ds.close()
    return dbc.Alert("Select a tab.", color="secondary")

# ──────────────────────────────────────────────────────────────
# Export callbacks
# ──────────────────────────────────────────────────────────────

@app.callback(
    Output("dl-csv",    "data"),
    Input("btn-csv",    "n_clicks"),
    State("export-store","data"),
    prevent_initial_call=True,
)
def cb_dl_csv(_, store):
    if not store: return no_update
    df = pd.read_json(store["stats_json"])
    return dcc.send_data_frame(df.to_csv, f"roi_stats_{store['aoi_name']}.csv", index=False)


@app.callback(
    Output("dl-report", "data"),
    Input("btn-report", "n_clicks"),
    State("export-store","data"),
    prevent_initial_call=True,
)
def cb_dl_report(_, store):
    if not store: return no_update
    df   = pd.read_json(store["stats_json"])
    html_content = build_report(df, store["aoi_name"], store["info"])
    return dict(content=html_content,
                filename=f"roi_report_{store['aoi_name']}.html",
                type="text/html")

# ──────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
