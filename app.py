import json, math, pathlib
import streamlit as st
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Pipe Risk Maps", layout="wide")

# ---------- Settings ----------
DATA_DIR = pathlib.Path("data")
FILES = {
    "Poisson Regression": DATA_DIR / "poisson_pred_pipa.geojson",
    "Gradient Boosted Trees (GBT)": DATA_DIR / "gbt_pred_pipa.geojson",
}
PALETTE = {
    "Very Low":       "#7fc97f",
    "Low":            "#beed90",
    "Moderate":       "#ffff99",
    "Moderately High":"#fdc86e",
    "High":           "#fdae61",
    "Very High":      "#d7191c"
}

# ---------- Helpers ----------
def get_bounds(geojson):
    minlat=minlon= math.inf
    maxlat=maxlon=-math.inf
    for feat in geojson["features"]:
        geom = feat.get("geometry", {})
        if geom.get("type") == "LineString":
            coords = geom.get("coordinates", [])
        elif geom.get("type") == "MultiLineString":
            coords = [pt for line in geom.get("coordinates", []) for pt in line]
        else:
            continue
        for lon, lat in coords:
            minlat, maxlat = min(minlat, lat), max(maxlat, lat)
            minlon, maxlon = min(minlon, lon), max(maxlon, lon)
    center = [(minlat+maxlat)/2, (minlon+maxlon)/2]
    bounds = [[minlat, minlon], [maxlat, maxlon]]
    return center, bounds

def build_map(gj, mode):
    # Tentukan field berdasarkan mode
    if mode == "Poisson Regression":
        bucket_field = "risk_bucket"
        p_field      = "p_ge1"
        lam_field    = "lambda_hat"
        layer_name   = "Risk (Poisson)"
    else:
        bucket_field = "risk_bucket_gbt"
        p_field      = "p_ge1_gbt"
        lam_field    = "lambda_hat_gbt"
        layer_name   = "Risk (GBT)"

    center, bounds = get_bounds(gj)
    m = folium.Map(location=center, zoom_start=13, tiles="cartodbpositron")

    def style_fn(feat):
        props  = feat.get("properties", {})
        bucket = props.get(bucket_field, "Moderate")
        color  = PALETTE.get(bucket, "#3186cc")
        return {"color": color, "weight": 3, "opacity": 0.9}

    def tooltip_html(props):
        show = ["id_segmen", "Jenis_pipa", "Diameter", "Length", "DMA_norm"]
        lines = [f"{k}: {props[k]}" for k in show if k in props]
        # angka dibulatkan agar rapi
        lam = props.get(lam_field, None)
        p1  = props.get(p_field, None)
        if lam is not None:
            try: lam = round(float(lam), 4)
            except: pass
        if p1 is not None:
            try: p1 = round(float(p1), 4)
            except: pass
        lines += [f"λ_hat: {lam}", f"P(≥1): {p1}", f"Risk: {props.get(bucket_field, '-')}"]
        return "<br>".join(lines)

    layer = folium.GeoJson(
        gj,
        name=layer_name,
        style_function=style_fn,
        highlight_function=lambda x: {"weight":5, "opacity":1.0}
    )
    layer.add_child(folium.features.GeoJsonTooltip(fields=[], labels=False, sticky=True))
    layer.add_to(m)

    # Inject tooltip per feature
    # (streamlit-folium akan render tooltip bawaan folium dengan baik)
    for feat in gj["features"]:
        props = feat.setdefault("properties", {})
        props["_tooltip"] = tooltip_html(props)

    # Legenda sederhana
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; background: white;
                padding:10px; border:1px solid #ccc; border-radius:8px; font-size:12px;">
      <b>Risk Legend</b><br>
      <div style="margin-top:6px;">
        <div><span style="display:inline-block;width:12px;height:12px;background:#7fc97f;margin-right:6px;"></span>Very Low</div>
        <div><span style="display:inline-block;width:12px;height:12px;background:#beed90;margin-right:6px;"></span>Low</div>
        <div><span style="display:inline-block;width:12px;height:12px;background:#ffff99;margin-right:6px;"></span>Moderate</div>
        <div><span style="display:inline-block;width:12px;height:12px;background:#fdc86e;margin-right:6px;"></span>Moderately High</div>
        <div><span style="display:inline-block;width:12px;height:12px;background:#fdae61;margin-right:6px;"></span>High</div>
        <div><span style="display:inline-block;width:12px;height:12px;background:#d7191c;margin-right:6px;"></span>Very High</div>
      </div>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl(collapsed=True).add_to(m)
    m.fit_bounds(bounds, padding=(10,10))
    return m

@st.cache_data(show_spinner=False)
def load_geojson(path: pathlib.Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------- UI ----------
st.title("Pipe Failure Risk – Poisson vs GBT")

with st.sidebar:
    st.header("Controls")
    mode = st.radio("Model:", list(FILES.keys()), index=1)
    st.markdown("**Tips presentasi**: zoom ke lokasi prioritas ➜ klik segmen untuk lihat atribut & risiko.")

# Opsi: unggah file custom (kalau tidak bundle)
st.markdown("**(Opsional)** Unggah GeoJSON Anda sendiri:")
up_poisson = st.file_uploader("Upload Poisson GeoJSON", type=["geojson"], key="up_p")
up_gbt     = st.file_uploader("Upload GBT GeoJSON", type=["geojson"], key="up_g")

# Muat sesuai pilihan
if mode == "Poisson Regression":
    if up_poisson is not None:
        gj = json.load(up_poisson)
    else:
        gj = load_geojson(FILES["Poisson Regression"])
else:
    if up_gbt is not None:
        gj = json.load(up_gbt)
    else:
        gj = load_geojson(FILES["Gradient Boosted Trees (GBT)"])

# Render folium map di Streamlit
m = build_map(gj, mode)
out = st_folium(m, width=None, height=700)
