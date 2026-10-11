 # =============================================================
#  DATA ANALYST – AI – CYBERSECURITY (TUNNELS)
#  Modernized version with enhanced design
# =============================================================

import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"

# ================================
# PATHS (PyInstaller compatible)
# ================================
DEFAULT_DATA_NAME = "tunnels_data_100_clean.csv"


def resource_path(relative_path: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, relative_path)


def locate_data_file(filename: str) -> str:
    candidates = [
        os.path.join(os.getcwd(), filename),
        os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), filename),
        resource_path(filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return resource_path(filename)


FILE_PATH = locate_data_file(DEFAULT_DATA_NAME)

# ================================
# IMPORTS
# ================================
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.colors as mcolors

import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, silhouette_score
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import chi2_contingency

sns.set(style="whitegrid")

# ================================
# CYBER DEMO DATA GENERATION
# ================================
def generate_cyber_excel(output_path: str = "cyber_demo_20x4.xlsx"):
    rng = np.random.default_rng(42)
    sites = [f"C{i}" for i in range(1, 21)]
    df_cyber = pd.DataFrame(
        {
            "Site": sites,
            "Connected_devices": rng.integers(5, 30, size=20),
            "Suspicious_attempts": rng.integers(0, 20, size=20),
            "Connection_speed": rng.integers(20, 100, size=20),
            "Vulnerability_score": np.clip(rng.normal(0.5, 0.25, size=20), 0, 1),
        }
    )
    df_cyber.loc[3, "Suspicious_attempts"] = 45
    df_cyber.loc[3, "Vulnerability_score"] = 0.95
    df_cyber.loc[15, "Connected_devices"] = 55
    df_cyber.loc[15, "Connection_speed"] = 10
    df_cyber.loc[15, "Vulnerability_score"] = 0.98
    df_cyber.to_excel(output_path, index=False)
    return output_path


# =============================================================
# 🎨 NEW MODERN THEME (Deep blue gradient)
# =============================================================

def _hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def mix_to_white(hex_color: str, amount: float = 0.6) -> str:
    r, g, b = _hex_to_rgb(hex_color)
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return _rgb_to_hex((r, g, b))


def mix_to_black(hex_color: str, amount: float = 0.3) -> str:
    r, g, b = _hex_to_rgb(hex_color)
    r = int(r * (1 - amount))
    g = int(g * (1 - amount))
    b = int(b * (1 - amount))
    return _rgb_to_hex((r, g, b))


# Modern palette with gradients
MODERN_BLUE = [
    "#0F172A",  # Very dark blue (almost black)
    "#1E293B",  # Dark blue
    "#334155",  # Blue-grey
    "#3B82F6",  # Bright blue
    "#60A5FA",  # Light blue
    "#93C5FD",  # Pastel blue
]

GRADIENT_COLORS = [
    "#0EA5E9",  # Cyan
    "#06B6D4",  # Cyan-turquoise
    "#14B8A6",  # Turquoise
    "#10B981",  # Green-turquoise
]

ACCENT_GLOW = [
    "#3B82F6",
    "#8B5CF6",
    "#EC4899",
    "#F59E0B",
    "#10B981",
    "#06B6D4",
]

ANOMALY_COLOR = "#EF4444"  # Bright red for anomalies

# Global theme
THEME = {
    "bg": "#0F172A",  # Very dark blue background
    "panel": "#1E293B",  # Slightly lighter panels
    "card": "#334155",  # Cards
    "title": "#F1F5F9",  # Off-white titles
    "text": "#E2E8F0",  # Main text
    "muted": "#94A3B8",  # Secondary text
    "accent": "#3B82F6",  # Bright blue accent
    "accent2": "#8B5CF6",  # Violet accent
    "success": "#10B981",  # Green
    "warning": "#F59E0B",  # Orange
    "danger": "#EF4444",  # Red
    "stroke": "#475569",  # Borders
}

# Modernized fonts
FONT_BTN = ("Inter", 13, "bold")
FONT_BTN_SMALL = ("Inter", 11, "bold")
FONT_TITLE = ("Inter", 24, "bold")
FONT_TITLE_LARGE = ("Inter", 32, "bold")
FONT_SUB = ("Inter", 14, "normal")
FONT_MONO = ("JetBrains Mono", 10, "normal")

# Modern colormap
MODERN_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "modern_blue",
    ["#1E293B", "#3B82F6", "#60A5FA", "#93C5FD"],
    N=256
)

# ================================
# MATPLOTLIB CONFIGURATION
# ================================
matplotlib.rcParams.update(
    {
        "font.size": 11,
        "font.family": "sans-serif",
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.facecolor": "#1E293B",
        "figure.facecolor": "#0F172A",
        "text.color": "#E2E8F0",
        "axes.labelcolor": "#E2E8F0",
        "xtick.color": "#94A3B8",
        "ytick.color": "#94A3B8",
        "axes.edgecolor": "#475569",
        "grid.color": "#334155",
        "grid.alpha": 0.3,
        "figure.dpi": 120,
        "savefig.dpi": 120,
    }
)

# ================================
# DATA SCHEMA
# ================================
LABEL_CANDIDATES = [
    "Nom du Tunnel", "Nom_Tunnel", "Tunnel", "Nom", "name", "Name",
    "ID", "Id", "id", "Code", "code",
]

ALIASES = {
    "Pression terrain (MPa)": ["Pression terrain (MPa)", "Pression_terrain_MPa", "Pression"],
    "Convergence (mm)": ["Convergence (mm)", "Convergence_mm", "Convergence"],
    "Humidité (%)": ["Humidité (%)", "Humidite_%", "Humidite", "Humidité"],
    "Température (°C)": ["Température (°C)", "Temperature_C", "Temperature", "Température"],
    "Vibrations (Hz)": ["Vibrations (Hz)", "Vibrations_Hz", "Vibration_Hz", "Vibrations"],
    "Fissures (mm)": ["Fissures (mm)", "Fissures_mm", "Fissures"],
    "Ventilation (m/s)": ["Ventilation (m/s)", "Ventilation_ms", "Ventilation"],
    "Durée exploitation (ans)": ["Durée exploitation (ans)", "Duree_exploitation_ans", "Durée", "Duree"],
}


def read_any_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df_ = pd.read_excel(path)
    elif ext == ".csv":
        try:
            df_ = pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            df_ = pd.read_csv(path, encoding="utf-8", errors="replace")
    else:
        df_ = pd.read_excel(path)
    df_.columns = df_.columns.astype(str).str.strip()
    return df_


def load_data_or_die() -> pd.DataFrame:
    if not os.path.exists(FILE_PATH):
        messagebox.showerror(
            "File not found",
            f"Could not find the file :\n{FILE_PATH}\n\n"
            "➡️ Place the file in the same folder as this script/executable\n"
            "➡️ or rebuild PyInstaller with --add-data.",
        )
        raise FileNotFoundError(FILE_PATH)
    return read_any_table(FILE_PATH)


def find_label_column(df_: pd.DataFrame):
    for c in LABEL_CANDIDATES:
        if c in df_.columns:
            return c
    n = len(df_)
    if n == 0:
        return None
    obj_cols = []
    for c in df_.columns:
        s = df_[c]
        if s.dtype == object or str(s.dtype).startswith("string"):
            obj_cols.append(c)
    if not obj_cols:
        return None
    best = None
    best_score = -1.0
    for c in obj_cols:
        s = df_[c].astype(str).replace("nan", "").fillna("")
        non_empty = (s.str.len() > 0).mean()
        nun = s[s.str.len() > 0].nunique(dropna=True)
        uniq_ratio = nun / max(1, n)
        score = 0.55 * uniq_ratio + 0.45 * non_empty
        if score > best_score:
            best_score = score
            best = c
    return best if best_score >= 0.45 else None


def extract_label_series(df_: pd.DataFrame, fallback_prefix: str = "Tunnel") -> pd.Series:
    label_col = find_label_column(df_)
    if label_col is not None:
        s = df_[label_col].astype(str).fillna("").replace("nan", "")
        fallback = [f"{fallback_prefix}_{i+1:03d}" for i in range(len(df_))]
        s = s.where(s.str.len() > 0, other=fallback)
        return s
    return pd.Series([f"{fallback_prefix}_{i+1:03d}" for i in range(len(df_))])


def build_X_from_df(df_: pd.DataFrame):
    resolved = {}
    for canon, opts in ALIASES.items():
        found = next((o for o in opts if o in df_.columns), None)
        if found is None:
            raise KeyError(f"Missing column : {canon} (expected: {opts})")
        resolved[canon] = found
    X_raw = df_[list(resolved.values())].rename(columns={v: k for k, v in resolved.items()})
    X_num = X_raw.apply(pd.to_numeric, errors="coerce")
    X_num = X_num.fillna(X_num.median(numeric_only=True))
    return X_num, list(X_num.columns)


def build_X_auto_from_df(df_: pd.DataFrame, min_cols: int = 2):
    label_col = find_label_column(df_)
    cols = list(df_.columns)
    if label_col in cols:
        cols.remove(label_col)
    if not cols:
        raise ValueError("CYBER: no usable columns (besides Name/ID column).")
    Xc = pd.DataFrame(index=df_.index)
    for c in cols:
        s = df_[c]
        if str(s.dtype) == "bool":
            Xc[c] = s.astype(int)
            continue
        if np.issubdtype(s.dtype, np.datetime64):
            dt = pd.to_datetime(s, errors="coerce")
            Xc[c] = (dt.view("int64") / 1e9)
            continue
        sn = pd.to_numeric(s, errors="coerce")
        if not sn.isna().all():
            Xc[c] = sn
            continue
        st = s.astype(str).replace("nan", "").fillna("")
        codes, _ = pd.factorize(st, sort=True)
        Xc[c] = codes.astype(float)
    keep = []
    for c in Xc.columns:
        nun = pd.Series(Xc[c]).nunique(dropna=True)
        if nun > 1:
            keep.append(c)
    Xc = Xc[keep]
    if Xc.shape[1] < min_cols:
        raise ValueError(
            f"CYBER: not enough useful columns detected ({Xc.shape[1]}). "
            f"At least {min_cols} columns are required."
        )
    med = Xc.median(numeric_only=True)
    Xc = Xc.fillna(med).fillna(0)
    return Xc, list(Xc.columns)


# ================================
# DATA LOADING
# ================================
df = load_data_or_die()
point_labels = extract_label_series(df, fallback_prefix="Tunnel")
X, variables = build_X_from_df(df)

# ================================
# PRE-CALCULATIONS (PCA / AI)
# ================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=variables)

pca = PCA()
X_pca = pca.fit_transform(X_scaled_df)
explained = pca.explained_variance_ratio_

# ALL TUNNELS FOR CLUSTERING
k_range = range(3, 8)
kmeans_models = {}
k_metrics = []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_k = km.fit_predict(X_scaled_df)
    sil = silhouette_score(X_scaled_df, labels_k) if k < len(X_scaled_df) else np.nan
    kmeans_models[k] = (km, labels_k)
    k_metrics.append({"k": k, "Inertia": km.inertia_, "Silhouette": sil})

clusters = kmeans_models[4][1]
k_metrics_df = pd.DataFrame(k_metrics).round(4)

Xtr, Xte, ytr, yte = train_test_split(X_scaled_df, clusters, test_size=0.30, random_state=42)
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(Xtr, ytr)
rf_pred = rf_model.predict(Xte)

rf_accuracy = float(accuracy_score(yte, rf_pred))
rf_report_dict = classification_report(yte, rf_pred, output_dict=True, zero_division=0)
rf_conf_mat = confusion_matrix(yte, rf_pred)

rf_report_txt = (
    f"Accuracy : {rf_accuracy:.4f}\n\n"
    f"Classification report :\n{classification_report(yte, rf_pred, zero_division=0)}\n"
    f"Confusion matrix :\n{rf_conf_mat}"
)

# ================================
# CYBER STATE
# ================================
def compute_cyber_state(df_in: pd.DataFrame, auto: bool = False):
    if auto:
        labels = extract_label_series(df_in, fallback_prefix="Row")
        Xc, vars_c = build_X_auto_from_df(df_in, min_cols=2)
    else:
        labels = extract_label_series(df_in, fallback_prefix="Tunnel")
        Xc, vars_c = build_X_from_df(df_in)

    scaler_c = StandardScaler()
    Xc_scaled = scaler_c.fit_transform(Xc)
    Xc_scaled_df = pd.DataFrame(Xc_scaled, columns=vars_c)

    pca_c = PCA()
    Xc_pca = pca_c.fit_transform(Xc_scaled_df)
    expl = pca_c.explained_variance_ratio_

    if Xc_pca.shape[1] < 2:
        raise ValueError("CYBER: PCA has less than 2 components.")

    n = len(Xc_scaled_df)

    if n >= 2:
        iso_c = IsolationForest(contamination=0.10, random_state=42)
        iso_pred_c = iso_c.fit_predict(Xc_scaled_df)
    else:
        iso_pred_c = np.ones(n, dtype=int)

    if n >= 3:
        nn = max(2, min(20, n - 1))
        lof_c = LocalOutlierFactor(n_neighbors=nn, contamination=0.10)
        lof_pred_c = lof_c.fit_predict(Xc_scaled_df)
    else:
        lof_pred_c = np.ones(n, dtype=int)

    return {
        "df": df_in,
        "labels": labels,
        "X": Xc,
        "variables": vars_c,
        "scaler": scaler_c,
        "X_scaled_df": Xc_scaled_df,
        "pca": pca_c,
        "X_pca": Xc_pca,
        "explained": expl,
        "iso_pred": iso_pred_c,
        "lof_pred": lof_pred_c,
    }


cyber_state = compute_cyber_state(df, auto=False)

# ================================
# GUI TOOLS
# ================================
def clear(frame):
    for w in frame.winfo_children():
        w.destroy()


def make_title(parent, text):
    lbl = tk.Label(
        parent,
        text=text,
        bg=THEME["bg"],
        fg=THEME["title"],
        font=FONT_TITLE,
    )
    lbl.pack(pady=(15, 10))
    return lbl


def show_table(frame, data: pd.DataFrame, title: str):
    clear(frame)
    frame.configure(bg=THEME["bg"])
    make_title(frame, title)
    _render_table_in(frame, data)


def show_tables_notebook(frame, title: str, tables: list):
    clear(frame)
    frame.configure(bg=THEME["bg"])
    make_title(frame, title)

    wrap = tk.Frame(frame, bg=THEME["bg"])
    wrap.pack(fill="both", expand=True, padx=20, pady=15)

    # Style for modern notebook
    style = ttk.Style()
    style.theme_use("default")
    style.configure(
        "Modern.TNotebook",
        background=THEME["panel"],
        borderwidth=0,
    )
    style.configure(
        "Modern.TNotebook.Tab",
        background=THEME["card"],
        foreground=THEME["text"],
        padding=[20, 10],
        font=("Inter", 11, "bold"),
    )
    style.map(
        "Modern.TNotebook.Tab",
        background=[("selected", THEME["accent"])],
        foreground=[("selected", "#FFFFFF")],
    )

    nb = ttk.Notebook(wrap, style="Modern.TNotebook")
    nb.pack(fill="both", expand=True)

    for tab_title, df_tab in tables:
        tab = tk.Frame(nb, bg=THEME["bg"])
        nb.add(tab, text=str(tab_title))
        _render_table_in(tab, df_tab)


def _render_table_in(parent, data: pd.DataFrame):
    parent.configure(bg=THEME["bg"])

    wrap = tk.Frame(parent, bg=THEME["bg"])
    wrap.pack(fill="both", expand=True, padx=15, pady=15)

    style = ttk.Style()
    style.theme_use("default")

    style.configure(
        "Modern.Treeview",
        background=THEME["panel"],
        fieldbackground=THEME["panel"],
        foreground=THEME["text"],
        rowheight=28,
        borderwidth=0,
    )
    style.configure(
        "Modern.Treeview.Heading",
        background=THEME["card"],
        foreground=THEME["title"],
        font=("Inter", 11, "bold"),
        relief="flat",
        borderwidth=1,
    )
    style.map(
        "Modern.Treeview.Heading",
        background=[("active", THEME["accent"])],
        foreground=[("active", "#FFFFFF")],
    )
    style.map(
        "Modern.Treeview",
        background=[("selected", THEME["accent"])],
        foreground=[("selected", "#FFFFFF")],
    )

    df_show = data.copy()
    df_show.columns = [str(c) for c in df_show.columns]

    cols = list(df_show.columns)
    tree = ttk.Treeview(wrap, columns=cols, show="headings", style="Modern.Treeview")

    vsb = ttk.Scrollbar(wrap, orient="vertical", command=tree.yview)
    hsb = ttk.Scrollbar(wrap, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

    tree.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    hsb.grid(row=1, column=0, sticky="ew")

    wrap.grid_rowconfigure(0, weight=1)
    wrap.grid_columnconfigure(0, weight=1)

    for c in cols:
        tree.heading(c, text=c)
        tree.column(c, anchor="center", width=180)

    tree.tag_configure("odd", background="#1E3A5F")
    tree.tag_configure("even", background=THEME["panel"])

    max_rows = 400
    if len(df_show) > max_rows:
        df_show = df_show.head(max_rows)

    for i, (_, r) in enumerate(df_show.iterrows()):
        tag = "even" if i % 2 == 0 else "odd"
        tree.insert("", "end", values=[str(v) for v in list(r)], tags=(tag,))


def setup_ax(ax):
    ax.set_facecolor(THEME["panel"])
    ax.grid(True, alpha=0.25, color=THEME["stroke"], linewidth=0.5)
    for sp in ax.spines.values():
        sp.set_color(THEME["stroke"])
        sp.set_linewidth(1.5)
    ax.tick_params(colors=THEME["muted"], labelsize=10)
    ax.title.set_color(THEME["title"])
    ax.xaxis.label.set_color(THEME["text"])
    ax.yaxis.label.set_color(THEME["text"])


def pick_label_indices(xs, ys, max_labels=25):
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    n = len(xs)
    if n <= max_labels:
        return np.arange(n)

    idx_ext = set([int(np.argmin(xs)), int(np.argmax(xs)), int(np.argmin(ys)), int(np.argmax(ys))])

    cx, cy = xs.mean(), ys.mean()
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    idx_far = np.argsort(d2)[::-1]

    out = list(idx_ext)
    for i in idx_far:
        if len(out) >= max_labels:
            break
        if int(i) not in idx_ext:
            out.append(int(i))
    return np.array(out, dtype=int)


def pretty_label(s: str, max_chars=18) -> str:
    s = str(s).replace("_", " ")
    if max_chars is None:
        return s
    try:
        mc = int(max_chars)
    except Exception:
        return s
    if mc <= 0:
        return s
    if len(s) <= mc:
        return s
    return s[: mc - 1] + "…"


def annotate_points(
    ax, xs, ys, labels,
    fontsize=9, color=None, max_labels=None,
    shorten_to=None, offset_strategy="radial",
):
    if color is None:
        color = THEME["text"]
    
    if max_labels is not None and max_labels <= 0:
        return

    xs = np.asarray(xs)
    ys = np.asarray(ys)
    labels = [pretty_label(l, shorten_to) for l in list(map(str, labels))]

    if max_labels is not None and len(labels) > max_labels:
        idx = pick_label_indices(xs, ys, max_labels=max_labels)
        xs = xs[idx]
        ys = ys[idx]
        labels = [labels[i] for i in idx]

    dx0 = (xs.max() - xs.min()) * 0.012 if xs.max() != xs.min() else 0.03
    dy0 = (ys.max() - ys.min()) * 0.012 if ys.max() != ys.min() else 0.03

    cx = float(xs.mean()) if xs.size else 0.0
    cy = float(ys.mean()) if ys.size else 0.0

    for x, y, lab in zip(xs, ys, labels):
        if offset_strategy == "radial":
            vx = float(x) - cx
            vy = float(y) - cy
            n = (vx * vx + vy * vy) ** 0.5
            if n < 1e-12:
                ox, oy = dx0, dy0
            else:
                ox = dx0 * (vx / n)
                oy = dy0 * (vy / n)
        else:
            ox, oy = dx0, dy0

        t = ax.text(
            float(x) + ox, float(y) + oy, lab,
            fontsize=fontsize, color=color,
            clip_on=False,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=THEME["panel"],
                edgecolor=THEME["accent"],
                alpha=0.8,
                linewidth=1,
            )
        )
        t.set_in_layout(True)


def add_hover_payload(fig, ax, xs, ys, labels, radius_px=14):
    payload = {
        "ax": ax,
        "xy": np.column_stack([np.asarray(xs), np.asarray(ys)]),
        "labels": list(map(str, labels)),
        "radius": float(radius_px),
    }
    if not hasattr(fig, "_hover_payloads"):
        fig._hover_payloads = []
    fig._hover_payloads.append(payload)


def install_hover_labels(canvas: FigureCanvasTkAgg, fig: Figure):
    if not hasattr(fig, "_hover_payloads"):
        return

    payloads = fig._hover_payloads
    ann_by_ax = {}

    for p in payloads:
        ax = p["ax"]
        if ax in ann_by_ax:
            continue

        ann = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(12, 12),
            textcoords="offset points",
            bbox=dict(
                boxstyle="round,pad=0.4",
                fc=THEME["card"],
                ec=THEME["accent"],
                lw=2,
            ),
            fontsize=10,
            color=THEME["title"],
            annotation_clip=False,
        )
        ann.set_visible(False)
        ann.set_in_layout(False)
        ann.set_zorder(10_000)
        ann.set_clip_on(False)
        ann_by_ax[ax] = ann

    def _hide_all():
        changed = False
        for ann in ann_by_ax.values():
            if ann.get_visible():
                ann.set_visible(False)
                changed = True
        if changed:
            canvas.draw_idle()

    def _on_move(event):
        if event.inaxes is None:
            _hide_all()
            return

        ax = event.inaxes
        if ax not in ann_by_ax:
            _hide_all()
            return

        payload_list = [p for p in payloads if p["ax"] is ax]
        if not payload_list:
            _hide_all()
            return

        best = None
        for p in payload_list:
            xy = p["xy"]
            if xy.size == 0:
                continue

            disp = ax.transData.transform(xy)
            dx = disp[:, 0] - event.x
            dy = disp[:, 1] - event.y
            dist = np.hypot(dx, dy)

            i = int(np.argmin(dist))
            if dist[i] <= p["radius"]:
                cand = (float(dist[i]), float(xy[i, 0]), float(xy[i, 1]), p["labels"][i])
                if best is None or cand[0] < best[0]:
                    best = cand

        if best is None:
            _hide_all()
            return

        _, x, y, lab = best
        ann = ann_by_ax[ax]
        ann.xy = (x, y)
        ann.set_text(str(lab))
        ann.set_visible(True)
        canvas.draw_idle()

    canvas.mpl_connect("motion_notify_event", _on_move)
    canvas.mpl_connect("axes_leave_event", lambda e: _hide_all())
    canvas.mpl_connect("figure_leave_event", lambda e: _hide_all())


def _find_point_in_fig(fig, query: str):
    if not hasattr(fig, "_hover_payloads"):
        return None
    q = (query or "").strip().lower()
    if not q:
        return None

    for p in fig._hover_payloads:
        labels = p.get("labels", [])
        xy = p.get("xy", None)
        ax = p.get("ax", None)
        if xy is None or ax is None:
            continue
        for i, lab in enumerate(labels):
            if q in str(lab).lower():
                x, y = float(xy[i, 0]), float(xy[i, 1])
                return ax, x, y, str(lab)
    return None


def _highlight_point(fig, canvas, ax, x, y, label, zoom_center=True):
    if hasattr(fig, "_search_artists"):
        for art in fig._search_artists:
            try:
                art.remove()
            except Exception:
                pass
    fig._search_artists = []

    ring = ax.scatter([x], [y], s=280, facecolors="none", edgecolors=THEME["danger"], linewidths=3, zorder=9999)
    cross = ax.scatter([x], [y], s=100, marker="x", c=THEME["danger"], linewidths=3, zorder=10000)

    ann = ax.annotate(
        f"✔ {label}",
        xy=(x, y),
        xytext=(15, 15),
        textcoords="offset points",
        fontsize=11,
        color=THEME["title"],
        bbox=dict(boxstyle="round,pad=0.5", fc=THEME["card"], ec=THEME["danger"], lw=2),
        zorder=10001,
        annotation_clip=False,
    )

    fig._search_artists.extend([ring, cross, ann])

    if zoom_center:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        xr = (xmax - xmin) if xmax != xmin else 1.0
        yr = (ymax - ymin) if ymax != ymin else 1.0
        ax.set_xlim(x - 0.25 * xr, x + 0.25 * xr)
        ax.set_ylim(y - 0.25 * yr, y + 0.25 * yr)

    canvas.draw_idle()


def center_axes(ax, xs, ys, pad_ratio=0.45):
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    if xs.size == 0:
        return

    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    xr = xmax - xmin
    yr = ymax - ymin

    pad_x = xr * pad_ratio if xr > 0 else 1.0
    pad_y = yr * pad_ratio if yr > 0 else 1.0

    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)


def fit_figure_to_canvas(fig: Figure, pad=0.01, max_iter=6):
    try:
        fig.set_constrained_layout(False)
    except Exception:
        pass

    for _ in range(max_iter):
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        bbox = fig.get_tightbbox(renderer)
        if bbox is None:
            break

        bbox_fig = bbox.transformed(fig.transFigure.inverted())
        sp = fig.subplotpars
        left, right, bottom, top = sp.left, sp.right, sp.bottom, sp.top
        changed = False

        if bbox_fig.x0 < 0:
            left = left + (-bbox_fig.x0) + pad
            changed = True
        if bbox_fig.x1 > 1:
            right = right - (bbox_fig.x1 - 1) - pad
            changed = True
        if bbox_fig.y0 < 0:
            bottom = bottom + (-bbox_fig.y0) + pad
            changed = True
        if bbox_fig.y1 > 1:
            top = top - (bbox_fig.y1 - 1) - pad
            changed = True

        left = min(max(left, 0.02), 0.80)
        right = max(min(right, 0.99), 0.20)
        bottom = min(max(bottom, 0.02), 0.80)
        top = max(min(top, 0.99), 0.20)

        if not changed:
            break
        fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)


def show_figure(frame, fig: Figure, title: str, margins=None, figsize=(10.0, 6.5), dpi=120):
    clear(frame)
    frame.configure(bg=THEME["bg"])
    make_title(frame, title)

    holder = tk.Frame(frame, bg=THEME["bg"])
    holder.pack(fill="both", expand=True, padx=15, pady=12)

    # Modern search bar
    search_bar = tk.Frame(holder, bg=THEME["card"], relief="flat", bd=2)
    search_bar.pack(fill="x", pady=(0, 12), padx=5)

    search_inner = tk.Frame(search_bar, bg=THEME["card"])
    search_inner.pack(fill="x", padx=15, pady=10)

    tk.Label(
        search_inner,
        text="🔍 Search for a tunnel :",
        bg=THEME["card"],
        fg=THEME["title"],
        font=("Inter", 11, "bold"),
    ).pack(side="left", padx=(0, 10))

    ent = tk.Entry(
        search_inner,
        font=("Consolas", 11),
        width=30,
        bg=THEME["panel"],
        fg=THEME["text"],
        insertbackground=THEME["accent"],
        relief="flat",
        bd=0,
    )
    ent.pack(side="left", padx=8, ipady=5)

    msg = tk.Label(
        search_inner,
        text="",
        bg=THEME["card"],
        fg=THEME["accent"],
        font=("Inter", 10, "bold"),
    )
    msg.pack(side="left", padx=15)

    canvas_wrap = tk.Frame(holder, bg=THEME["bg"])
    canvas_wrap.pack(fill="both", expand=True)

    try:
        fig.set_dpi(dpi)
        fig.set_size_inches(figsize[0], figsize[1], forward=True)
    except Exception:
        pass

    fig.patch.set_facecolor(THEME["bg"])

    if margins is None:
        margins = dict(left=0.18, right=0.95, bottom=0.22, top=0.88)
    try:
        fig.subplots_adjust(**margins)
    except Exception:
        pass

    fit_figure_to_canvas(fig, pad=0.01, max_iter=6)

    canvas = FigureCanvasTkAgg(fig, master=canvas_wrap)
    canvas.draw()
    w = canvas.get_tk_widget()
    w.configure(highlightthickness=0, bg=THEME["bg"])
    w.pack(fill="both", expand=True)

    w.focus_set()
    install_hover_labels(canvas, fig)

    def do_search():
        q = ent.get().strip()
        if not q:
            msg.config(text="⚠ Type a name/ID", fg=THEME["warning"])
            return

        res = _find_point_in_fig(fig, q)
        if res is None:
            msg.config(text="❌ No tunnel found", fg=THEME["danger"])
            return

        ax, x, y, lab = res
        _highlight_point(fig, canvas, ax, x, y, lab, zoom_center=True)
        msg.config(text=f"✓ Found : {lab}", fg=THEME["success"])

    btn = tk.Button(
        search_inner,
        text="🔎 Search",
        command=do_search,
        bg=THEME["accent"],
        fg="white",
        bd=0,
        cursor="hand2",
        activebackground=THEME["accent2"],
        activeforeground="white",
        font=("Inter", 10, "bold"),
        padx=15,
        pady=6,
        relief="flat",
    )
    btn.pack(side="left", padx=8)

    ent.bind("<Return>", lambda e: do_search())


# ================================
# BUSINESS FUNCTIONS
# ================================

# ---------- PCA ----------
def acp_stats(target):
    stats = pd.DataFrame({
        "Mean": X.mean(),
        "Std": X.std(),
        "Min": X.min(),
        "Max": X.max()
    }).round(3)
    show_table(target, stats.reset_index().rename(columns={"index": "Variable"}), "Descriptive statistics")


def acp_cr(target):
    show_table(target, X_scaled_df.round(3).head(120), "Centered-reduced matrix (first 120)")


def acp_corr(target):
    fig = Figure(figsize=(11, 7.5))
    ax = fig.add_subplot(111)
    setup_ax(ax)
    
    corr_matrix = X.corr()
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap=MODERN_CMAP,
        ax=ax,
        vmin=-1,
        vmax=1,
        cbar=True,
        annot_kws={"fontsize": 9, "color": THEME["title"]},
        linewidths=1,
        linecolor=THEME["stroke"],
    )
    ax.set_title("Correlation Matrix", pad=15, color=THEME["title"])
    
    show_figure(
        target, fig,
        "Heatmap – Correlation between variables",
        margins=dict(left=0.28, right=0.95, bottom=0.32, top=0.90),
        figsize=(11, 7.5)
    )


def acp_inerties(target):
    fig = Figure()
    ax = fig.add_subplot(111)
    setup_ax(ax)
    
    cumsum = np.cumsum(explained)
    ax.plot(
        range(1, len(cumsum) + 1),
        cumsum * 100,
        marker="o",
        linewidth=3,
        color=THEME["accent"],
        markersize=8,
        markerfacecolor=THEME["accent2"],
        markeredgecolor="white",
        markeredgewidth=2,
    )
    ax.axhline(y=80, color=THEME["warning"], linestyle="--", linewidth=2, alpha=0.7, label="80%")
    ax.fill_between(range(1, len(cumsum) + 1), 0, cumsum * 100, alpha=0.15, color=THEME["accent"])
    
    ax.set_xlabel("Number of components", fontsize=12)
    ax.set_ylabel("Cumulative explained variance (%)", fontsize=12)
    ax.set_title("Cumulative Explained Variance (PCA)", pad=15)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    show_figure(target, fig, "Inertia calculation & variance")


def acp_plan(target):
    fig = Figure()
    ax = fig.add_subplot(111)
    setup_ax(ax)

    # Gradient palette for points
    n_points = len(point_labels)
    colors = [ACCENT_GLOW[i % len(ACCENT_GLOW)] for i in range(n_points)]
    
    ax.scatter(
        X_pca[:, 0], X_pca[:, 1],
        s=70,
        c=colors,
        alpha=0.85,
        edgecolors="white",
        linewidths=1.2,
    )
    
    ax.set_xlabel(f"F1 ({explained[0]*100:.1f}%)", fontsize=13)
    ax.set_ylabel(f"F2 ({explained[1]*100:.1f}%)", fontsize=13)
    ax.set_title("Factorial Plane (PCA) - All tunnels", pad=15)

    labels_tunnel = list(point_labels)
    add_hover_payload(fig, ax, X_pca[:, 0], X_pca[:, 1], labels_tunnel, radius_px=16)
    annotate_points(
        ax, X_pca[:, 0], X_pca[:, 1], labels_tunnel,
        fontsize=8, max_labels=35, shorten_to=20
    )

    center_axes(ax, X_pca[:, 0], X_pca[:, 1], pad_ratio=0.55)
    show_figure(target, fig, "Factorial plane of tunnels (labels + hover)")


def acp_cercle(target):
    fig = Figure()
    ax = fig.add_subplot(111)
    setup_ax(ax)
    
    # Correlation circle
    circle = plt.Circle((0, 0), 1, fill=False, color=THEME["accent"], linewidth=2.5, alpha=0.8)
    ax.add_artist(circle)

    for i, v in enumerate(variables):
        col = ACCENT_GLOW[i % len(ACCENT_GLOW)]
        ax.arrow(
            0, 0,
            pca.components_[0, i],
            pca.components_[1, i],
            head_width=0.04,
            color=col,
            linewidth=2.5,
            length_includes_head=True,
            alpha=0.9,
        )
        
        # Labels with badges
        x_pos = pca.components_[0, i] * 1.15
        y_pos = pca.components_[1, i] * 1.15
        t = ax.text(
            x_pos, y_pos, v,
            fontsize=9,
            color=THEME["title"],
            clip_on=False,
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor=THEME["card"],
                edgecolor=col,
                alpha=0.9,
                linewidth=1.5,
            ),
            ha="center",
            va="center",
        )
        t.set_in_layout(True)

    ax.set_aspect("equal")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel("F1", fontsize=13)
    ax.set_ylabel("F2", fontsize=13)
    ax.set_title("Correlation Circle (Variables)", pad=15)
    
    show_figure(target, fig, "Correlation circle (factorial plane)")


def acp_quality(target):
    denom = np.sum(X_pca**2, axis=1, keepdims=True)
    denom = np.where(denom == 0, 1, denom)
    cos2_ind = (X_pca**2) / denom
    df_ind = pd.DataFrame(cos2_ind[:, :2], columns=["COS² F1", "COS² F2"])
    df_ind.insert(0, "Tunnel", list(point_labels)[: len(df_ind)])

    eigs = pca.explained_variance_
    loadings = pca.components_.T * np.sqrt(eigs)
    denv = np.sum(loadings**2, axis=1, keepdims=True)
    cos2_var = (loadings[:, :2] ** 2) / np.where(denv == 0, 1, denv)
    df_var = pd.DataFrame(cos2_var, columns=["COS² F1", "COS² F2"])
    df_var.insert(0, "Variable", variables)

    show_tables_notebook(
        target,
        "Representation quality (COS²)",
        [
            ("Tunnels (first 120)", df_ind.round(4).head(120)),
            ("Variables", df_var.round(4)),
        ],
    )


def acp_contrib(target):
    n = len(X_pca)
    eig = pca.explained_variance_

    contrib_ind = 100 * (X_pca[:, :2] ** 2) / (n * eig[:2])
    df_ind = pd.DataFrame(contrib_ind, columns=["Contrib F1 %", "Contrib F2 %"])
    df_ind.insert(0, "Tunnel", list(point_labels)[: len(df_ind)])

    df_var = pd.DataFrame({
        "Variable": variables,
        "Contrib F1 %": 100 * (pca.components_[0, :] ** 2),
        "Contrib F2 %": 100 * (pca.components_[1, :] ** 2)
    })

    show_tables_notebook(
        target,
        "Contributions (PCA)",
        [
            ("Tunnels (first 120)", df_ind.round(4).head(120)),
            ("Variables", df_var.round(4)),
        ],
    )


# ---------- AI ----------
def ia_clusters(target):
    fig = Figure()
    ax = fig.add_subplot(111)
    setup_ax(ax)

    cluster_palette = [THEME["accent"], THEME["success"], THEME["warning"], THEME["accent2"]]
    colors = [cluster_palette[int(c) % len(cluster_palette)] for c in clusters]

    ax.scatter(
        X_pca[:, 0], X_pca[:, 1],
        s=75,
        c=colors,
        alpha=0.85,
        edgecolors="white",
        linewidths=1.2,
    )
    ax.set_xlabel("F1", fontsize=13)
    ax.set_ylabel("F2", fontsize=13)
    ax.set_title("KMeans Clustering (k=4) - All tunnels", pad=15)

    labels_tunnel = list(point_labels)
    add_hover_payload(fig, ax, X_pca[:, 0], X_pca[:, 1], labels_tunnel, radius_px=16)
    annotate_points(
        ax, X_pca[:, 0], X_pca[:, 1], labels_tunnel,
        fontsize=8, max_labels=35, shorten_to=20
    )

    center_axes(ax, X_pca[:, 0], X_pca[:, 1], pad_ratio=0.65)
    show_figure(target, fig, "Display of clusters – tunnels")


def ia_pourcentage(target):
    counts = pd.Series(clusters).value_counts(normalize=True).sort_index() * 100
    out = counts.reset_index()
    out.columns = ["Cluster", "Percentage (%)"]
    show_table(target, out.round(2), "Cluster distribution")


def ia_rf_metrics(target):
    rep = pd.DataFrame(rf_report_dict).T.reset_index().rename(columns={"index": "Class"})
    for c in ["precision", "recall", "f1-score", "support"]:
        if c in rep.columns:
            rep[c] = pd.to_numeric(rep[c], errors="coerce")
    rep = rep.round(4)

    cm_df = pd.DataFrame(
        rf_conf_mat,
        columns=[f"Predicted {i}" for i in range(rf_conf_mat.shape[1])],
        index=[f"Real {i}" for i in range(rf_conf_mat.shape[0])],
    ).reset_index().rename(columns={"index": "Real Class"})

    km_df = k_metrics_df.copy()

    show_tables_notebook(
        target,
        "AI Metrics (Random Forest & KMeans)",
        [
            (f"RF Accuracy = {rf_accuracy:.4f}", rep),
            ("Confusion matrix", cm_df),
            ("KMeans: Inertia & Silhouette", km_df),
        ],
    )


def ia_predict_new(target):
    clear(target)
    target.configure(bg=THEME["bg"])
    make_title(target, "Predict a new tunnel (Random Forest)")

    form_wrap = tk.Frame(target, bg=THEME["card"], relief="flat", bd=2)
    form_wrap.pack(fill="x", padx=30, pady=15)

    form = tk.Frame(form_wrap, bg=THEME["card"])
    form.pack(fill="x", padx=20, pady=20)

    entries = {}
    med = X.median(numeric_only=True)

    for var in variables:
        row = tk.Frame(form, bg=THEME["card"])
        row.pack(fill="x", pady=5)
        
        tk.Label(
            row,
            text=var,
            bg=THEME["card"],
            fg=THEME["title"],
            font=("Inter", 11, "bold"),
            width=30,
            anchor="w",
        ).pack(side="left")
        
        e = tk.Entry(
            row,
            font=("Consolas", 11),
            width=20,
            bg=THEME["panel"],
            fg=THEME["text"],
            insertbackground=THEME["accent"],
            relief="flat",
            bd=0,
        )
        e.insert(0, str(float(med[var])))
        e.pack(side="left", ipady=4)
        entries[var] = e

    out_lbl = tk.Label(
        target,
        text="",
        bg=THEME["bg"],
        fg=THEME["accent"],
        font=("Inter", 15, "bold"),
    )
    out_lbl.pack(pady=15)

    def do_predict():
        try:
            vals = [float(entries[v].get()) for v in variables]
        except ValueError:
            messagebox.showerror("Error", "Please enter numbers only.")
            return
        
        x_new = pd.DataFrame([vals], columns=variables)
        x_new_sc = scaler.transform(x_new)
        x_new_sc_df = pd.DataFrame(x_new_sc, columns=variables)
        pred = rf_model.predict(x_new_sc_df)[0]
        
        cluster_colors = {0: THEME["accent"], 1: THEME["success"], 2: THEME["warning"], 3: THEME["accent2"]}
        color = cluster_colors.get(pred, THEME["text"])
        out_lbl.config(text=f"➡️ Predicted cluster : {pred}", fg=color)

    btn = tk.Button(
        target,
        text="🤖 Predict cluster",
        command=do_predict,
        bg=THEME["accent"],
        fg="white",
        bd=0,
        cursor="hand2",
        activebackground=THEME["accent2"],
        activeforeground="white",
        font=("Inter", 13, "bold"),
        padx=25,
        pady=10,
        relief="flat",
    )
    btn.pack(pady=12)


# ---------- AFC ----------
def build_afc():
    X_afc = X.iloc[:8].copy()
    for c in X_afc.columns:
        q = pd.qcut(X_afc[c], 10, labels=False, duplicates="drop")
        q = q.fillna(0).astype(int) + 1
        X_afc[c] = q

    cont = X_afc.values.astype(float)
    freq = cont / cont.sum() if cont.sum() != 0 else cont
    chi2, pval, dof, expected = chi2_contingency(cont)

    labels_ind = list(point_labels.iloc[:8])
    labels_var = list(X_afc.columns.astype(str))
    return X_afc, cont, freq, chi2, pval, labels_ind, labels_var


X_afc, cont_afc, freq_afc, chi2_afc, pval_afc, labels_afc_ind, labels_afc_var = build_afc()


def afc_cont(target):
    df_cont = pd.DataFrame(
        cont_afc,
        columns=variables,
        index=labels_afc_ind
    ).reset_index().rename(columns={"index": "Tunnel"})
    show_table(target, df_cont, "AFC – Contingency table")


def afc_freq(target):
    df_freq = pd.DataFrame(
        freq_afc,
        columns=variables,
        index=labels_afc_ind
    ).round(4).reset_index().rename(columns={"index": "Tunnel"})
    show_table(target, df_freq, "AFC – Frequency matrix")


def afc_plan(target):
    r = freq_afc.sum(axis=1)
    c = freq_afc.sum(axis=0)
    r = np.where(r == 0, 1e-12, r)
    c = np.where(c == 0, 1e-12, c)

    S = np.diag(1 / np.sqrt(r)) @ (freq_afc - np.outer(r, c)) @ np.diag(1 / np.sqrt(c))
    U, s, Vt = np.linalg.svd(S, full_matrices=False)

    F = np.diag(1 / np.sqrt(r)) @ U @ np.diag(s)
    G = np.diag(1 / np.sqrt(c)) @ Vt.T @ np.diag(s)

    fig = Figure()
    ax = fig.add_subplot(111)
    setup_ax(ax)

    # Row points (tunnels)
    ax.scatter(
        F[:, 0], F[:, 1],
        s=100,
        c=[ACCENT_GLOW[i % len(ACCENT_GLOW)] for i in range(len(F))],
        edgecolors="white",
        linewidths=1.5,
        alpha=0.9,
        label="Tunnels",
    )
    
    # Column points (variables)
    ax.scatter(
        G[:, 0], G[:, 1],
        s=120,
        c=[ACCENT_GLOW[(i+3) % len(ACCENT_GLOW)] for i in range(len(G))],
        marker="D",
        edgecolors="white",
        linewidths=1.5,
        alpha=0.9,
        label="Variables",
    )

    add_hover_payload(fig, ax, F[:, 0], F[:, 1], labels_afc_ind, radius_px=18)
    add_hover_payload(fig, ax, G[:, 0], G[:, 1], labels_afc_var, radius_px=18)

    annotate_points(ax, F[:, 0], F[:, 1], labels_afc_ind, fontsize=10, max_labels=8, shorten_to=20)
    annotate_points(ax, G[:, 0], G[:, 1], labels_afc_var, fontsize=10, max_labels=8, shorten_to=20)

    ax.set_title("Factorial Plane (AFC)", pad=15)
    ax.set_xlabel("Dimension 1", fontsize=13)
    ax.set_ylabel("Dimension 2", fontsize=13)
    ax.legend(loc="best", framealpha=0.9)

    center_axes(ax, np.r_[F[:, 0], G[:, 0]], np.r_[F[:, 1], G[:, 1]], pad_ratio=0.95)
    show_figure(target, fig, "AFC – Factorial plane & associations")


def afc_test(target):
    out = pd.DataFrame({
        "Statistic χ²": [chi2_afc],
        "p-value": [pval_afc],
        "Interpretation": ["Significant" if pval_afc < 0.05 else "Not significant"]
    })
    show_table(target, out.round(6), "AFC – χ² test & Interpretation")


# ---------- CYBER ----------
def cyber_import_dialog(target):
    global cyber_state
    path = filedialog.askopenfilename(
        title="Import new data (Excel/CSV)",
        filetypes=[
            ("Excel", "*.xlsx *.xls"),
            ("CSV", "*.csv"),
            ("All files", "*.*")
        ],
    )
    if not path:
        return

    try:
        df_new = read_any_table(path)
        cyber_state = compute_cyber_state(df_new, auto=True)

        cols_used = ", ".join(cyber_state["variables"][:12])
        more = " ..." if len(cyber_state["variables"]) > 12 else ""
        messagebox.showinfo(
            "CYBER - Import successful",
            f"✓ File loaded successfully!\n\n"
            f"Processed columns :\n{cols_used}{more}\n\n"
            f"Total : {len(cyber_state['df'])} rows"
        )

    except Exception as e:
        messagebox.showerror(
            "CYBER import impossible",
            f"❌ Error : {e}\n\n"
            f"➡️ The file must contain at least 2 usable columns\n"
            f"➡️ A Name/ID column is recommended"
        )
        return

    show_table(
        target,
        cyber_state["df"].head(200),
        f"CYBER – Imported data ({os.path.basename(path)})"
    )


def cyber_if(target):
    st = cyber_state
    Xp = st["X_pca"]
    labels = st["labels"]
    iso_p = st["iso_pred"]
    expl = st["explained"]

    fig = Figure()
    ax = fig.add_subplot(111)
    setup_ax(ax)

    colors = [ANOMALY_COLOR if iso_p[i] == -1 else ACCENT_GLOW[i % len(ACCENT_GLOW)] for i in range(len(iso_p))]
    sizes = [120 if iso_p[i] == -1 else 70 for i in range(len(iso_p))]
    
    ax.scatter(
        Xp[:, 0], Xp[:, 1],
        s=sizes,
        c=colors,
        alpha=0.85,
        edgecolors="white",
        linewidths=1.5,
    )
    
    ax.set_title("Isolation Forest - Anomaly Detection", pad=15)
    ax.set_xlabel(f"F1 ({expl[0]*100:.1f}%)" if len(expl) > 0 else "F1", fontsize=13)
    ax.set_ylabel(f"F2 ({expl[1]*100:.1f}%)" if len(expl) > 1 else "F2", fontsize=13)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Normal',
               markerfacecolor=ACCENT_GLOW[0], markersize=10, markeredgecolor='white'),
        Line2D([0], [0], marker='o', color='w', label='Anomaly',
               markerfacecolor=ANOMALY_COLOR, markersize=12, markeredgecolor='white')
    ]
    ax.legend(handles=legend_elements, loc='best', framealpha=0.9)

    add_hover_payload(fig, ax, Xp[:, 0], Xp[:, 1], labels, radius_px=16)
    annotate_points(
        ax, Xp[:, 0], Xp[:, 1], labels,
        fontsize=8, max_labels=30, shorten_to=20
    )

    center_axes(ax, Xp[:, 0], Xp[:, 1], pad_ratio=0.65)
    show_figure(target, fig, "CYBER – Isolation Forest (anomalies)")


def cyber_lof(target):
    st = cyber_state
    Xp = st["X_pca"]
    labels = st["labels"]
    lof_p = st["lof_pred"]
    expl = st["explained"]

    fig = Figure()
    ax = fig.add_subplot(111)
    setup_ax(ax)

    colors = [ANOMALY_COLOR if lof_p[i] == -1 else ACCENT_GLOW[i % len(ACCENT_GLOW)] for i in range(len(lof_p))]
    sizes = [120 if lof_p[i] == -1 else 70 for i in range(len(lof_p))]
    
    ax.scatter(
        Xp[:, 0], Xp[:, 1],
        s=sizes,
        c=colors,
        alpha=0.85,
        edgecolors="white",
        linewidths=1.5,
    )
    
    ax.set_title("Local Outlier Factor (LOF) - Anomaly Detection", pad=15)
    ax.set_xlabel(f"F1 ({expl[0]*100:.1f}%)" if len(expl) > 0 else "F1", fontsize=13)
    ax.set_ylabel(f"F2 ({expl[1]*100:.1f}%)" if len(expl) > 1 else "F2", fontsize=13)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Normal',
               markerfacecolor=ACCENT_GLOW[0], markersize=10, markeredgecolor='white'),
        Line2D([0], [0], marker='o', color='w', label='Anomaly',
               markerfacecolor=ANOMALY_COLOR, markersize=12, markeredgecolor='white')
    ]
    ax.legend(handles=legend_elements, loc='best', framealpha=0.9)

    add_hover_payload(fig, ax, Xp[:, 0], Xp[:, 1], labels, radius_px=16)
    annotate_points(
        ax, Xp[:, 0], Xp[:, 1], labels,
        fontsize=8, max_labels=30, shorten_to=20
    )

    center_axes(ax, Xp[:, 0], Xp[:, 1], pad_ratio=0.65)
    show_figure(target, fig, "CYBER – LOF (anomalies)")


def cyber_interp(target):
    st = cyber_state
    labels = st["labels"]
    idx_if = np.where(st["iso_pred"] == -1)[0]
    idx_lof = np.where(st["lof_pred"] == -1)[0]

    df_if = pd.DataFrame({
        "IF Anomalies": [str(labels.iloc[i]) for i in idx_if]
    }) if len(idx_if) else pd.DataFrame({"IF Anomalies": ["✓ No anomaly detected"]})
    
    df_lof = pd.DataFrame({
        "LOF Anomalies": [str(labels.iloc[i]) for i in idx_lof]
    }) if len(idx_lof) else pd.DataFrame({"LOF Anomalies": ["✓ No anomaly detected"]})

    show_tables_notebook(
        target,
        "CYBER – Interpretation (anomaly lists)",
        [
            (f"Isolation Forest ({len(idx_if)} detected)", df_if),
            (f"LOF ({len(idx_lof)} detected)", df_lof),
        ],
    )


# =============================================================
# MAIN TKINTER INTERFACE
# =============================================================

root = tk.Tk()
root.title("Data Analyst – AI – Cybersecurity (Modern Design)")
root.geometry("1280x800")
root.configure(bg=THEME["bg"])
root.minsize(1200, 750)

# Global style for ttk widgets
style = ttk.Style()
style.theme_use("default")

container = tk.Frame(root, bg=THEME["bg"])
container.pack(fill="both", expand=True)
container.grid_rowconfigure(0, weight=1)
container.grid_columnconfigure(0, weight=1)

frames = {}


def show_frame(name):
    frames[name].tkraise()


def make_centered_page(name: str):
    page = tk.Frame(container, bg=THEME["bg"])
    frames[name] = page
    page.grid(row=0, column=0, sticky="nsew")

    center = tk.Frame(page, bg=THEME["bg"])
    center.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.97, relheight=0.94)
    return page, center


def create_gradient_button(canvas, widgets_store, x0, y0, x1, y1, text, command, bg_color, font):
    """Creates a modern button with colored border"""
    # Outer border
    canvas.create_rectangle(x0, y0, x1, y1, outline=bg_color, width=3)
    
    btn = tk.Button(
        canvas,
        text=text,
        command=command,
        bg=THEME["card"],
        fg=THEME["title"],
        activebackground=bg_color,
        activeforeground="white",
        bd=0,
        relief="flat",
        cursor="hand2",
        font=font,
        justify="center",
        wraplength=max(120, int((x1 - x0) * 0.90)),
    )
    widgets_store.append(btn)
    
    canvas.create_window(
        (x0 + x1) / 2,
        (y0 + y1) / 2,
        window=btn,
        width=(x1 - x0) - 12,
        height=(y1 - y0) - 12,
    )


def make_module_page(name: str):
    page, center = make_centered_page(name)

    menu_layer = tk.Frame(center, bg=THEME["bg"])
    menu_layer.place(relx=0, rely=0, relwidth=1, relheight=1)

    result_layer = tk.Frame(center, bg=THEME["bg"])
    result_layer.place(relx=0, rely=0, relwidth=1, relheight=1)

    nav = tk.Frame(result_layer, bg=THEME["panel"], relief="flat")
    nav.pack(fill="x", padx=15, pady=(10, 0))

    nav_inner = tk.Frame(nav, bg=THEME["panel"])
    nav_inner.pack(pady=8, padx=10)

    def back_to_menu():
        menu_layer.tkraise()

    btn_back = tk.Button(
        nav_inner,
        text="⬅ Back",
        command=back_to_menu,
        bg=THEME["card"],
        fg=THEME["title"],
        bd=0,
        cursor="hand2",
        activebackground=THEME["accent"],
        activeforeground="white",
        font=("Inter", 11, "bold"),
        padx=15,
        pady=6,
        relief="flat",
    )
    btn_back.pack(side="left", padx=5)

    btn_home = tk.Button(
        nav_inner,
        text="🏠 Main Menu",
        command=lambda: show_frame("MENU"),
        bg=THEME["card"],
        fg=THEME["title"],
        bd=0,
        cursor="hand2",
        activebackground=THEME["accent"],
        activeforeground="white",
        font=("Inter", 11, "bold"),
        padx=15,
        pady=6,
        relief="flat",
    )
    btn_home.pack(side="left", padx=5)

    content = tk.Frame(result_layer, bg=THEME["bg"])
    content.pack(fill="both", expand=True)

    def open_result(func):
        result_layer.tkraise()
        func()

    menu_layer.tkraise()
    return page, menu_layer, result_layer, content, open_result


# =============================================================
# START PAGE
# =============================================================
start, start_center = make_centered_page("START")
start_canvas = tk.Canvas(start_center, bg=THEME["bg"], highlightthickness=0)
start_canvas.pack(fill="both", expand=True)
start_widgets = []


def draw_start_layout():
    for w in start_widgets:
        try:
            w.destroy()
        except Exception:
            pass
    start_widgets.clear()

    start_canvas.delete("all")
    w = start_canvas.winfo_width()
    h = start_canvas.winfo_height()
    if w < 10 or h < 10:
        return

    # Main border
    pad = 12
    start_canvas.create_rectangle(pad, pad, w - pad, h - pad, outline=THEME["accent"], width=4)

    # Main title
    title_w, title_h = int(w * 0.75), int(h * 0.20)
    tx0 = (w - title_w) // 2
    ty0 = int(h * 0.12)
    start_canvas.create_rectangle(
        tx0, ty0, tx0 + title_w, ty0 + title_h,
        outline=THEME["accent2"], width=3
    )

    title_lbl = tk.Label(
        start_canvas,
        text="Data Analyst, AI & Cybersecurity\nTunnel Analysis",
        bg=THEME["bg"],
        fg=THEME["title"],
        font=FONT_TITLE_LARGE,
        justify="center",
    )
    start_widgets.append(title_lbl)
    start_canvas.create_window(w // 2, ty0 + title_h // 2, window=title_lbl)

    # Main button
    btn_w, btn_h = int(w * 0.60), int(h * 0.14)
    bx0 = (w - btn_w) // 2
    by0 = int(h * 0.44)
    start_canvas.create_rectangle(
        bx0, by0, bx0 + btn_w, by0 + btn_h,
        outline=THEME["success"], width=3
    )

    btn_start = tk.Button(
        start_canvas,
        text="🚀 START ANALYSIS",
        command=lambda: show_frame("MENU"),
        bg=THEME["accent"],
        fg="white",
        activebackground=THEME["accent2"],
        activeforeground="white",
        bd=0,
        relief="flat",
        cursor="hand2",
        font=("Inter", 24, "bold"),
    )
    start_widgets.append(btn_start)
    start_canvas.create_window(w // 2, by0 + btn_h // 2, window=btn_start)

    # Credits
    box_w, box_h = int(w * 0.38), int(h * 0.13)
    gap = int(w * 0.12)
    yb = int(h * 0.80)
    x1 = (w - (2 * box_w + gap)) // 2
    x2 = x1 + box_w + gap

    start_canvas.create_rectangle(x1, yb, x1 + box_w, yb + box_h, outline=THEME["warning"], width=3)
    start_canvas.create_rectangle(x2, yb, x2 + box_w, yb + box_h, outline=THEME["accent"], width=3)

    lbl1 = tk.Label(
        start_canvas,
        text="Dr. EL MKHALET MOUNA",
        bg=THEME["bg"],
        fg=THEME["warning"],
        font=("Inter", 14, "bold"),
    )
    lbl2 = tk.Label(
        start_canvas,
        text="Lahlou Chaimaa",
        bg=THEME["bg"],
        fg=THEME["accent"],
        font=("Inter", 14, "bold"),
    )
    start_widgets.extend([lbl1, lbl2])
    start_canvas.create_window(x1 + box_w // 2, yb + box_h // 2, window=lbl1)
    start_canvas.create_window(x2 + box_w // 2, yb + box_h // 2, window=lbl2)


start_canvas.bind("<Configure>", lambda e: draw_start_layout())


# =============================================================
# MAIN MENU
# =============================================================
menu, menu_center = make_centered_page("MENU")
menu_canvas = tk.Canvas(menu_center, bg=THEME["bg"], highlightthickness=0)
menu_canvas.pack(fill="both", expand=True)
menu_widgets = []


def draw_menu_layout():
    for wdg in menu_widgets:
        try:
            wdg.destroy()
        except Exception:
            pass
    menu_widgets.clear()

    menu_canvas.delete("all")
    w = menu_canvas.winfo_width()
    h = menu_canvas.winfo_height()
    if w < 10 or h < 10:
        return

    pad = 12
    menu_canvas.create_rectangle(pad, pad, w - pad, h - pad, outline=THEME["accent"], width=4)

    box_w = int(w * 0.40)
    box_h = int(h * 0.24)
    gx = int(w * 0.08)
    gy = int(h * 0.08)

    left_x = int(w * 0.08)
    top_y = int(h * 0.16)

    positions = [
        (left_x, top_y),
        (left_x + box_w + gx, top_y),
        (left_x, top_y + box_h + gy),
        (left_x + box_w + gx, top_y + box_h + gy),
    ]

    modules = [
        ("📊 PCA\n(Principal Component\nAnalysis)", "ACP", THEME["accent"]),
        ("🌀 AFC\n(Correspondence Analysis)", "AFC", THEME["accent2"]),
        ("🤖 AI\nClustering & Machine\nLearning", "IA", THEME["success"]),
        ("🔒 CYBERSECURITY\nAnomaly Detection", "CYBER", THEME["warning"]),
    ]

    for (x, y), (text, dest, color) in zip(positions, modules):
        x1, y1 = x + box_w, y + box_h
        
        menu_canvas.create_rectangle(x, y, x1, y1, outline=color, width=3)
        
        btn = tk.Button(
            menu_canvas,
            text=text,
            command=lambda d=dest: show_frame(d),
            bg=color,
            fg="white",
            activebackground=mix_to_black(color, 0.2),
            activeforeground="white",
            bd=0,
            relief="flat",
            cursor="hand2",
            font=("Inter", 15, "bold"),
            justify="center",
            wraplength=int(box_w * 0.88),
        )
        menu_widgets.append(btn)
        menu_canvas.create_window(
            x + box_w // 2,
            y + box_h // 2,
            window=btn,
            width=box_w - 15,
            height=box_h - 15,
        )


menu_canvas.bind("<Configure>", lambda e: draw_menu_layout())


# =============================================================
# MODULE PAGES
# =============================================================

# ---------- PCA PAGE ----------
acp_page, acp_menu_layer, acp_result_layer, acp_content, acp_open = make_module_page("ACP")
acp_canvas = tk.Canvas(acp_menu_layer, bg=THEME["bg"], highlightthickness=0)
acp_canvas.pack(fill="both", expand=True)
acp_widgets = []


def draw_acp_layout():
    for wdg in acp_widgets:
        try:
            wdg.destroy()
        except Exception:
            pass
    acp_widgets.clear()

    w = acp_canvas.winfo_width()
    h = acp_canvas.winfo_height()
    acp_canvas.delete("all")
    if w < 10 or h < 10:
        return

    pad = 12
    acp_canvas.create_rectangle(pad, pad, w - pad, h - pad, outline=THEME["accent"], width=4)

    rows = 4
    margin_x = int(w * 0.10)
    margin_y = int(h * 0.10)
    gap_x = int(w * 0.08)
    gap_y = int(h * 0.08)

    bw = int((w - 2 * margin_x - gap_x) / 2)
    bh = int((h - 2 * margin_y - (rows - 1) * gap_y) / rows)

    buttons = [
        ("Descriptive\nstatistics", lambda: acp_open(lambda: acp_stats(acp_content)), THEME["accent"]),
        ("Centered-reduced\nmatrix", lambda: acp_open(lambda: acp_cr(acp_content)), ACCENT_GLOW[1]),
        ("Correlation\nheatmap", lambda: acp_open(lambda: acp_corr(acp_content)), ACCENT_GLOW[2]),
        ("Explained\nvariance", lambda: acp_open(lambda: acp_inerties(acp_content)), ACCENT_GLOW[3]),
        ("Factorial plane\n(tunnels)", lambda: acp_open(lambda: acp_plan(acp_content)), ACCENT_GLOW[4]),
        ("Correlation\ncircle", lambda: acp_open(lambda: acp_cercle(acp_content)), ACCENT_GLOW[5]),
        ("Representation\nquality", lambda: acp_open(lambda: acp_quality(acp_content)), ACCENT_GLOW[0]),
        ("Contributions\n(COS²)", lambda: acp_open(lambda: acp_contrib(acp_content)), ACCENT_GLOW[1]),
    ]

    for i, (txt, cmd, col) in enumerate(buttons):
        r = i // 2
        c = i % 2
        x0 = margin_x + c * (bw + gap_x)
        y0 = margin_y + r * (bh + gap_y)
        x1 = x0 + bw
        y1 = y0 + bh
        
        create_gradient_button(acp_canvas, acp_widgets, x0, y0, x1, y1, txt, cmd, col, FONT_BTN_SMALL)


acp_canvas.bind("<Configure>", lambda e: draw_acp_layout())


# ---------- AI PAGE ----------
ia_page, ia_menu_layer, ia_result_layer, ia_content, ia_open = make_module_page("IA")
ia_canvas = tk.Canvas(ia_menu_layer, bg=THEME["bg"], highlightthickness=0)
ia_canvas.pack(fill="both", expand=True)
ia_widgets = []


def draw_ia_layout():
    for wdg in ia_widgets:
        try:
            wdg.destroy()
        except Exception:
            pass
    ia_widgets.clear()

    w = ia_canvas.winfo_width()
    h = ia_canvas.winfo_height()
    ia_canvas.delete("all")
    if w < 10 or h < 10:
        return

    pad = 12
    ia_canvas.create_rectangle(pad, pad, w - pad, h - pad, outline=THEME["success"], width=4)

    margin_x = int(w * 0.12)
    margin_y = int(h * 0.18)
    gap_x = int(w * 0.10)
    gap_y = int(h * 0.12)

    bw = int((w - 2 * margin_x - gap_x) / 2)
    bh = int((h - 2 * margin_y - gap_y) / 2)

    buttons = [
        ("Visualization of\nclusters", lambda: ia_open(lambda: ia_clusters(ia_content)), THEME["success"]),
        ("Cluster\ndistribution (%)", lambda: ia_open(lambda: ia_pourcentage(ia_content)), ACCENT_GLOW[3]),
        ("Random Forest\nmetrics", lambda: ia_open(lambda: ia_rf_metrics(ia_content)), ACCENT_GLOW[4]),
        ("Predict new\ntunnel", lambda: ia_open(lambda: ia_predict_new(ia_content)), ACCENT_GLOW[5]),
    ]

    for i, (txt, cmd, col) in enumerate(buttons):
        r = i // 2
        c = i % 2
        x0 = margin_x + c * (bw + gap_x)
        y0 = margin_y + r * (bh + gap_y)
        x1 = x0 + bw
        y1 = y0 + bh
        
        create_gradient_button(ia_canvas, ia_widgets, x0, y0, x1, y1, txt, cmd, col, FONT_BTN)


ia_canvas.bind("<Configure>", lambda e: draw_ia_layout())


# ---------- AFC PAGE ----------
afc_page, afc_menu_layer, afc_result_layer, afc_content, afc_open = make_module_page("AFC")
afc_canvas = tk.Canvas(afc_menu_layer, bg=THEME["bg"], highlightthickness=0)
afc_canvas.pack(fill="both", expand=True)
afc_widgets = []


def draw_afc_layout():
    for wdg in afc_widgets:
        try:
            wdg.destroy()
        except Exception:
            pass
    afc_widgets.clear()

    w = afc_canvas.winfo_width()
    h = afc_canvas.winfo_height()
    afc_canvas.delete("all")
    if w < 10 or h < 10:
        return

    pad = 12
    afc_canvas.create_rectangle(pad, pad, w - pad, h - pad, outline=THEME["accent2"], width=4)

    margin_x = int(w * 0.10)
    margin_y = int(h * 0.18)
    gap_x = int(w * 0.08)
    gap_y = int(h * 0.12)

    bw = int((w - 2 * margin_x - gap_x) / 2)
    bh = int((h - 2 * margin_y - gap_y) / 2)

    buttons = [
        ("Contingency\ntable", lambda: afc_open(lambda: afc_cont(afc_content)), THEME["accent2"]),
        ("Frequency\nmatrix", lambda: afc_open(lambda: afc_freq(afc_content)), ACCENT_GLOW[2]),
        ("Factorial plane\nAFC", lambda: afc_open(lambda: afc_plan(afc_content)), ACCENT_GLOW[1]),
        ("χ² test\n& p-value", lambda: afc_open(lambda: afc_test(afc_content)), ACCENT_GLOW[0]),
    ]

    for i, (txt, cmd, col) in enumerate(buttons):
        r = i // 2
        c = i % 2
        x0 = margin_x + c * (bw + gap_x)
        y0 = margin_y + r * (bh + gap_y)
        x1 = x0 + bw
        y1 = y0 + bh
        
        create_gradient_button(afc_canvas, afc_widgets, x0, y0, x1, y1, txt, cmd, col, FONT_BTN)


afc_canvas.bind("<Configure>", lambda e: draw_afc_layout())


# ---------- CYBER PAGE ----------
cyb_page, cyb_menu_layer, cyb_result_layer, cyb_content, cyb_open = make_module_page("CYBER")
cyb_canvas = tk.Canvas(cyb_menu_layer, bg=THEME["bg"], highlightthickness=0)
cyb_canvas.pack(fill="both", expand=True)
cyb_widgets = []


def draw_cyber_layout():
    for wdg in cyb_widgets:
        try:
            wdg.destroy()
        except Exception:
            pass
    cyb_widgets.clear()

    w = cyb_canvas.winfo_width()
    h = cyb_canvas.winfo_height()
    cyb_canvas.delete("all")
    if w < 10 or h < 10:
        return

    pad = 12
    cyb_canvas.create_rectangle(pad, pad, w - pad, h - pad, outline=THEME["warning"], width=4)

    margin_x = int(w * 0.10)
    margin_y = int(h * 0.18)
    gap_x = int(w * 0.08)
    gap_y = int(h * 0.12)

    bw = int((w - 2 * margin_x - gap_x) / 2)
    bh = int((h - 2 * margin_y - gap_y) / 2)

    buttons = [
        ("📁 IMPORT\nExcel/CSV File", lambda: cyb_open(lambda: cyber_import_dialog(cyb_content)), THEME["warning"]),
        ("Isolation Forest\n(Anomalies)", lambda: cyb_open(lambda: cyber_if(cyb_content)), THEME["danger"]),
        ("LOF\n(Local Outlier Factor)", lambda: cyb_open(lambda: cyber_lof(cyb_content)), ACCENT_GLOW[2]),
        ("Interpretation\nresults", lambda: cyb_open(lambda: cyber_interp(cyb_content)), ACCENT_GLOW[0]),
    ]

    for i, (txt, cmd, col) in enumerate(buttons):
        r = i // 2
        c = i % 2
        x0 = margin_x + c * (bw + gap_x)
        y0 = margin_y + r * (bh + gap_y)
        x1 = x0 + bw
        y1 = y0 + bh
        
        create_gradient_button(cyb_canvas, cyb_widgets, x0, y0, x1, y1, txt, cmd, col, FONT_BTN)

cyb_canvas.bind("<Configure>", lambda e: draw_cyber_layout())

# =============================================================
# MAIN LAUNCH
# =============================================================
def on_closing():
    plt.close('all')
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Center window
root.update_idletasks()
w = root.winfo_width()
h = root.winfo_height()
x = (root.winfo_screenwidth() // 2) - (w // 2)
y = (root.winfo_screenheight() // 2) - (h // 2)
root.geometry(f"{w}x{h}+{x}+{y}")

# Entry animation
def fade_in():
    alpha = 0.0
    while alpha < 1.0:
        try:
            root.attributes('-alpha', alpha)
            root.update()
            alpha += 0.02
        except:
            break
    root.attributes('-alpha', 1.0)

# Launch
if __name__ == "__main__":
    try:
        fade_in()
        show_frame("START")
        root.mainloop()
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        messagebox.showerror("Fatal error", f"❌ Error: {str(e)}")
        print(f"ERROR: {e}")