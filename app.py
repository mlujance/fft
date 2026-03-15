# ===========================================
# File: C:\projects\panel_lab\app.py
# Local Decision Cockpit MVP
# Stack: Panel + Plotly + DuckDB + pandas
# Run:
#   python -m panel serve app.py --show --autoreload
# ===========================================

import io
import math
from datetime import datetime, timedelta

import duckdb
import numpy as np
import pandas as pd
import panel as pn
import plotly.express as px

# -------------------------------------------
# Panel config
# -------------------------------------------
pn.extension(
    "plotly",
    "tabulator",
    notifications=True,
    sizing_mode="stretch_width",
)

# Tema oscuro
pn.config.theme = "dark"

# -------------------------------------------
# App constants
# -------------------------------------------
APP_TITLE = "Local Decision Cockpit"
APP_SUBTITLE = "Panel + Plotly + DuckDB · MVP local-first"
TABLE_NAME = "investments"

# -------------------------------------------
# Helpers
# -------------------------------------------
def debug_log(msg: str) -> None:
    if "debug_toggle" in globals():
        if debug_toggle.value:
            debug_box.value = (
                f"{debug_box.value}\n[{datetime.now().strftime('%H:%M:%S')}] {msg}"
            ).strip()


def money(x: float) -> str:
    try:
        return f"{float(x):,.2f}"
    except Exception:
        return "0.00"


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def make_demo_data(n: int = 350) -> pd.DataFrame:
    np.random.seed(42)

    assets = [
        "BTC", "ETH", "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA",
        "Gold ETF", "Trade Republic Cash", "IBEX ETF", "Airbus"
    ]

    asset_class_map = {
        "BTC": "Crypto",
        "ETH": "Crypto",
        "SPY": "ETF",
        "QQQ": "ETF",
        "AAPL": "Equity",
        "MSFT": "Equity",
        "NVDA": "Equity",
        "TSLA": "Equity",
        "Gold ETF": "Commodity ETF",
        "Trade Republic Cash": "Cash",
        "IBEX ETF": "ETF",
        "Airbus": "Equity",
    }

    base_date = datetime(2025, 1, 1)
    rows = []

    for i in range(n):
        asset = np.random.choice(
            assets,
            p=[0.08, 0.07, 0.10, 0.08, 0.10, 0.10, 0.10, 0.07, 0.07, 0.12, 0.06, 0.05]
        )
        dt = base_date + timedelta(days=int(i * 1.5))
        amount = np.round(np.random.uniform(1, 25), 4)
        price = np.round(np.random.uniform(15, 800), 2)
        fee = np.round(np.random.uniform(0, 4), 2)
        expected_return = np.round(np.random.uniform(-0.08, 0.25), 4)
        risk_score = int(np.random.randint(1, 11))
        notes = f"Operación demo #{i+1}"

        rows.append(
            {
                "id": i + 1,
                "date": dt.date().isoformat(),
                "asset": asset,
                "asset_class": asset_class_map[asset],
                "amount": float(amount),
                "price": float(price),
                "fee": float(fee),
                "expected_return": float(expected_return),
                "risk_score": int(risk_score),
                "notes": notes,
            }
        )

    df = pd.DataFrame(rows)
    df["capital"] = df["amount"] * df["price"] + df["fee"]
    df["weighted_expected_profit"] = df["capital"] * df["expected_return"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = df["date"].dt.to_period("M").astype(str)
    return df


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza columnas si el usuario sube su propio CSV.
    Requiere al menos: date, asset, amount, price
    """
    original_columns = list(df.columns)
    rename_map = {}

    for col in df.columns:
        c = str(col).strip().lower()
        if c in ["fecha", "date"]:
            rename_map[col] = "date"
        elif c in ["activo", "asset", "platform", "plataforma"]:
            rename_map[col] = "asset"
        elif c in ["asset_class", "clase", "clase_activo", "asset class"]:
            rename_map[col] = "asset_class"
        elif c in ["amount", "cantidad", "units", "shares"]:
            rename_map[col] = "amount"
        elif c in ["price", "precio", "precio_unitario", "unit_price"]:
            rename_map[col] = "price"
        elif c in ["fee", "fees", "comision", "comisión"]:
            rename_map[col] = "fee"
        elif c in ["expected_return", "retorno_esperado", "return"]:
            rename_map[col] = "expected_return"
        elif c in ["risk_score", "riesgo", "risk"]:
            rename_map[col] = "risk_score"
        elif c in ["notes", "nota", "notas", "comments"]:
            rename_map[col] = "notes"

    df = df.rename(columns=rename_map).copy()

    required = ["date", "asset", "amount", "price"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas obligatorias en el CSV: {missing}. "
            f"Columnas detectadas: {original_columns}"
        )

    if "asset_class" not in df.columns:
        df["asset_class"] = "Unknown"
    if "fee" not in df.columns:
        df["fee"] = 0.0
    if "expected_return" not in df.columns:
        df["expected_return"] = 0.0
    if "risk_score" not in df.columns:
        df["risk_score"] = 5
    if "notes" not in df.columns:
        df["notes"] = ""

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    df["fee"] = pd.to_numeric(df["fee"], errors="coerce").fillna(0.0)
    df["expected_return"] = pd.to_numeric(df["expected_return"], errors="coerce").fillna(0.0)
    df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce").fillna(5).astype(int)
    df["notes"] = df["notes"].fillna("").astype(str)

    df = df[df["date"].notna()].copy()
    df = df[df["amount"] > 0].copy()
    df = df[df["price"] > 0].copy()

    df["capital"] = df["amount"] * df["price"] + df["fee"]
    df["weighted_expected_profit"] = df["capital"] * df["expected_return"]
    df["month"] = df["date"].dt.to_period("M").astype(str)

    if "id" not in df.columns:
        df["id"] = range(1, len(df) + 1)

    ordered_cols = [
        "id", "date", "asset", "asset_class", "amount", "price", "fee",
        "expected_return", "risk_score", "notes", "capital",
        "weighted_expected_profit", "month"
    ]
    for c in ordered_cols:
        if c not in df.columns:
            df[c] = None

    return df[ordered_cols].copy()


def build_connection() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(database=":memory:")


def register_df(df: pd.DataFrame) -> None:
    global current_df
    current_df = df.copy()
    con.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
    con.register("df_temp", current_df)
    con.execute(f"CREATE TABLE {TABLE_NAME} AS SELECT * FROM df_temp")
    con.unregister("df_temp")
    debug_log(f"Tabla '{TABLE_NAME}' registrada en DuckDB con {len(current_df)} filas.")


def get_filter_options(df: pd.DataFrame):
    asset_classes = ["Todos"] + sorted(df["asset_class"].dropna().astype(str).unique().tolist())
    assets = ["Todos"] + sorted(df["asset"].dropna().astype(str).unique().tolist())
    return asset_classes, assets


def filtered_df() -> pd.DataFrame:
    df = current_df.copy()

    if class_select.value != "Todos":
        df = df[df["asset_class"] == class_select.value]

    if asset_select.value != "Todos":
        df = df[df["asset"] == asset_select.value]

    min_cap = safe_float(min_capital_slider.value, 0.0)
    df = df[df["capital"] >= min_cap]

    min_risk, max_risk = risk_slider.value
    df = df[(df["risk_score"] >= min_risk) & (df["risk_score"] <= max_risk)]

    q = search_input.value.strip().lower()
    if q:
        df = df[
            df["asset"].astype(str).str.lower().str.contains(q)
            | df["notes"].astype(str).str.lower().str.contains(q)
            | df["asset_class"].astype(str).str.lower().str.contains(q)
        ]

    if sort_select.value == "Fecha desc":
        df = df.sort_values(["date", "id"], ascending=[False, False])
    elif sort_select.value == "Capital desc":
        df = df.sort_values(["capital", "id"], ascending=[False, False])
    elif sort_select.value == "Riesgo desc":
        df = df.sort_values(["risk_score", "capital"], ascending=[False, False])
    elif sort_select.value == "Retorno desc":
        df = df.sort_values(["expected_return", "capital"], ascending=[False, False])

    return df.head(row_limit_slider.value).copy()


def compute_kpis(df: pd.DataFrame):
    if df.empty:
        return {
            "rows": 0,
            "capital_total": 0.0,
            "avg_risk": 0.0,
            "expected_profit": 0.0,
            "unique_assets": 0,
        }

    return {
        "rows": int(len(df)),
        "capital_total": float(df["capital"].sum()),
        "avg_risk": float(df["risk_score"].mean()),
        "expected_profit": float(df["weighted_expected_profit"].sum()),
        "unique_assets": int(df["asset"].nunique()),
    }


def render_kpis(df: pd.DataFrame):
    k = compute_kpis(df)
    return pn.Row(
        metric_card("Filas visibles", f"{k['rows']}"),
        metric_card("Capital total", money(k["capital_total"])),
        metric_card("Riesgo medio", f"{k['avg_risk']:.2f}"),
        metric_card("Profit esperado", money(k["expected_profit"])),
        metric_card("Activos únicos", f"{k['unique_assets']}"),
    )


def metric_card(title: str, value: str):
    return pn.pane.HTML(
        f"""
        <div style="
            background:#151b24;
            border:1px solid #2a3340;
            border-radius:14px;
            padding:18px;
            min-height:110px;
            box-shadow:0 4px 20px rgba(0,0,0,0.20);
        ">
            <div style="font-size:13px;color:#9fb0c3;text-transform:uppercase;letter-spacing:0.6px;">
                {title}
            </div>
            <div style="font-size:28px;font-weight:700;color:#f3f6fb;margin-top:10px;">
                {value}
            </div>
        </div>
        """,
        sizing_mode="stretch_width",
    )


def capital_by_asset_chart(df: pd.DataFrame):
    if df.empty:
        return empty_chart("Capital por activo")

    chart_df = (
        df.groupby("asset", as_index=False)["capital"]
        .sum()
        .sort_values("capital", ascending=False)
    )
    fig = px.bar(
        chart_df,
        x="asset",
        y="capital",
        title="Capital por activo",
        template="plotly_dark",
    )
    fig.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20))
    return pn.pane.Plotly(fig, config={"responsive": True})


def timeline_chart(df: pd.DataFrame):
    if df.empty:
        return empty_chart("Capital acumulado")

    chart_df = (
        df.groupby("date", as_index=False)["capital"]
        .sum()
        .sort_values("date")
        .rename(columns={"capital": "capital_day"})
    )
    chart_df["capital_cum"] = chart_df["capital_day"].cumsum()

    fig = px.line(
        chart_df,
        x="date",
        y="capital_cum",
        markers=True,
        title="Capital acumulado",
        template="plotly_dark",
    )
    fig.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20))
    return pn.pane.Plotly(fig, config={"responsive": True})


def risk_return_chart(df: pd.DataFrame):
    if df.empty:
        return empty_chart("Mapa riesgo / retorno")

    chart_df = (
        df.groupby("asset", as_index=False)
        .agg(
            avg_return=("expected_return", "mean"),
            avg_risk=("risk_score", "mean"),
            capital=("capital", "sum"),
        )
        .sort_values("capital", ascending=False)
    )

    fig = px.scatter(
        chart_df,
        x="avg_risk",
        y="avg_return",
        size="capital",
        color="asset",
        hover_name="asset",
        title="Mapa riesgo / retorno",
        template="plotly_dark",
    )
    fig.update_layout(height=430, margin=dict(l=20, r=20, t=50, b=20))
    return pn.pane.Plotly(fig, config={"responsive": True})


def empty_chart(title: str):
    return pn.pane.HTML(
        f"""
        <div style="
            height:380px;
            display:flex;
            align-items:center;
            justify-content:center;
            background:#151b24;
            border:1px solid #2a3340;
            border-radius:14px;
            color:#9fb0c3;
            font-size:18px;
        ">
            {title}: sin datos
        </div>
        """,
        sizing_mode="stretch_width",
    )


def recommendations_panel(df: pd.DataFrame):
    recs = []

    if df.empty:
        recs.append(("Baja", "No hay datos para analizar."))
    else:
        high_risk = df[df["risk_score"] >= 8]
        if len(high_risk) > 0:
            recs.append(("Alta", f"Tienes {len(high_risk)} posiciones visibles de riesgo alto."))

        concentration = (
            df.groupby("asset", as_index=False)["capital"]
            .sum()
            .sort_values("capital", ascending=False)
        )
        if not concentration.empty:
            share = concentration.iloc[0]["capital"] / max(df["capital"].sum(), 1e-9)
            if share > 0.35:
                recs.append(("Alta", f"Concentración elevada: {concentration.iloc[0]['asset']} pesa {share:.1%}."))

        weak = df[df["expected_return"] < 0]
        if len(weak) > 0:
            recs.append(("Media", f"{len(weak)} posiciones tienen retorno esperado negativo."))

        if not recs:
            recs.append(("Baja", "No se detectan alertas simples con las reglas actuales."))

    items = []
    for priority, text in recs:
        color = {"Alta": "#ff6b6b", "Media": "#ffd166", "Baja": "#4cc9f0"}.get(priority, "#9fb0c3")
        items.append(
            pn.pane.HTML(
                f"""
                <div style="
                    background:#151b24;
                    border:1px solid #2a3340;
                    border-left:4px solid {color};
                    border-radius:12px;
                    padding:14px;
                    margin-bottom:10px;
                ">
                    <div style="font-size:12px;color:{color};font-weight:700;text-transform:uppercase;">
                        {priority}
                    </div>
                    <div style="font-size:14px;color:#e8edf5;margin-top:6px;">
                        {text}
                    </div>
                </div>
                """
            )
        )

    return pn.Column(*items, sizing_mode="stretch_width")


def dataframe_download(df: pd.DataFrame):
    sio = io.StringIO()
    df.to_csv(sio, index=False)
    return sio.getvalue().encode("utf-8")


def run_sql_query(sql_text: str) -> pd.DataFrame:
    sql_text = sql_text.strip()
    if not sql_text:
        raise ValueError("La consulta SQL está vacía.")
    debug_log(f"Ejecutando SQL: {sql_text[:120]}")
    return con.execute(sql_text).df()


# -------------------------------------------
# State
# -------------------------------------------
con = build_connection()
current_df = make_demo_data()
register_df(current_df)

# -------------------------------------------
# Widgets
# -------------------------------------------
file_input = pn.widgets.FileInput(accept=".csv", multiple=False, name="Subir CSV")
load_demo_btn = pn.widgets.Button(name="Cargar demo", button_type="primary")
refresh_btn = pn.widgets.Button(name="Refrescar", button_type="default")
export_btn = pn.widgets.FileDownload(
    filename="filtered_investments.csv",
    button_type="success",
    label="Exportar CSV filtrado",
    callback=lambda: dataframe_download(filtered_df()),
)

asset_classes, assets = get_filter_options(current_df)

class_select = pn.widgets.Select(name="Clase de activo", options=asset_classes, value="Todos")
asset_select = pn.widgets.Select(name="Activo", options=assets, value="Todos")
min_capital_slider = pn.widgets.FloatSlider(name="Capital mínimo", start=0, end=10000, step=50, value=0)
risk_slider = pn.widgets.IntRangeSlider(name="Rango riesgo", start=1, end=10, value=(1, 10))
row_limit_slider = pn.widgets.IntSlider(name="Máx filas", start=10, end=500, step=10, value=150)
sort_select = pn.widgets.Select(
    name="Orden",
    options=["Fecha desc", "Capital desc", "Riesgo desc", "Retorno desc"],
    value="Fecha desc",
)
search_input = pn.widgets.TextInput(name="Buscar", placeholder="activo, notas, clase...")

sql_editor = pn.widgets.TextAreaInput(
    name="SQL local DuckDB",
    value=f"SELECT asset, SUM(capital) AS total_capital FROM {TABLE_NAME} GROUP BY asset ORDER BY total_capital DESC LIMIT 20;",
    height=160,
)
sql_run_btn = pn.widgets.Button(name="Ejecutar SQL", button_type="primary")
sql_result_table = pn.widgets.Tabulator(pd.DataFrame(), disabled=True, height=320, pagination="local", page_size=12)

debug_toggle = pn.widgets.Checkbox(name="Modo debug", value=False)
debug_box = pn.widgets.TextAreaInput(name="Log debug", value="", height=180, disabled=True)

table_widget = pn.widgets.Tabulator(
    pd.DataFrame(),
    disabled=True,
    pagination="local",
    page_size=15,
    layout="fit_data_stretch",
    height=420,
)

# -------------------------------------------
# Actions
# -------------------------------------------
def sync_filter_options():
    global asset_classes, assets
    asset_classes, assets = get_filter_options(current_df)
    class_select.options = asset_classes
    asset_select.options = assets

    if class_select.value not in asset_classes:
        class_select.value = "Todos"
    if asset_select.value not in assets:
        asset_select.value = "Todos"

    max_cap = float(max(10000, math.ceil(current_df["capital"].max() / 100.0) * 100.0)) if not current_df.empty else 10000
    min_capital_slider.end = max_cap


def update_dashboard():
    df = filtered_df()
    debug_log(f"Dashboard actualizado con {len(df)} filas filtradas.")

    # KPIs
    kpi_row.objects = [render_kpis(df)]

    # charts
    charts_grid.objects = [
        pn.Row(capital_by_asset_chart(df), timeline_chart(df)),
        risk_return_chart(df),
    ]

    # table
    display_df = df.copy()
    if not display_df.empty:
        display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
    table_widget.value = display_df

    # recommendations
    insights_col.objects = [recommendations_panel(df)]

    # sql default sync
    export_btn.filename = f"filtered_investments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"


def load_demo(_=None):
    register_df(make_demo_data())
    sync_filter_options()
    pn.state.notifications.success("Dataset demo cargado.")
    update_dashboard()


def refresh_data(_=None):
    register_df(current_df)
    sync_filter_options()
    pn.state.notifications.info("Datos refrescados.")
    update_dashboard()


def handle_file_upload(event):
    if not file_input.value:
        return
    try:
        raw = io.BytesIO(file_input.value)
        uploaded_df = pd.read_csv(raw)
        uploaded_df = normalize_dataframe(uploaded_df)
        register_df(uploaded_df)
        sync_filter_options()
        pn.state.notifications.success(f"CSV cargado correctamente: {len(uploaded_df)} filas.")
        update_dashboard()
    except Exception as e:
        pn.state.notifications.error(f"Error cargando CSV: {e}")
        debug_log(f"Error CSV: {e}")


def execute_sql(_=None):
    try:
        result = run_sql_query(sql_editor.value)
        sql_result_table.value = result
        pn.state.notifications.success(f"Consulta ejecutada: {len(result)} filas.")
    except Exception as e:
        sql_result_table.value = pd.DataFrame({"error": [str(e)]})
        pn.state.notifications.error(f"Error SQL: {e}")
        debug_log(f"Error SQL: {e}")


load_demo_btn.on_click(load_demo)
refresh_btn.on_click(refresh_data)
sql_run_btn.on_click(execute_sql)
file_input.param.watch(handle_file_upload, "value")

for w in [class_select, asset_select, min_capital_slider, risk_slider, row_limit_slider, sort_select, search_input]:
    w.param.watch(lambda event: update_dashboard(), "value")

# -------------------------------------------
# Layout containers
# -------------------------------------------
header = pn.pane.HTML(
    f"""
    <div style="padding:10px 0 2px 0;">
        <div style="font-size:34px;font-weight:800;color:#f4f7fb;">{APP_TITLE}</div>
        <div style="font-size:14px;color:#9fb0c3;margin-top:6px;">{APP_SUBTITLE}</div>
    </div>
    """
)

control_card = pn.Column(
    "## Controles",
    file_input,
    pn.Row(load_demo_btn, refresh_btn),
    export_btn,
    pn.Spacer(height=10),
    class_select,
    asset_select,
    min_capital_slider,
    risk_slider,
    row_limit_slider,
    sort_select,
    search_input,
    pn.Spacer(height=10),
    debug_toggle,
    debug_box,
    sizing_mode="stretch_width",
)

kpi_row = pn.Column()
charts_grid = pn.Column()
insights_col = pn.Column()

dashboard_tab = pn.Column(
    kpi_row,
    pn.Spacer(height=12),
    charts_grid,
    pn.Spacer(height=12),
    pn.pane.Markdown("## Insights"),
    insights_col,
    sizing_mode="stretch_width",
)

table_tab = pn.Column(
    "## Tabla interactiva",
    table_widget,
    sizing_mode="stretch_width",
)

sql_tab = pn.Column(
    "## Consola SQL local",
    pn.pane.Markdown(
        f"""
        Consulta directamente la tabla `**{TABLE_NAME}**` dentro de DuckDB.
        Ejemplos útiles:

        - `SELECT * FROM {TABLE_NAME} LIMIT 20;`
        - `SELECT asset, AVG(price) FROM {TABLE_NAME} GROUP BY asset;`
        - `SELECT month, SUM(capital) FROM {TABLE_NAME} GROUP BY month ORDER BY month;`
        """
    ),
    sql_editor,
    sql_run_btn,
    sql_result_table,
    sizing_mode="stretch_width",
)

about_tab = pn.Column(
    "## Qué aporta esta base",
    pn.pane.Markdown(
        """
        Este MVP ya te deja construir algo con sentido:

        - cockpit local para decisiones
        - análisis reproducible
        - ingestión CSV
        - exploración visual
        - trazabilidad vía SQL
        - base sólida para añadir:
          - LLM local con Ollama
          - embeddings / búsqueda semántica
          - scoring
          - OCR / PDFs
          - histórico persistente en DuckDB fichero
        """
    ),
    pn.pane.Markdown(
        """
        **Siguiente paso lógico**:
        conectar esta base a un flujo local de documentos y un modelo local
        para generar explicaciones, comparativas y recomendaciones justificadas.
        """
    ),
    sizing_mode="stretch_width",
)

tabs = pn.Tabs(
    ("Dashboard", dashboard_tab),
    ("Tabla", table_tab),
    ("SQL", sql_tab),
    ("About", about_tab),
    dynamic=True,
)

main_layout = pn.Row(
    pn.Column(control_card, width=360),
    pn.Column(header, tabs, sizing_mode="stretch_width"),
    sizing_mode="stretch_width",
)

template = pn.template.FastListTemplate(
    title=APP_TITLE,
    theme="dark",
    main=[main_layout],
    header_background="#0f1722",
    accent_base_color="#00c2ff",
)
# -------------------------------------------
# Init
# -------------------------------------------
sync_filter_options()
update_dashboard()
execute_sql()

template.servable()