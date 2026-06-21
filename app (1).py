import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="💻 LaptopPriceAI",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

.hero-box {
    background: linear-gradient(135deg, rgba(102,126,234,0.25), rgba(118,75,162,0.25));
    border: 1px solid rgba(102,126,234,0.4);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    backdrop-filter: blur(10px);
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.hero-subtitle {
    color: rgba(255,255,255,0.65);
    font-size: 1.05rem;
    margin-top: 0.5rem;
}

.result-box {
    background: linear-gradient(135deg, rgba(102,126,234,0.3), rgba(118,75,162,0.3));
    border: 2px solid rgba(102,126,234,0.6);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { box-shadow: 0 0 20px rgba(102,126,234,0.3); }
    50%       { box-shadow: 0 0 40px rgba(102,126,234,0.6); }
}
.result-label { color: rgba(255,255,255,0.65); font-size: 0.9rem; letter-spacing: 0.08em; }
.result-price { font-size: 3rem; font-weight: 700; color: #f093fb; }
.result-range { color: rgba(255,255,255,0.5); font-size: 0.85rem; margin-top: 0.3rem; }

.metric-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}

section[data-testid="stSidebar"] {
    background: rgba(15,12,41,0.85) !important;
    border-right: 1px solid rgba(102,126,234,0.2);
}

div.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2.5rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    width: 100%;
}

h1,h2,h3 { color: white; }
p, li { color: rgba(255,255,255,0.7); }
</style>
""", unsafe_allow_html=True)

# ─── DATA PREPROCESSING ─────────────────────────────────────────────────────────
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("laptop_data.csv")
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    # Clean Ram and Weight
    df["Ram"]    = df["Ram"].astype(str).str.replace("GB", "", regex=False).str.strip().astype("int64")
    df["Weight"] = df["Weight"].astype(str).str.replace("kg", "", regex=False).str.strip().astype("float64")

    # Touchscreen & IPS
    df["touchscreen"] = df["ScreenResolution"].apply(lambda x: 1 if "Touchscreen" in str(x) else 0)
    df["IPS Panel"]   = df["ScreenResolution"].apply(lambda x: 1 if "IPS Panel"   in str(x) else 0)

    # PPI
    new = df["ScreenResolution"].str.split("x", n=1, expand=True)
    df["x_res"] = new[0].str.findall(r"(\d+)").apply(lambda x: int(x[-1]) if x else 0)
    df["y_res"] = new[1].str.strip().str.extract(r"(\d+)")[0].astype(float).fillna(0).astype(int)
    df["Inches"] = pd.to_numeric(df["Inches"], errors="coerce").fillna(15.0)
    df["ppi"] = (((df["x_res"]**2) + (df["y_res"]**2))**0.5 / df["Inches"]).astype("float64")
    df.drop(columns=["ScreenResolution", "x_res", "y_res", "Inches"], inplace=True)

    # CPU brand
    def fetch_processor(text):
        parts = str(text).split()
        brand = " ".join(parts[:3])
        if brand in ["Intel Core i5", "Intel Core i7", "Intel Core i3"]:
            return brand
        elif parts[0] == "Intel":
            return "other intel processor"
        else:
            return "AMD processor"
    df["cpu brand"] = df["Cpu"].apply(fetch_processor)
    df.drop(columns=["Cpu"], inplace=True)

    # Memory parsing
    df["Memory"] = df["Memory"].astype(str).str.replace(r"\.0\b", "", regex=True)
    df["Memory"] = df["Memory"].str.replace("GB", " ", regex=False).str.replace("TB", "000", regex=False)
    mem_split = df["Memory"].str.split("+", n=1, expand=True)
    df["first"]  = mem_split[0].str.strip()
    df["second"] = mem_split[1].fillna("0 ").str.strip()

    df["layer1HDD"] = df["first"].apply(lambda x: 1 if "HDD"          in str(x) else 0)
    df["layer1SSD"] = df["first"].apply(lambda x: 1 if "SSD"          in str(x) else 0)
    df["layer2HDD"] = df["second"].apply(lambda x: 1 if "HDD"         in str(x) else 0)
    df["layer2SSD"] = df["second"].apply(lambda x: 1 if "SSD"         in str(x) else 0)

    df["first"]  = df["first"].str.replace(r"[^\d]", " ", regex=True).str.strip()
    df["second"] = df["second"].str.replace(r"[^\d]", " ", regex=True).str.strip()
    df["first"]  = df["first"].str.split().str[0].fillna("0").astype(int)
    df["second"] = df["second"].str.split().str[0].fillna("0").astype(int)

    df["HDD"] = (df["first"] * df["layer1HDD"] + df["second"] * df["layer2HDD"]).astype(int)
    df["SSD"] = (df["first"] * df["layer1SSD"] + df["second"] * df["layer2SSD"]).astype(int)
    df.drop(columns=["first", "second", "layer1HDD", "layer1SSD",
                     "layer2HDD", "layer2SSD", "Memory"], inplace=True)

    # GPU brand
    df["GPU Brand"] = df["Gpu"].apply(lambda x: str(x).split()[0])
    df = df[df["GPU Brand"] != "ARM"].copy()
    df.drop(columns=["Gpu"], inplace=True)

    # OS
    def cat_os(inp):
        if inp in ["Windows 10", "Windows 7", "Windows 10 S"]:
            return "Windows"
        elif inp in ["macOS", "Mac OS X"]:
            return "Mac"
        else:
            return "Other/No OS/Linux"
    df["OS"] = df["OpSys"].apply(cat_os)
    df.drop(columns=["OpSys"], inplace=True)

    # Final type enforcement — ensures all numeric cols are numeric
    for col in ["Ram", "Weight", "touchscreen", "IPS Panel", "ppi", "HDD", "SSD", "Price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()
    df = df.reset_index(drop=True)
    return df

# ─── MODEL TRAINING ─────────────────────────────────────────────────────────────
@st.cache_resource
def train_model(_df):
    # Column order: Company(0), TypeName(1), Ram(2), Weight(3), touchscreen(4),
    #               IPS Panel(5), ppi(6), cpu brand(7), HDD(8), SSD(9), GPU Brand(10), OS(11)
    x = _df.drop(columns=["Price"])
    y = np.log(_df["Price"].astype(float))

    cat_idx = [
        list(x.columns).index("Company"),
        list(x.columns).index("TypeName"),
        list(x.columns).index("cpu brand"),
        list(x.columns).index("GPU Brand"),
        list(x.columns).index("OS"),
    ]

    step1 = ColumnTransformer(transformers=[
        ("ohe", OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore"), cat_idx)
    ], remainder="passthrough")

    rf   = RandomForestRegressor(n_estimators=100, max_depth=17, random_state=42)
    gbdt = GradientBoostingRegressor(learning_rate=0.1, n_estimators=150, max_depth=5, random_state=42)
    xgb  = XGBRegressor(max_depth=5, learning_rate=0.4, n_estimators=200,
                        random_state=42, eval_metric="logloss", verbosity=0)

    step2 = VotingRegressor([("rf", rf), ("gbr", gbdt), ("xgb", xgb)], weights=[5, 1, 1])
    pipe  = Pipeline([("step1", step1), ("step2", step2)])
    pipe.fit(x, y)
    return pipe

# ─── DROPDOWN VALUES ────────────────────────────────────────────────────────────
COMPANIES   = ['Apple','Asus','Chuwi','Dell','Fujitsu','Google','HP','Huawei',
               'LG','Lenovo','MSI','Mediacom','Microsoft','Razer','Samsung',
               'Toshiba','Vero','Xiaomi']
TYPES       = ['2 in 1 Convertible','Gaming','Netbook','Notebook','Ultrabook','Workstation']
RAM_OPTIONS = [2, 4, 6, 8, 12, 16, 24, 32, 64]
CPU_BRANDS  = ['AMD processor','Intel Core i3','Intel Core i5',
               'Intel Core i7','other intel processor']
HDD_OPTIONS = [0, 32, 500, 1000, 2000]
SSD_OPTIONS = [0, 8, 16, 32, 64, 128, 256, 512, 1000]
GPU_BRANDS  = ['AMD','Intel','Nvidia']
OS_OPTIONS  = ['Mac','Other/No OS/Linux','Windows']

# ─── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem;'>
        <div style='font-size:2.5rem'>💻</div>
        <div style='color:white; font-weight:700; font-size:1.1rem; margin-top:0.5rem'>LaptopPriceAI</div>
        <div style='color:rgba(255,255,255,0.4); font-size:0.78rem'>Configure your laptop specs</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🏷️ Brand & Type")
    company   = st.selectbox("Brand", COMPANIES, index=COMPANIES.index("Dell"))
    type_name = st.selectbox("Laptop Type", TYPES, index=TYPES.index("Notebook"))

    st.markdown("### 🖥️ Display")
    touchscreen = st.selectbox("Touchscreen", ["No", "Yes"])
    ips         = st.selectbox("IPS Panel", ["No", "Yes"])
    ppi         = st.slider("Screen PPI", 80, 350, 141, help="Pixels per inch — higher = sharper")

    st.markdown("### ⚙️ Performance")
    ram       = st.selectbox("RAM (GB)", RAM_OPTIONS, index=RAM_OPTIONS.index(8))
    cpu_brand = st.selectbox("Processor", CPU_BRANDS, index=CPU_BRANDS.index("Intel Core i5"))
    gpu_brand = st.selectbox("GPU Brand", GPU_BRANDS, index=GPU_BRANDS.index("Intel"))

    st.markdown("### 💾 Storage")
    hdd = st.selectbox("HDD (GB)", HDD_OPTIONS, index=2)
    ssd = st.selectbox("SSD (GB)", SSD_OPTIONS, index=SSD_OPTIONS.index(256))

    st.markdown("### 🖱️ System")
    weight = st.slider("Weight (kg)", 0.9, 4.5, 2.1, 0.1)
    os_val = st.selectbox("Operating System", OS_OPTIONS, index=OS_OPTIONS.index("Windows"))

    predict_btn = st.button("🔮 Predict Price", use_container_width=True)

# ─── MAIN ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-box">
    <p class="hero-title">💻 Laptop Price Predictor</p>
    <p class="hero-subtitle">
        Voting Ensemble · Random Forest + Gradient Boosting + XGBoost<br>
        Configure specs in the sidebar → hit <strong>Predict Price</strong>
    </p>
</div>
""", unsafe_allow_html=True)

if not os.path.exists("laptop_data.csv"):
    st.error("⚠️ `laptop_data.csv` not found. Upload it to the same folder as `app.py`.")
    st.stop()

with st.spinner("🔧 Loading data & training model…"):
    df    = load_and_prepare_data()
    model = train_model(df)

col_left, col_right = st.columns([1.3, 1], gap="large")

with col_left:
    st.markdown("### 📋 Your Configuration")
    specs = [
        ("🏷️ Brand",       company),
        ("💻 Type",        type_name),
        ("🧠 CPU",         cpu_brand),
        ("🎮 GPU",         gpu_brand),
        ("🖥️ RAM",        f"{ram} GB"),
        ("💾 SSD",        f"{ssd} GB"),
        ("🗄️ HDD",        f"{hdd} GB"),
        ("📺 Touchscreen", touchscreen),
        ("🔲 IPS Panel",   ips),
        ("📐 PPI",        str(ppi)),
        ("⚖️ Weight",     f"{weight} kg"),
        ("🖱️ OS",          os_val),
    ]
    for i in range(0, len(specs), 3):
        row = specs[i:i+3]
        cols = st.columns(3)
        for j, (label, value) in enumerate(row):
            with cols[j]:
                st.markdown(f"""
                <div class="metric-card">
                    <div style='font-size:0.7rem;color:rgba(255,255,255,0.45);margin-bottom:0.3rem'>{label}</div>
                    <div style='font-size:0.95rem;font-weight:600;color:#e2d9f3'>{value}</div>
                </div>
                """, unsafe_allow_html=True)
        st.write("")

    st.markdown("### 📊 Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Laptops", f"{len(df):,}")
    c2.metric("Avg Price",     f"₹{int(df['Price'].mean()):,}")
    c3.metric("Price Range",   f"₹{int(df['Price'].min()):,} – ₹{int(df['Price'].max()):,}")

with col_right:
    st.markdown("### 🔮 Price Prediction")

    if predict_btn:
        query = pd.DataFrame([[
            company, type_name, int(ram), float(weight),
            1 if touchscreen == "Yes" else 0,
            1 if ips == "Yes" else 0,
            float(ppi),
            cpu_brand, int(hdd), int(ssd),
            gpu_brand, os_val
        ]], columns=[
            "Company", "TypeName", "Ram", "Weight",
            "touchscreen", "IPS Panel", "ppi",
            "cpu brand", "HDD", "SSD",
            "GPU Brand", "OS"
        ])

        log_price = model.predict(query)[0]
        price     = int(np.exp(log_price))
        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">ESTIMATED PRICE</div>
            <div class="result-price">₹{price:,}</div>
            <div class="result-range">Range: ₹{int(price*0.9):,} – ₹{int(price*1.1):,}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 🔍 Similar Laptops in Dataset")
        mask = (df["Company"] == company) & (df["TypeName"] == type_name) & (df["Ram"] == ram)
        similar = df[mask].copy()
        if len(similar):
            similar["Price (₹)"] = similar["Price"].apply(lambda x: f"₹{int(x):,}")
            show = [c for c in ["Company","TypeName","Ram","SSD","HDD","OS","Price (₹)"] if c in similar.columns]
            st.dataframe(similar[show].head(5).reset_index(drop=True), use_container_width=True, hide_index=True)
        else:
            st.info("No exact matches found for this combination.")
    else:
        st.markdown("""
        <div style='border:2px dashed rgba(102,126,234,0.3);border-radius:16px;
                    padding:3rem 2rem;text-align:center;color:rgba(255,255,255,0.35);'>
            <div style='font-size:3rem'>🎯</div>
            <div style='margin-top:1rem'>Configure specs in the sidebar<br>
            and hit <strong style='color:#a78bfa'>Predict Price</strong></div>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("ℹ️ How does this model work?"):
        st.markdown("""
        **Voting Ensemble** of three models:

        | Model | Weight |
        |-------|--------|
        | 🌲 Random Forest (100 trees, depth 17) | 5 |
        | 📈 Gradient Boosting (150 estimators)  | 1 |
        | ⚡ XGBoost (200 estimators)            | 1 |

        Target is `log(Price)` to stabilise variance. Categorical features
        (Brand, Type, CPU, GPU, OS) are OneHotEncoded. Numeric features
        (RAM, SSD, HDD, Weight, PPI, Touchscreen, IPS) are passed through directly.
        """)

st.markdown("""
<div style='text-align:center;color:rgba(255,255,255,0.25);font-size:0.8rem;padding:2rem 0 1rem'>
    Built with ❤️ using Streamlit · Scikit-learn · XGBoost
</div>
""", unsafe_allow_html=True)
