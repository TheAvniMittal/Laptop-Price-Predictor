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

/* Hero */
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

/* Cards */
.card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(8px);
}
.card-title {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #a78bfa;
    margin-bottom: 0.8rem;
}

/* Result box */
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

/* Metric cards */
.metric-row { display: flex; gap: 1rem; margin-top: 1rem; }
.metric-card {
    flex: 1;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.metric-val { font-size: 1.5rem; font-weight: 600; color: #a78bfa; }
.metric-lbl { font-size: 0.72rem; color: rgba(255,255,255,0.5); margin-top: 0.2rem; }

/* Sidebar tweaks */
section[data-testid="stSidebar"] {
    background: rgba(15,12,41,0.85) !important;
    border-right: 1px solid rgba(102,126,234,0.2);
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stNumberInput label {
    color: rgba(255,255,255,0.8) !important;
    font-size: 0.85rem;
}

/* Predict button */
div.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2.5rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    width: 100%;
    letter-spacing: 0.04em;
    transition: transform 0.2s, box-shadow 0.2s;
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(102,126,234,0.5) !important;
}

/* Selectbox & inputs */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(102,126,234,0.3) !important;
    border-radius: 10px !important;
    color: white !important;
}

/* Badge */
.badge {
    display: inline-block;
    background: rgba(102,126,234,0.25);
    border: 1px solid rgba(102,126,234,0.5);
    color: #a78bfa;
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 0.2rem;
}

/* Info tags */
.tag-row { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.6rem; }

h1,h2,h3 { color: white; }
p, li { color: rgba(255,255,255,0.7); }
</style>
""", unsafe_allow_html=True)

# ─── DATA & MODEL ───────────────────────────────────────────────────────────────
MODEL_FILE = "laptop_model.pkl"

@st.cache_data
def load_and_prepare_data():
    """Load CSV and replicate all preprocessing from the notebook."""
    df = pd.read_csv("laptop_data.csv")
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    df["Ram"] = df["Ram"].str.replace("GB", "").astype("int64")
    df["Weight"] = df["Weight"].str.replace("kg", "").astype("float64")

    # Touchscreen & IPS
    df["touchscreen"] = df["ScreenResolution"].apply(lambda x: 1 if "Touchscreen" in x else 0)
    df["IPS Panel"]   = df["ScreenResolution"].apply(lambda x: 1 if "IPS Panel"    in x else 0)

    # PPI
    new = df["ScreenResolution"].str.split("x", n=1, expand=True)
    df["x_res"] = new[0].str.findall(r"(\d+\.?\d+)").apply(lambda x: x[0]).astype("int64")
    df["y_res"] = new[1].astype("int64")
    df["ppi"] = (((df["x_res"]**2) + (df["y_res"]**2))**0.5 / df["Inches"]).astype("float")
    df.drop(columns=["ScreenResolution","x_res","y_res","Inches"], inplace=True)

    # CPU brand
    df["CPU"] = df["Cpu"].apply(lambda x: " ".join(x.split()[0:3]))
    def fetch_processor(text):
        if text in ["Intel Core i5","Intel Core i7","Intel Core i3"]:
            return text
        elif text.split()[0] == "Intel":
            return "other intel processor"
        else:
            return "AMD processor"
    df["cpu brand"] = df["CPU"].apply(fetch_processor)
    df.drop(columns=["Cpu","CPU"], inplace=True)

    # Memory
    df["Memory"] = df["Memory"].astype(str).replace(r"\.0","",regex=True)
    df["Memory"] = df["Memory"].str.replace("GB"," ").str.replace("TB","000")
    mem_split = df["Memory"].str.split("+", n=1, expand=True)
    df["first"] = mem_split[0].str.strip()
    df["second"] = mem_split[1]

    for col in ["first","second"]:
        df[f"layer1HDD_{col}"] = df["first"].apply(lambda x: 1 if "HDD" in str(x) else 0)
        df[f"layer1SSD_{col}"] = df["first"].apply(lambda x: 1 if "SSD" in str(x) else 0)

    df["layer1HDD"] = df["first"].apply(lambda x: 1 if "HDD" in str(x) else 0)
    df["layer1SSD"] = df["first"].apply(lambda x: 1 if "SSD" in str(x) else 0)
    df["layer2HDD"] = df["second"].apply(lambda x: 1 if "HDD" in str(x) else 0)
    df["layer2SSD"] = df["second"].apply(lambda x: 1 if "SSD" in str(x) else 0)

    df["first"]  = df["first"].str.replace(r"\D"," ",regex=True).str.strip()
    df["second"] = df["second"].fillna("0").str.replace(r"\D"," ",regex=True).str.strip()
    df["first"]  = df["first"].astype(int)
    df["second"] = df["second"].astype(int)

    df["HDD"] = df["first"]*df["layer1HDD"] + df["second"]*df["layer2HDD"]
    df["SSD"] = df["first"]*df["layer1SSD"] + df["second"]*df["layer2SSD"]
    df.drop(columns=["first","second","layer1HDD","layer1SSD","layer2HDD","layer2SSD",
                     "layer1HDD_first","layer1SSD_first","layer1HDD_second","layer1SSD_second"], errors="ignore", inplace=True)

    # GPU brand
    df["GPU Brand"] = df["Gpu"].apply(lambda x: x.split()[0])
    df = df[df["GPU Brand"] != "ARM"]
    df.drop(columns=["Gpu"], inplace=True)

    # OS
    def cat_os(inp):
        if inp in ["Windows 10","Windows 7","Windows 10 S"]:
            return "Windows"
        elif inp in ["macOS","Mac OS X"]:
            return "Mac"
        else:
            return "Other/No OS/Linux"
    df["OS"] = df["OpSys"].apply(cat_os)
    df.drop(columns=["OpSys"], inplace=True)

    return df

@st.cache_resource
def train_model(df):
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE,"rb") as f:
            return pickle.load(f)

    x = df.drop(columns=["Price"])
    y = np.log(df["Price"])
    cat_cols = [0, 1, 7, 10, 11]   # Company, TypeName, cpu brand, GPU Brand, OS

    step1 = ColumnTransformer(transformers=[
        ("ohe", OneHotEncoder(sparse_output=False, drop="first"), cat_cols)
    ], remainder="passthrough")

    rf   = RandomForestRegressor(n_estimators=100, max_depth=17, random_state=42, oob_score=True)
    gbdt = GradientBoostingRegressor(learning_rate=0.1, n_estimators=150, max_depth=5, random_state=42)
    xgb  = XGBRegressor(max_depth=5, learning_rate=0.4, n_estimators=200, random_state=42, eval_metric="logloss")

    step2 = VotingRegressor([("rf",rf),("gbr",gbdt),("xgb",xgb)], weights=[5,1,1])
    pipe  = Pipeline([("step1",step1),("step2",step2)])
    pipe.fit(x, y)

    with open(MODEL_FILE,"wb") as f:
        pickle.dump(pipe, f)
    return pipe

# ─── UNIQUE VALUES FROM DATASET ─────────────────────────────────────────────────
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

# ─── SIDEBAR INPUTS ─────────────────────────────────────────────────────────────
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
    touchscreen = st.selectbox("Touchscreen", ["No","Yes"])
    ips         = st.selectbox("IPS Panel", ["No","Yes"])
    ppi         = st.slider("Screen PPI (pixels per inch)", 80, 350, 141,
                             help="Higher PPI = sharper display")

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

# ─── MAIN AREA ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-box">
    <p class="hero-title">💻 Laptop Price Predictor</p>
    <p class="hero-subtitle">
        Powered by a Voting Ensemble (Random Forest + Gradient Boosting + XGBoost)<br>
        Configure specs in the sidebar → hit <strong>Predict Price</strong>
    </p>
</div>
""", unsafe_allow_html=True)

# Check for CSV
if not os.path.exists("laptop_data.csv"):
    st.warning("⚠️ **`laptop_data.csv` not found.** Place it in the same folder as `app.py` and re-run.")
    st.stop()

# Load & train
with st.spinner("🔧 Loading data & training model (first run only)…"):
    df    = load_and_prepare_data()
    model = train_model(df)

# ─── TWO-COLUMN LAYOUT ──────────────────────────────────────────────────────────
col_left, col_right = st.columns([1.3, 1], gap="large")

with col_left:
    st.markdown("### 📋 Your Configuration")

    # Summary cards
    specs = [
        ("🏷️ Brand",       company),
        ("💻 Type",        type_name),
        ("🧠 CPU",         cpu_brand),
        ("🎮 GPU",         gpu_brand),
        ("🖥️ RAM",        f"{ram} GB"),
        ("💾 SSD",        f"{ssd} GB"),
        ("🗄️ HDD",        f"{hdd} GB"),
        ("📺 Touchscreen", "Yes" if touchscreen=="Yes" else "No"),
        ("🔲 IPS Panel",   "Yes" if ips=="Yes" else "No"),
        ("📐 PPI",        f"{ppi}"),
        ("⚖️ Weight",     f"{weight} kg"),
        ("🖱️ OS",          os_val),
    ]

    cols_per_row = 3
    for i in range(0, len(specs), cols_per_row):
        row_specs = specs[i:i+cols_per_row]
        cols = st.columns(cols_per_row)
        for j, (icon_label, value) in enumerate(row_specs):
            with cols[j]:
                st.markdown(f"""
                <div class="metric-card">
                    <div style='font-size:0.7rem;color:rgba(255,255,255,0.45);margin-bottom:0.3rem'>{icon_label}</div>
                    <div style='font-size:0.95rem;font-weight:600;color:#e2d9f3'>{value}</div>
                </div>
                """, unsafe_allow_html=True)
        st.write("")

    # Dataset insight
    st.markdown("### 📊 Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Laptops", f"{len(df):,}")
    c2.metric("Avg Price", f"₹{int(df['Price'].mean()):,}")
    c3.metric("Price Range", f"₹{int(df['Price'].min()):,}–₹{int(df['Price'].max()):,}")

with col_right:
    st.markdown("### 🔮 Price Prediction")

    if predict_btn:
        ts  = 1 if touchscreen == "Yes" else 0
        ips_val = 1 if ips == "Yes" else 0

        # Build the same feature order as notebook's x
        # Columns: Company, TypeName, Ram, Weight, touchscreen, IPS Panel, ppi,
        #          cpu brand, HDD, SSD, GPU Brand, OS
        query = pd.DataFrame([[
            company, type_name, ram, weight,
            ts, ips_val, ppi,
            cpu_brand, hdd, ssd,
            gpu_brand, os_val
        ]], columns=[
            "Company","TypeName","Ram","Weight",
            "touchscreen","IPS Panel","ppi",
            "cpu brand","HDD","SSD",
            "GPU Brand","OS"
        ])

        log_price = model.predict(query)[0]
        price     = int(np.exp(log_price))
        low       = int(price * 0.90)
        high      = int(price * 1.10)

        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">ESTIMATED PRICE</div>
            <div class="result-price">₹{price:,}</div>
            <div class="result-range">Range: ₹{low:,} – ₹{high:,}</div>
        </div>
        """, unsafe_allow_html=True)

        # Similar laptops
        st.markdown("#### 🔍 Similar Laptops in Dataset")
        mask = (
            (df["Company"] == company) &
            (df["TypeName"] == type_name) &
            (df["Ram"] == ram)
        )
        similar = df[mask].copy()
        if len(similar) > 0:
            similar["Price (₹)"] = similar["Price"].apply(lambda x: f"₹{int(x):,}")
            show_cols = ["Company","TypeName","Ram","SSD","HDD","OS","Price (₹)"]
            show_cols = [c for c in show_cols if c in similar.columns]
            st.dataframe(
                similar[show_cols].head(5).reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No exact matches found in the dataset for this combination.")

    else:
        st.markdown("""
        <div style='
            border: 2px dashed rgba(102,126,234,0.3);
            border-radius: 16px;
            padding: 3rem 2rem;
            text-align: center;
            color: rgba(255,255,255,0.35);
        '>
            <div style='font-size:3rem'>🎯</div>
            <div style='margin-top:1rem; font-size:1rem'>
                Configure your specs in the sidebar<br>and hit <strong style='color:#a78bfa'>Predict Price</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # How it works
    with st.expander("ℹ️ How does this model work?"):
        st.markdown("""
        This app uses a **Voting Ensemble** of three models:

        | Model | Weight |
        |-------|--------|
        | 🌲 Random Forest (100 trees, depth 17) | 5 |
        | 📈 Gradient Boosting (150 estimators) | 1 |
        | ⚡ XGBoost (200 estimators) | 1 |

        **Target**: `log(Price)` — log-transform stabilises variance.

        **Categorical encoding**: OneHotEncoder on Company, TypeName, CPU Brand, GPU Brand, OS.

        **Key features**: RAM, SSD, HDD, Weight, PPI, Touchscreen, IPS Panel.
        """)

# ─── FOOTER ──────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; color:rgba(255,255,255,0.25); font-size:0.8rem; padding:1rem 0'>
    Built with ❤️ using Streamlit · Scikit-learn · XGBoost
</div>
""", unsafe_allow_html=True)
