"""
app.py
------
Streamlit Live Demonstration Dashboard for Xilinx VC707 FPGA (RISC-V Edge-AI SoC).
Compact Zero-Scroll Demonstration Cockpit for Real Silicon & Offline Simulation.
"""

import os
import glob
import time
import numpy as np
#from pyparsing import line
import streamlit as st
from PIL import Image

from serial_bridge import (
    list_available_com_ports,
    stream_hardware_inference,
    VC707TelemetryParser
)

st.set_page_config(
    page_title="Edge-AI Hardware Acceleration",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for zero-scroll compact layout
st.markdown("""
<style>
    /* Synchronize sidebar and main container top padding to align on the exact same line */
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 0.6rem !important;
    }
    .block-container {
        padding-top: 0.6rem !important;
        padding-bottom: 0.4rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 100% !important;
    }
    header { visibility: hidden !important; height: 0 !important; }
    footer { visibility: hidden !important; height: 0 !important; }
    
    /* Compact typography with matched top margin */
    h1 { margin-top: 0 !important; margin-bottom: 0.1rem !important; padding-top: 0 !important; font-size: 1.55rem !important; line-height: 1.2 !important; }
    h2, h3 { margin-top: 0.2rem !important; margin-bottom: 0.15rem !important; font-size: 1.05rem !important; }
    p { margin-bottom: 0.2rem !important; }
    
    /* Metrics compact styling */
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.03);
        padding: 6px 10px !important;
        border-radius: 6px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    div[data-testid="stMetricValue"] { font-size: 1.15rem !important; font-weight: 700 !important; }
    div[data-testid="stMetricLabel"] { font-size: 0.72rem !important; margin-bottom: -4px !important; }
    
    /* Compact dividers and widgets */
    hr { margin-top: 0.35rem !important; margin-bottom: 0.35rem !important; }
    .stProgress > div > div > div > div { height: 10px !important; }
    div[data-testid="stExpander"] { margin-top: 0.2rem !important; }
</style>
""", unsafe_allow_html=True)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(THIS_DIR, "sample_images")

# Softmax calculation helper
def softmax(logits_dict):
    names = list(logits_dict.keys())
    vals = np.array(list(logits_dict.values()), dtype=np.float64)
    vals = vals - np.max(vals)
    exp_vals = np.exp(vals / 50.0) # Temperature scaling for integer logit display
    probs = exp_vals / np.sum(exp_vals)
    return {name: float(p) for name, p in zip(names, probs)}

# --- Sidebar Controls ---
st.sidebar.title("🎛️ Control Panel")


# Serial Port Selection
available_ports = list_available_com_ports() # List of (port_id, label)
port_ids = [p[0] for p in available_ports]
port_labels = [p[1] for p in available_ports]

selected_idx = st.sidebar.selectbox(
    "🔌 Select VC707 COM / TTY Port",
    range(len(port_ids)),
    format_func=lambda i: port_labels[i],
    index=0
)
selected_port = port_ids[selected_idx]
selected_label = port_labels[selected_idx]

baud_rate = st.sidebar.selectbox(
    "⚡ Baud Rate",
    [115200, 921600, 57600],
    index=0,
    help="115200 is the standard hardware UART speed configured in the VC707 RISC-V SoC firmware."
)

if selected_port == "SIMULATION_MODE":
    st.sidebar.warning("🟡 Mode: Hardware Simulator (Offline)")
else:
    st.sidebar.success(f"🟢 Target Port: `{selected_port}`")
    st.sidebar.caption(f"Device: {selected_label}")
# --- Reconnect button (force-close cached serial handle) ---
if st.sidebar.button("🔄 Reconnect board"):
    if "serial_conn" in st.session_state:
        try:
            st.session_state["serial_conn"].close()
        except Exception:
            pass
        del st.session_state["serial_conn"]
    st.rerun()
    
st.sidebar.markdown("---")
st.sidebar.subheader("🖼️ Select Input Image")

sample_options = {
    "Sample #0: ✈️ Airplane": ("sample_0_airplane.png", "airplane", 0),
    "Sample #1: 🚗 Automobile": ("sample_1_automobile.png", "automobile", 1),
    "Sample #2: 🐦 Bird": ("sample_2_bird.png", "bird", 2),
    "Sample #3: 🐱 Cat": ("sample_3_cat.png", "cat", 3),
    "Sample #4: 🦌 Deer": ("sample_4_deer.png", "deer", 4),
    "Sample #5: 🐶 Dog": ("sample_5_dog.png", "dog", 5),
    "Sample #6: 🐸 Frog": ("sample_6_frog.png", "frog", 6),
    "Sample #7: 🐎 Horse": ("sample_7_horse.png", "horse", 7),
    "Sample #8: 🚢 Ship": ("sample_8_ship.png", "ship", 8),
    "Sample #9: 🚚 Truck": ("sample_9_truck.png", "truck", 9),
}

chosen_sample_key = st.sidebar.selectbox("Built-in Samples (0 ms Transfer):", list(sample_options.keys()), index=0)
img_file, ground_truth, sample_idx = sample_options[chosen_sample_key]
img_path = os.path.join(SAMPLE_DIR, img_file)

uploaded_file = st.sidebar.file_uploader("Or Upload Custom Image:", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    ground_truth = "unknown"      # we don't know what this is
    sample_idx   = -1              # signal "custom", not one of 0-9
    st.sidebar.success(f"📸 Custom image loaded: `{uploaded_file.name}`")
    st.sidebar.info("Ground truth is unknown — waiting for hardware prediction.")
else:
    img = Image.open(img_path) if os.path.exists(img_path) else None

run_button = st.sidebar.button("🚀 Execute", type="primary", use_container_width=True)




# --- Main Dashboard Header ---
st.title("⚡ Edge-AI Hardware Acceleration")
st.caption("Real-Time Quantized VGG-11 (8-bit Uniform) Deep Neural Network Inference on 64-bit RISC-V SoC")

# --- Top Section: Image Preview & Top-5 Confidences ---
col_img, col_top5 = st.columns([0.8, 1.4])

with col_img:
    st.subheader("🖼️ Input Image")
    if img is not None:
        st.image(img, caption=f"Sample #{sample_idx}: {ground_truth.upper()}", width=160)
    else:
        st.info("No image loaded")

with col_top5:
    st.subheader("📊 Top-5 Classification Confidence")
    top5_container = st.empty()

# --- Middle Section: Real-Time Layer Progress ---
st.markdown("---")
progress_bar = st.progress(0.0)
status_text = st.empty()

# --- Hardware Performance Metrics Row ---
metrics_container = st.empty()

# --- Bottom Section: Telemetry Log (Compact Expander) ---
with st.expander("📜 Live Hardware UART Serial Stream (Raw Telemetry)", expanded=False):
    log_box = st.container(height=300)   # fixed-height scrollable box
    terminal_log_container = log_box.empty()

# --- Execution Handling ---
if run_button:
    status_text.info(f"Connecting to VC707 & Triggering Inference on Image #{sample_idx} ({ground_truth.upper()})...")
    raw_terminal_lines = []

    def ui_callback(parser: VC707TelemetryParser, line: str):
        raw_terminal_lines.append(line)
        terminal_log_container.code("\n".join(raw_terminal_lines), language="bash")
        
        # Update progress
        progress_bar.progress(min(1.0, parser.progress_pct / 100.0))
        if parser.current_layer:
            status_text.success(f"⚙️ Computing: **{parser.current_layer}** ({parser.progress_pct:.1f}%)")

        # Update Top-5 chart if available
        if parser.top5_scores:
            probs = softmax(parser.top5_scores)
            sorted_top = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
            
            with top5_container.container():
                for rank, (cname, prob) in enumerate(sorted_top, 1):
                    emoji = "🐱" if cname == "cat" else "🐶" if cname == "dog" else "✈️" if cname == "airplane" else "🚗" if cname == "automobile" else "🚢" if cname == "ship" else "🏷️"
                    st.write(f"**#{rank}: {emoji} {cname.capitalize()}** — `{prob*100:.1f}%`")
                    st.progress(prob)

    custom_bytes = None
    if uploaded_file is not None and img is not None:
        img_32 = img.resize((32, 32), Image.Resampling.BILINEAR)
        arr = np.array(img_32, dtype=np.float32) / 255.0
        mean_c = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 1, 3)
        std_c  = np.array([0.2470, 0.2435, 0.2616]).reshape(1, 1, 3)
        norm = (arr - mean_c) / std_c
        norm_chw = norm.transpose(2, 0, 1) # (3, 32, 32) CHW
        in_scale = 0.035773955285549164
        q = np.clip(np.round(norm_chw / in_scale), -128, 127).astype(np.int8)
        custom_bytes = q.tobytes()

    final_parser = stream_hardware_inference(
        port=selected_port,
        baud_rate=baud_rate,
        target_class=ground_truth,
        image_idx=sample_idx,
        custom_raw_bytes=custom_bytes,
        callback=ui_callback
    )

    progress_bar.progress(1.0)
    status_text.success(f"✅ Hardware Inference Completed! [Status: {final_parser.status}]")

    # Render Final Performance Metrics (5 KPI Cards including CPI)
    with metrics_container.container():
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        m_col1.metric("⏱️ Latency", f"{final_parser.latency_ms:.2f} ms")
        m_col2.metric("🔄 Total Cycles", f"{final_parser.cycles:,}")
        m_col3.metric("🔢 Instructions", f"{final_parser.instructions:,}")
        m_col4.metric("⚖️ CPI", f"{final_parser.cpi:.2f}")
        throughput = (1000.0 / final_parser.latency_ms) if final_parser.latency_ms > 0 else 70.4
        m_col5.metric("⏳ Time / image", f"{final_parser.latency_ms/1000:.2f} s")
else:
    top5_container.info("Click **'🚀 Execute'** in the sidebar to run on hardware.")
    with metrics_container.container():
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        m_col1.metric("⏱️ Latency",       "—")
        m_col2.metric("🔄 Clock Cycles",  "—")
        m_col3.metric("🔢 Instructions",  "—")
        m_col4.metric("⚖️ CPI",           "—")
        m_col5.metric("⏳ Time / image",  "—")
