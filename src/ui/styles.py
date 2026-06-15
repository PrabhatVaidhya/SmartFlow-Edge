import streamlit as st

def inject_premium_css():
    """
    Injects high-end, premium cyber-industrial CSS overrides into Streamlit.
    Features: Glassmorphism containers, neon-glow accents, custom typography,
    pulsing status indicators, and localized system console panels.
    """
    custom_css = """
    <style>
    /* Import Premium Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Inter:wght@300;400;600;700&display=swap');

    /* Global Typography & Background Adjustments */
    .stApp {
        background-color: #0b0c10 !important;
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Orbitron', sans-serif !important;
        letter-spacing: 1px !important;
        font-weight: 700 !important;
    }
    
    /* Hide Default Streamlit Elements for a clean Digital Twin display */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Modern Cyber Title Grid Header */
    .cyber-header {
        background: linear-gradient(135deg, #1f2833 0%, #0b0c10 100%);
        border: 1px solid #66fcf1;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 25px;
        box-shadow: 0 0 25px rgba(102, 252, 241, 0.15);
        position: relative;
        overflow: hidden;
    }
    
    .cyber-header::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, #66fcf1, #45f3ff, #ff007f);
    }
    
    .cyber-title {
        color: #66fcf1;
        font-size: 2.2rem;
        margin: 0;
        text-transform: uppercase;
        font-weight: 900 !important;
        text-shadow: 0 0 15px rgba(102, 252, 241, 0.4);
    }
    
    .cyber-subtitle {
        color: #c5c6c7;
        font-size: 0.95rem;
        margin-top: 5px;
        margin-bottom: 0;
        font-family: 'Share Tech Mono', monospace;
    }

    /* Glassmorphism Panel Cards */
    .glass-card {
        background: rgba(31, 40, 51, 0.45);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    }
    
    .glass-card:hover {
        border-color: rgba(102, 252, 241, 0.3);
        box-shadow: 0 12px 40px 0 rgba(102, 252, 241, 0.08);
        transform: translateY(-2px);
    }
    
    .card-title {
        color: #c5c6c7;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 15px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
        font-family: 'Orbitron', sans-serif;
    }

    /* glowing status tags */
    .status-badge {
        font-family: 'Share Tech Mono', monospace;
        font-size: 1rem;
        padding: 4px 12px;
        border-radius: 4px;
        font-weight: bold;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    
    .status-safe {
        background-color: rgba(0, 230, 118, 0.15);
        color: #00e676;
        border: 1px solid rgba(0, 230, 118, 0.35);
        box-shadow: 0 0 10px rgba(0, 230, 118, 0.15);
    }
    
    .status-warning {
        background-color: rgba(255, 145, 0, 0.15);
        color: #ff9100;
        border: 1px solid rgba(255, 145, 0, 0.35);
        box-shadow: 0 0 10px rgba(255, 145, 0, 0.15);
    }
    
    .status-critical {
        background-color: rgba(255, 23, 68, 0.15);
        color: #ff1744;
        border: 1px solid rgba(255, 23, 68, 0.35);
        box-shadow: 0 0 15px rgba(255, 23, 68, 0.3);
        animation: pulse-red 1.5s infinite alternate;
    }
    
    @keyframes pulse-red {
        0% { box-shadow: 0 0 5px rgba(255, 23, 68, 0.2); }
        100% { box-shadow: 0 0 20px rgba(255, 23, 68, 0.6); }
    }
    
    @keyframes blink {
        0% { opacity: 0.2; }
        100% { opacity: 1.0; }
    }

    /* Glowing Big Metric Values */
    .metric-value {
        font-family: 'Orbitron', sans-serif;
        font-size: 2.2rem;
        font-weight: 900;
        color: #ffffff;
        margin: 5px 0;
        text-shadow: 0 0 8px rgba(255, 255, 255, 0.1);
    }
    
    .metric-unit {
        font-size: 0.8rem;
        color: #c5c6c7;
        font-family: 'Share Tech Mono', monospace;
    }

    /* Retro Terminal Console Panel */
    .terminal-console {
        background: #050508;
        border: 1px solid #1f2833;
        border-left: 3px solid #66fcf1;
        border-radius: 6px;
        padding: 15px;
        font-family: 'Share Tech Mono', monospace;
        color: #66fcf1;
        height: 250px;
        overflow-y: auto;
        box-shadow: inset 0 0 15px rgba(0, 0, 0, 0.8);
    }
    
    .terminal-line {
        margin-bottom: 6px;
        font-size: 0.85rem;
        line-height: 1.3;
    }
    
    .terminal-timestamp {
        color: #8f9aa5;
        margin-right: 8px;
    }
    
    .terminal-warn { color: #ff9100; }
    .terminal-crit { color: #ff1744; font-weight: bold; }
    .terminal-success { color: #00e676; }

    /* Control Panel Buttons style override */
    div.stButton > button {
        background: linear-gradient(135deg, #1f2833 0%, #0f141c 100%) !important;
        color: #66fcf1 !important;
        border: 1px solid #66fcf1 !important;
        border-radius: 6px !important;
        padding: 10px 24px !important;
        font-family: 'Orbitron', sans-serif !important;
        font-weight: bold !important;
        letter-spacing: 1px !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        text-shadow: 0 0 5px rgba(102, 252, 241, 0.3) !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3) !important;
    }
    
    div.stButton > button:hover {
        background: #66fcf1 !important;
        color: #0b0c10 !important;
        box-shadow: 0 0 15px rgba(102, 252, 241, 0.6) !important;
        transform: translateY(-1px) !important;
    }

    div.stButton > button:active {
        transform: translateY(1px) !important;
    }

    /* Critical Halt Button Variant */
    .halt-btn div.stButton > button {
        background: linear-gradient(135deg, #400d12 0%, #1a0305 100%) !important;
        color: #ff1744 !important;
        border: 1px solid #ff1744 !important;
        text-shadow: 0 0 5px rgba(255, 23, 68, 0.3) !important;
    }
    
    .halt-btn div.stButton > button:hover {
        background: #ff1744 !important;
        color: #ffffff !important;
        box-shadow: 0 0 20px rgba(255, 23, 68, 0.8) !important;
    }

    /* Custom scrollbars */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #0b0c10;
    }
    ::-webkit-scrollbar-thumb {
        background: #1f2833;
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #66fcf1;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
