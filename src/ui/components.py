import streamlit as st

def render_cyber_header():
    """Renders the top digital twin title banner."""
    st.markdown(
        """
        <div class="cyber-header">
            <div class="cyber-title">SmartFlow // Edge AI Digital Twin</div>
            <div class="cyber-subtitle">> COGNITIVE CLOSED-LOOP MITIGATION ENGINE // LOCAL HOST ACTIVE</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_metric_card(title, value, unit="", status_color=""):
    """
    Renders a custom glassmorphism metric widget.
    status_color options: 'green', 'amber', 'red', or empty
    """
    color_style = ""
    if status_color == "green":
        color_style = "color: #00e676; text-shadow: 0 0 10px rgba(0, 230, 118, 0.3);"
    elif status_color == "amber":
        color_style = "color: #ff9100; text-shadow: 0 0 10px rgba(255, 145, 0, 0.3);"
    elif status_color == "red":
        color_style = "color: #ff1744; text-shadow: 0 0 15px rgba(255, 23, 68, 0.4);"
        
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="card-title">
                <span>⚡</span> {title}
            </div>
            <div class="metric-value" style="{color_style}">
                {value} <span class="metric-unit">{unit}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_ai_analysis_card(ai_response):
    """
    Renders the local Ollama / Cognitive engine analysis card.
    Expects response dictionary with assessment, mitigation_action, gcode_command, material_saved_grams, confidence.
    """
    mitigation = ai_response.get("mitigation_action", "CONTINUE PRINT")
    gcode = ai_response.get("gcode_command", "NONE")
    saved_weight = ai_response.get("material_saved_grams", 0.0)
    confidence = ai_response.get("confidence", 1.0)
    engine = ai_response.get("engine", "Edge Engine")
    latency = ai_response.get("latency_ms", 0.0)
    
    # Class mapping for mitigation alert level
    if mitigation == "EMERGENCY HALT":
        badge_class = "status-critical"
        icon = "🚨"
    elif mitigation == "REDUCE FEED RATE & TEMP":
        badge_class = "status-warning"
        icon = "⚠️"
    else:
        badge_class = "status-safe"
        icon = "✅"
        
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="card-title">
                <span>🧠</span> Edge AI Intelligence Layer
            </div>
            
            <div style="margin-bottom: 12px; font-size: 0.9rem; color: #c5c6c7; line-height: 1.4;">
                <strong style="color: #66fcf1;">SYSTEM ASSESSMENT:</strong><br/>
                <em>"{ai_response.get('assessment', 'Print operating under stable conditions.')}"</em>
            </div>
            
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; background: rgba(0,0,0,0.2); padding: 8px 12px; border-radius: 6px;">
                <span style="font-size: 0.85rem; color: #8f9aa5; font-family: 'Share Tech Mono', monospace;">Mitigation Action:</span>
                <span class="status-badge {badge_class}">{icon} {mitigation}</span>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 15px;">
                <div style="background: rgba(0,0,0,0.15); padding: 8px; border-radius: 6px; border-left: 2px solid #66fcf1;">
                    <div style="font-size: 0.75rem; color: #8f9aa5; font-family: 'Share Tech Mono', monospace;">G-CODE TRIGGER</div>
                    <div style="font-family: 'Share Tech Mono', monospace; font-size: 0.9rem; color: #ffffff; font-weight: bold; margin-top: 3px;">{gcode}</div>
                </div>
                <div style="background: rgba(0,0,0,0.15); padding: 8px; border-radius: 6px; border-left: 2px solid #00e676;">
                    <div style="font-size: 0.75rem; color: #8f9aa5; font-family: 'Share Tech Mono', monospace;">FILAMENT SAVED</div>
                    <div style="font-family: 'Orbitron', sans-serif; font-size: 0.9rem; color: #00e676; font-weight: bold; margin-top: 3px;">{saved_weight} <span style="font-size: 0.7rem;">grams</span></div>
                </div>
            </div>
            
            <div style="display: flex; justify-content: space-between; font-size: 0.7rem; color: #8f9aa5; border-top: 1px solid rgba(255, 255, 255, 0.05); padding-top: 8px; font-family: 'Share Tech Mono', monospace;">
                <span>ENGINE: {engine}</span>
                <span>LATENCY: {latency}ms</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_terminal_console(events):
    """
    Renders the retro command console displaying raw pipeline system logs.
    events list expects tuples of (timestamp_string, type, text)
    type options: 'info', 'warn', 'error', 'success'
    """
    log_lines = ""
    for ts, event_type, text in reversed(events):
        type_class = ""
        if event_type == "warn":
            type_class = "class='terminal-warn'"
        elif event_type == "error":
            type_class = "class='terminal-crit'"
        elif event_type == "success":
            type_class = "class='terminal-success'"
            
        log_lines += f"<div class='terminal-line'><span class='terminal-timestamp'>[{ts}]</span><span {type_class}>{text}</span></div>"
        
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="card-title">
                <span>💻</span> Physical Device Console Logs
            </div>
            <div class="terminal-console">
                {log_lines if log_lines else "<div class='terminal-line' style='color:#8f9aa5;'>> System initializing. Awaiting camera frame stream...</div>"}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
