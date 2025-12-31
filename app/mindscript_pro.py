"""
MindScript Pro - Recruiter Impressive Edition
Features: Character Stats Card, Hologram Display, 3D Visualization
"""

import gradio as gr
import requests
import json

API_URL = "http://localhost:8000"

def analyze(text):
    if len(text.split()) < 10:
        return "<div style='color: #ff4757; text-align: center; padding: 50px;'>❌ Enter at least 10 words</div>"
    
    try:
        response = requests.post(f"{API_URL}/analyze", json={"text": text}, timeout=30)
        if response.status_code != 200:
            return "<div style='color: #ff4757;'>❌ API Error - Make sure server is running on port 8000</div>"
        
        r = response.json()
        dims = r['dimensions']
        
        # Calculate level based on average
        avg_score = sum(dims.values()) / len(dims)
        level = int(avg_score * 100)
        
        # Get rank
        if avg_score > 0.8: rank = "LEGENDARY"
        elif avg_score > 0.6: rank = "ELITE"
        elif avg_score > 0.4: rank = "ADVANCED"
        else: rank = "NOVICE"
        
        # Character class based on top dimension
        classes = {
            'analytical': ('🧙‍♂️', 'STRATEGIST', '#3498db'),
            'creative': ('🎨', 'VISIONARY', '#9b59b6'),
            'social': ('👑', 'LEADER', '#e74c3c'),
            'structured': ('⚔️', 'ARCHITECT', '#2ecc71'),
            'emotional': ('💫', 'EMPATH', '#f39c12')
        }
        
        top_dim = max(dims, key=dims.get)
        char_icon, char_class, char_color = classes.get(top_dim, ('🎯', 'EXPLORER', '#1abc9c'))
        
        # Word stats
        words = len(text.split())
        chars = len(text)
        
        html = f"""
        <div class="mindscript-container">
            
            <!-- HEADER HOLOGRAM -->
            <div class="hologram-header">
                <div class="scan-line"></div>
                <div class="glitch-text" data-text="MINDSCRIPT">MINDSCRIPT</div>
                <div class="hologram-subtitle">COGNITIVE ANALYSIS SYSTEM v2.0</div>
                <div class="hologram-status">
                    <span class="status-dot"></span> ANALYSIS COMPLETE
                </div>
            </div>
            
            <!-- MAIN CONTENT -->
            <div class="main-grid">
                
                <!-- LEFT: 3D CHARACTER CARD -->
                <div class="card-3d-container">
                    <div class="card-3d">
                        <div class="card-front">
                            <div class="card-header">
                                <div class="card-rank">{rank}</div>
                                <div class="card-level">LVL {level}</div>
                            </div>
                            
                            <div class="card-avatar">
                                <div class="avatar-glow"></div>
                                <div class="avatar-icon">{char_icon}</div>
                                <div class="avatar-ring"></div>
                            </div>
                            
                            <div class="card-class" style="color: {char_color}">{char_class}</div>
                            <div class="card-archetype">{r['cognitive_archetype']}</div>
                            
                            <div class="stats-container">
                                <div class="stat-row">
                                    <span class="stat-icon">🧮</span>
                                    <span class="stat-name">ANL</span>
                                    <div class="stat-bar"><div class="stat-fill" style="width: {dims['analytical']*100}%; background: #3498db;"></div></div>
                                    <span class="stat-value">{int(dims['analytical']*100)}</span>
                                </div>
                                <div class="stat-row">
                                    <span class="stat-icon">🎨</span>
                                    <span class="stat-name">CRE</span>
                                    <div class="stat-bar"><div class="stat-fill" style="width: {dims['creative']*100}%; background: #9b59b6;"></div></div>
                                    <span class="stat-value">{int(dims['creative']*100)}</span>
                                </div>
                                <div class="stat-row">
                                    <span class="stat-icon">👥</span>
                                    <span class="stat-name">SOC</span>
                                    <div class="stat-bar"><div class="stat-fill" style="width: {dims['social']*100}%; background: #e74c3c;"></div></div>
                                    <span class="stat-value">{int(dims['social']*100)}</span>
                                </div>
                                <div class="stat-row">
                                    <span class="stat-icon">📊</span>
                                    <span class="stat-name">STR</span>
                                    <div class="stat-bar"><div class="stat-fill" style="width: {dims['structured']*100}%; background: #2ecc71;"></div></div>
                                    <span class="stat-value">{int(dims['structured']*100)}</span>
                                </div>
                                <div class="stat-row">
                                    <span class="stat-icon">💭</span>
                                    <span class="stat-name">EMO</span>
                                    <div class="stat-bar"><div class="stat-fill" style="width: {dims['emotional']*100}%; background: #f39c12;"></div></div>
                                    <span class="stat-value">{int(dims['emotional']*100)}</span>
                                </div>
                            </div>
                            
                            <div class="card-footer">
                                <div class="confidence-meter">
                                    <div class="confidence-label">CONFIDENCE</div>
                                    <div class="confidence-bar">
                                        <div class="confidence-fill" style="width: {r['confidence']*100}%"></div>
                                    </div>
                                    <div class="confidence-value">{int(r['confidence']*100)}%</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- RIGHT: HOLOGRAM DATA DISPLAY -->
                <div class="hologram-display">
                    <div class="holo-panel">
                        <div class="holo-title">
                            <span class="holo-icon">◈</span> NEURAL SCAN RESULTS
                        </div>
                        
                        <!-- 3D Radar Visualization -->
                        <div class="radar-3d">
                            <svg viewBox="0 0 200 200" class="radar-svg">
                                <!-- Background circles -->
                                <circle cx="100" cy="100" r="80" class="radar-circle"/>
                                <circle cx="100" cy="100" r="60" class="radar-circle"/>
                                <circle cx="100" cy="100" r="40" class="radar-circle"/>
                                <circle cx="100" cy="100" r="20" class="radar-circle"/>
                                
                                <!-- Axis lines -->
                                <line x1="100" y1="20" x2="100" y2="180" class="radar-axis"/>
                                <line x1="20" y1="100" x2="180" y2="100" class="radar-axis"/>
                                <line x1="35" y1="35" x2="165" y2="165" class="radar-axis"/>
                                <line x1="165" y1="35" x2="35" y2="165" class="radar-axis"/>
                                
                                <!-- Data polygon -->
                                <polygon class="radar-data" points="
                                    100,{100 - dims['analytical']*80}
                                    {100 + dims['creative']*80*0.95},{100 - dims['creative']*80*0.31}
                                    {100 + dims['social']*80*0.59},{100 + dims['social']*80*0.81}
                                    {100 - dims['structured']*80*0.59},{100 + dims['structured']*80*0.81}
                                    {100 - dims['emotional']*80*0.95},{100 - dims['emotional']*80*0.31}
                                "/>
                                
                                <!-- Data points -->
                                <circle cx="100" cy="{100 - dims['analytical']*80}" r="5" class="radar-point"/>
                                <circle cx="{100 + dims['creative']*80*0.95}" cy="{100 - dims['creative']*80*0.31}" r="5" class="radar-point"/>
                                <circle cx="{100 + dims['social']*80*0.59}" cy="{100 + dims['social']*80*0.81}" r="5" class="radar-point"/>
                                <circle cx="{100 - dims['structured']*80*0.59}" cy="{100 + dims['structured']*80*0.81}" r="5" class="radar-point"/>
                                <circle cx="{100 - dims['emotional']*80*0.95}" cy="{100 - dims['emotional']*80*0.31}" r="5" class="radar-point"/>
                            </svg>
                            <div class="radar-labels">
                                <span class="radar-label" style="top: 0; left: 50%; transform: translateX(-50%);">ANL</span>
                                <span class="radar-label" style="top: 20%; right: 5%;">CRE</span>
                                <span class="radar-label" style="bottom: 10%; right: 15%;">SOC</span>
                                <span class="radar-label" style="bottom: 10%; left: 15%;">STR</span>
                                <span class="radar-label" style="top: 20%; left: 5%;">EMO</span>
                            </div>
                        </div>
                        
                        <!-- Data Metrics -->
                        <div class="holo-metrics">
                            <div class="metric-box">
                                <div class="metric-icon">📝</div>
                                <div class="metric-value">{words}</div>
                                <div class="metric-label">WORDS</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-icon">🔤</div>
                                <div class="metric-value">{chars}</div>
                                <div class="metric-label">CHARS</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-icon">⚡</div>
                                <div class="metric-value">{r.get('processing_time_ms', 42):.0f}</div>
                                <div class="metric-label">MS</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-icon">🎯</div>
                                <div class="metric-value">{int(r['confidence']*100)}</div>
                                <div class="metric-label">CONF%</div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- AI Recommendation Panel -->
                    <div class="holo-panel recommendation-panel">
                        <div class="holo-title">
                            <span class="holo-icon">◈</span> AI RECOMMENDATION
                        </div>
                        <div class="recommendation-content">
                            <div class="rec-badge" style="background: {char_color}20; border-color: {char_color};">
                                {char_icon} {char_class}
                            </div>
                            <p class="rec-text">{get_recommendation(top_dim, dims)}</p>
                            <div class="rec-tags">
                                <span class="rec-tag">#{top_dim}</span>
                                <span class="rec-tag">#cognitive-profile</span>
                                <span class="rec-tag">#ai-analysis</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- BOTTOM: DIMENSION DETAILS -->
            <div class="dimension-details">
                <div class="holo-title">
                    <span class="holo-icon">◈</span> DIMENSION BREAKDOWN
                </div>
                <div class="dimension-grid">
                    {generate_dimension_cards(dims)}
                </div>
            </div>
            
        </div>
        """
        
        return html
        
    except Exception as e:
        return f"<div style='color: #ff4757; padding: 50px; text-align: center;'>❌ Error: {str(e)}<br><br>Make sure API is running on port 8000</div>"


def get_recommendation(top_dim, dims):
    recs = {
        'analytical': "Your exceptional analytical abilities make you ideal for roles in data science, research, strategic planning, and technical problem-solving. You excel at breaking down complex problems into logical components.",
        'creative': "Your creative mindset positions you perfectly for innovation-driven roles, design thinking, content creation, and entrepreneurial ventures. You see possibilities where others see obstacles.",
        'social': "Your outstanding social intelligence makes you a natural fit for leadership, team management, sales, networking, and any role requiring strong interpersonal skills.",
        'structured': "Your organizational excellence suits project management, operations, quality assurance, and any role requiring attention to detail and systematic approaches.",
        'emotional': "Your emotional intelligence is perfect for roles in mentoring, coaching, HR, customer success, and positions requiring empathy and human understanding."
    }
    return recs.get(top_dim, "You have a balanced cognitive profile suitable for diverse roles.")


def generate_dimension_cards(dims):
    cards = ""
    details = {
        'analytical': ('🧮', 'Analytical', 'Logical reasoning & systematic thinking', '#3498db'),
        'creative': ('🎨', 'Creative', 'Innovation & imaginative solutions', '#9b59b6'),
        'social': ('👥', 'Social', 'Interpersonal & collaboration skills', '#e74c3c'),
        'structured': ('📊', 'Structured', 'Organization & planning abilities', '#2ecc71'),
        'emotional': ('💭', 'Emotional', 'Empathy & emotional intelligence', '#f39c12')
    }
    
    for dim, score in sorted(dims.items(), key=lambda x: x[1], reverse=True):
        icon, name, desc, color = details[dim]
        pct = int(score * 100)
        cards += f"""
        <div class="dim-card" style="--card-color: {color};">
            <div class="dim-icon">{icon}</div>
            <div class="dim-name">{name}</div>
            <div class="dim-score">{pct}%</div>
            <div class="dim-bar-container">
                <div class="dim-bar" style="width: {pct}%; background: {color};"></div>
            </div>
            <div class="dim-desc">{desc}</div>
        </div>
        """
    return cards


# CSS Styles
css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

.mindscript-container {
    font-family: 'Rajdhani', sans-serif;
    background: linear-gradient(135deg, #0a0a1a 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
    padding: 30px;
    color: #fff;
}

/* HOLOGRAM HEADER */
.hologram-header {
    text-align: center;
    padding: 40px;
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(0, 255, 255, 0.2);
    border-radius: 20px;
    background: rgba(0, 255, 255, 0.02);
    margin-bottom: 30px;
}

.scan-line {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, transparent, #00ffff, transparent);
    animation: scan 2s linear infinite;
}

@keyframes scan {
    0% { top: 0; opacity: 1; }
    100% { top: 100%; opacity: 0; }
}

.glitch-text {
    font-family: 'Orbitron', sans-serif;
    font-size: 4em;
    font-weight: 900;
    color: #00ffff;
    text-shadow: 
        0 0 10px #00ffff,
        0 0 20px #00ffff,
        0 0 40px #00ffff,
        0 0 80px #00ffff;
    letter-spacing: 15px;
    animation: glitch 3s infinite;
    position: relative;
}

@keyframes glitch {
    0%, 90%, 100% { transform: translate(0); }
    92% { transform: translate(-5px, 0); }
    94% { transform: translate(5px, 0); }
    96% { transform: translate(-3px, 0); }
    98% { transform: translate(3px, 0); }
}

.hologram-subtitle {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.2em;
    color: rgba(0, 255, 255, 0.7);
    letter-spacing: 8px;
    margin-top: 15px;
}

.hologram-status {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    margin-top: 20px;
    padding: 10px 25px;
    background: rgba(0, 255, 0, 0.1);
    border: 1px solid rgba(0, 255, 0, 0.3);
    border-radius: 30px;
    color: #00ff00;
    font-family: 'Orbitron', sans-serif;
    font-size: 0.9em;
    letter-spacing: 3px;
}

.status-dot {
    width: 10px;
    height: 10px;
    background: #00ff00;
    border-radius: 50%;
    animation: pulse-dot 1s infinite;
}

@keyframes pulse-dot {
    0%, 100% { opacity: 1; box-shadow: 0 0 10px #00ff00; }
    50% { opacity: 0.5; box-shadow: 0 0 5px #00ff00; }
}

/* MAIN GRID */
.main-grid {
    display: grid;
    grid-template-columns: 400px 1fr;
    gap: 30px;
    margin-bottom: 30px;
}

@media (max-width: 1200px) {
    .main-grid { grid-template-columns: 1fr; }
}

/* 3D CHARACTER CARD */
.card-3d-container {
    perspective: 1000px;
}

.card-3d {
    width: 100%;
    height: 600px;
    position: relative;
    transform-style: preserve-3d;
    transition: transform 0.6s;
}

.card-3d-container:hover .card-3d {
    transform: rotateY(5deg) rotateX(5deg);
}

.card-front {
    width: 100%;
    height: 100%;
    background: linear-gradient(145deg, #1a1a2e 0%, #16213e 50%, #0f0f23 100%);
    border-radius: 25px;
    border: 2px solid rgba(0, 255, 255, 0.3);
    padding: 30px;
    display: flex;
    flex-direction: column;
    align-items: center;
    position: relative;
    overflow: hidden;
    box-shadow: 
        0 0 30px rgba(0, 255, 255, 0.1),
        inset 0 0 60px rgba(0, 255, 255, 0.05);
}

.card-front::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: linear-gradient(45deg, transparent 40%, rgba(0, 255, 255, 0.03) 50%, transparent 60%);
    animation: shine 3s infinite;
}

@keyframes shine {
    0% { transform: translateX(-100%) rotate(45deg); }
    100% { transform: translateX(100%) rotate(45deg); }
}

.card-header {
    width: 100%;
    display: flex;
    justify-content: space-between;
    margin-bottom: 20px;
}

.card-rank, .card-level {
    font-family: 'Orbitron', sans-serif;
    padding: 8px 20px;
    border-radius: 20px;
    font-size: 0.9em;
    letter-spacing: 2px;
}

.card-rank {
    background: linear-gradient(135deg, #f39c12, #e74c3c);
    color: white;
}

.card-level {
    background: rgba(0, 255, 255, 0.2);
    border: 1px solid rgba(0, 255, 255, 0.5);
    color: #00ffff;
}

.card-avatar {
    width: 150px;
    height: 150px;
    position: relative;
    margin: 20px 0;
}

.avatar-glow {
    position: absolute;
    inset: -20px;
    background: radial-gradient(circle, rgba(0, 255, 255, 0.3) 0%, transparent 70%);
    animation: glow-pulse 2s infinite;
}

@keyframes glow-pulse {
    0%, 100% { transform: scale(1); opacity: 0.5; }
    50% { transform: scale(1.1); opacity: 0.8; }
}

.avatar-icon {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 80px;
    z-index: 2;
}

.avatar-ring {
    position: absolute;
    inset: 0;
    border: 3px solid rgba(0, 255, 255, 0.5);
    border-radius: 50%;
    animation: rotate-ring 10s linear infinite;
}

.avatar-ring::before {
    content: '';
    position: absolute;
    top: -5px;
    left: 50%;
    width: 10px;
    height: 10px;
    background: #00ffff;
    border-radius: 50%;
    box-shadow: 0 0 15px #00ffff;
}

@keyframes rotate-ring {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.card-class {
    font-family: 'Orbitron', sans-serif;
    font-size: 2em;
    font-weight: 800;
    letter-spacing: 5px;
    text-shadow: 0 0 20px currentColor;
}

.card-archetype {
    color: rgba(255, 255, 255, 0.7);
    font-size: 1.1em;
    letter-spacing: 3px;
    margin-bottom: 25px;
}

.stats-container {
    width: 100%;
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.stat-row {
    display: grid;
    grid-template-columns: 30px 45px 1fr 40px;
    align-items: center;
    gap: 10px;
}

.stat-icon { font-size: 1.3em; }
.stat-name {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.85em;
    color: rgba(255, 255, 255, 0.7);
    letter-spacing: 2px;
}

.stat-bar {
    height: 8px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    overflow: hidden;
}

.stat-fill {
    height: 100%;
    border-radius: 4px;
    box-shadow: 0 0 10px currentColor;
    transition: width 1s ease;
}

.stat-value {
    font-family: 'Orbitron', sans-serif;
    font-weight: 700;
    color: #00ffff;
    text-align: right;
}

.card-footer {
    width: 100%;
    margin-top: auto;
    padding-top: 20px;
    border-top: 1px solid rgba(0, 255, 255, 0.2);
}

.confidence-meter {
    display: grid;
    grid-template-columns: auto 1fr auto;
    align-items: center;
    gap: 15px;
}

.confidence-label {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.8em;
    color: rgba(0, 255, 255, 0.7);
    letter-spacing: 2px;
}

.confidence-bar {
    height: 6px;
    background: rgba(0, 255, 255, 0.1);
    border-radius: 3px;
    overflow: hidden;
}

.confidence-fill {
    height: 100%;
    background: linear-gradient(90deg, #00ffff, #00ff00);
    border-radius: 3px;
    box-shadow: 0 0 15px rgba(0, 255, 255, 0.5);
}

.confidence-value {
    font-family: 'Orbitron', sans-serif;
    font-weight: 700;
    color: #00ff00;
}

/* HOLOGRAM DISPLAY */
.hologram-display {
    display: flex;
    flex-direction: column;
    gap: 20px;
}

.holo-panel {
    background: rgba(0, 255, 255, 0.02);
    border: 1px solid rgba(0, 255, 255, 0.2);
    border-radius: 20px;
    padding: 25px;
    position: relative;
    overflow: hidden;
}

.holo-panel::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, #00ffff, transparent);
}

.holo-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.1em;
    color: #00ffff;
    letter-spacing: 3px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.holo-icon {
    animation: pulse-icon 2s infinite;
}

@keyframes pulse-icon {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* 3D RADAR */
.radar-3d {
    position: relative;
    width: 100%;
    max-width: 300px;
    margin: 0 auto;
    padding: 30px;
}

.radar-svg {
    width: 100%;
    height: auto;
    filter: drop-shadow(0 0 10px rgba(0, 255, 255, 0.3));
}

.radar-circle {
    fill: none;
    stroke: rgba(0, 255, 255, 0.15);
    stroke-width: 1;
}

.radar-axis {
    stroke: rgba(0, 255, 255, 0.1);
    stroke-width: 1;
}

.radar-data {
    fill: rgba(0, 255, 255, 0.2);
    stroke: #00ffff;
    stroke-width: 2;
    filter: drop-shadow(0 0 8px #00ffff);
    animation: radar-pulse 3s infinite;
}

@keyframes radar-pulse {
    0%, 100% { opacity: 0.8; }
    50% { opacity: 1; }
}

.radar-point {
    fill: #00ffff;
    filter: drop-shadow(0 0 6px #00ffff);
}

.radar-labels {
    position: absolute;
    inset: 0;
}

.radar-label {
    position: absolute;
    font-family: 'Orbitron', sans-serif;
    font-size: 0.8em;
    color: rgba(0, 255, 255, 0.8);
    letter-spacing: 2px;
}

/* HOLO METRICS */
.holo-metrics {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 15px;
    margin-top: 25px;
}

.metric-box {
    text-align: center;
    padding: 20px 10px;
    background: rgba(0, 255, 255, 0.05);
    border: 1px solid rgba(0, 255, 255, 0.15);
    border-radius: 15px;
    transition: all 0.3s;
}

.metric-box:hover {
    background: rgba(0, 255, 255, 0.1);
    transform: translateY(-5px);
    box-shadow: 0 10px 30px rgba(0, 255, 255, 0.1);
}

.metric-icon { font-size: 1.8em; margin-bottom: 10px; }

.metric-value {
    font-family: 'Orbitron', sans-serif;
    font-size: 2em;
    font-weight: 700;
    color: #00ffff;
    text-shadow: 0 0 15px rgba(0, 255, 255, 0.5);
}

.metric-label {
    font-size: 0.85em;
    color: rgba(255, 255, 255, 0.5);
    letter-spacing: 2px;
    margin-top: 5px;
}

/* RECOMMENDATION PANEL */
.recommendation-panel {
    flex: 1;
}

.recommendation-content {
    display: flex;
    flex-direction: column;
    gap: 15px;
}

.rec-badge {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    padding: 10px 20px;
    border-radius: 25px;
    border: 1px solid;
    font-family: 'Orbitron', sans-serif;
    font-size: 0.9em;
    letter-spacing: 2px;
    width: fit-content;
}

.rec-text {
    color: rgba(255, 255, 255, 0.85);
    line-height: 1.7;
    font-size: 1.05em;
}

.rec-tags {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}

.rec-tag {
    padding: 5px 15px;
    background: rgba(0, 255, 255, 0.1);
    border-radius: 15px;
    font-size: 0.85em;
    color: rgba(0, 255, 255, 0.7);
}

/* DIMENSION DETAILS */
.dimension-details {
    background: rgba(0, 255, 255, 0.02);
    border: 1px solid rgba(0, 255, 255, 0.2);
    border-radius: 20px;
    padding: 30px;
}

.dimension-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
    margin-top: 20px;
}

.dim-card {
    background: linear-gradient(145deg, rgba(0, 0, 0, 0.3), rgba(0, 0, 0, 0.1));
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 15px;
    padding: 25px;
    text-align: center;
    transition: all 0.3s;
    position: relative;
    overflow: hidden;
}

.dim-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 4px;
    height: 100%;
    background: var(--card-color);
}

.dim-card:hover {
    transform: translateY(-10px);
    border-color: var(--card-color);
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
}

.dim-icon { font-size: 2.5em; margin-bottom: 15px; }

.dim-name {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.1em;
    letter-spacing: 2px;
    margin-bottom: 10px;
}

.dim-score {
    font-family: 'Orbitron', sans-serif;
    font-size: 2.5em;
    font-weight: 800;
    background: linear-gradient(135deg, #fff, var(--card-color));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 15px;
}

.dim-bar-container {
    height: 6px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 3px;
    overflow: hidden;
    margin-bottom: 15px;
}

.dim-bar {
    height: 100%;
    border-radius: 3px;
    box-shadow: 0 0 10px currentColor;
}

.dim-desc {
    font-size: 0.9em;
    color: rgba(255, 255, 255, 0.6);
    line-height: 1.5;
}
</style>
"""

# Sample texts
samples = [
    "I solve every problem with systematic analysis and logical thinking. Data drives my decisions. I break complex issues into smaller components and address each methodically. Spreadsheets and metrics guide my work. Precision matters to me.",
    "Creativity flows through everything I do. I see connections others miss. Innovation excites me more than tradition. I love pushing boundaries and challenging conventional thinking. Art and imagination fuel my soul.",
    "People energize me like nothing else. I thrive in teams and love building connections. Collaboration brings out my best work. I genuinely enjoy networking and social events. Relationships matter most to me.",
    "Organization is my superpower. I plan everything meticulously and never miss deadlines. My workspace is always clean and structured. Systems and processes bring me peace. Details matter as much as the big picture.",
    "I feel deeply and trust my intuition. Emotions provide insights that logic misses. Empathy guides my interactions with others. I understand people on a profound level. Authenticity matters more than perfection."
]

# Build interface
with gr.Blocks(title="MindScript Pro", theme=gr.themes.Base()) as app:
    
    gr.HTML(css)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## ✍️ Enter Text for Analysis")
            text_input = gr.Textbox(
                lines=8,
                placeholder="Write naturally about any topic (minimum 50 words)...",
                label="",
                show_label=False
            )
            
            gr.Markdown("**Quick Examples:**")
            example = gr.Radio(
                choices=["🧮 Analytical", "🎨 Creative", "👥 Social", "📊 Structured", "💭 Emotional"],
                label="",
                show_label=False
            )
            
            btn = gr.Button("🚀 ANALYZE", variant="primary", size="lg")
    
    output = gr.HTML(label="")
    
    # Events
    def load_sample(choice):
        idx = ["🧮 Analytical", "🎨 Creative", "👥 Social", "📊 Structured", "💭 Emotional"].index(choice)
        return samples[idx]
    
    example.change(load_sample, inputs=[example], outputs=[text_input])
    btn.click(analyze, inputs=[text_input], outputs=[output])

if __name__ == "__main__":
    app.launch(server_port=7860)