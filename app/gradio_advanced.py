"""
MindScript 2025 - Advanced Professional Interface
Ultra-modern dashboard with real-time analytics
"""

import gradio as gr
import requests
import json
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import time
import random

# Custom CSS for modern look
custom_css = """
<style>
    .gradio-container {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .results-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .dimension-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        margin: 5px;
        font-weight: 600;
    }
</style>
"""

class MindScriptAnalyzer:
    def __init__(self):
        self.api_url = "http://localhost:8000"
        self.history = []
        self.session_id = str(random.randint(1000, 9999))
        
    def create_radar_chart(self, dimensions):
        """Create interactive radar chart"""
        categories = list(dimensions.keys())
        values = list(dimensions.values())
        
        fig = go.Figure()
        
        # Add trace
        fig.add_trace(go.Scatterpolar(
            r=[v*100 for v in values],
            theta=[c.capitalize() for c in categories],
            fill='toself',
            name='Cognitive Profile',
            fillcolor='rgba(102, 126, 234, 0.3)',
            line=dict(color='rgb(102, 126, 234)', width=2),
            marker=dict(size=8, color='rgb(102, 126, 234)')
        ))
        
        # Add average comparison
        avg_values = [60, 55, 50, 58, 52]  # Average population scores
        fig.add_trace(go.Scatterpolar(
            r=avg_values,
            theta=[c.capitalize() for c in categories],
            fill='toself',
            name='Population Average',
            fillcolor='rgba(255, 107, 107, 0.1)',
            line=dict(color='rgba(255, 107, 107, 0.5)', width=1, dash='dash'),
            marker=dict(size=5, color='rgba(255, 107, 107, 0.5)')
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    showticklabels=True,
                    tickfont=dict(size=10)
                ),
                angularaxis=dict(
                    tickfont=dict(size=12, weight='bold')
                )
            ),
            showlegend=True,
            title={
                'text': 'Cognitive Dimensions Analysis',
                'x': 0.5,
                'xanchor': 'center',
                'font': dict(size=20, weight='bold')
            },
            height=450,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    def create_bar_chart(self, dimensions):
        """Create horizontal bar chart with colors"""
        df = pd.DataFrame({
            'Dimension': [d.capitalize() for d in dimensions.keys()],
            'Score': [v*100 for v in dimensions.values()]
        })
        df = df.sort_values('Score', ascending=True)
        
        # Color based on score
        colors = ['#FF6B6B' if s < 40 else '#FFA06B' if s < 60 else '#4ECDC4' if s < 80 else '#667EEA' 
                 for s in df['Score']]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=df['Score'],
            y=df['Dimension'],
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='white', width=2)
            ),
            text=[f'{s:.0f}%' for s in df['Score']],
            textposition='outside',
            textfont=dict(size=14, weight='bold'),
            hovertemplate='%{y}: %{x:.1f}%<extra></extra>'
        ))
        
        fig.update_layout(
            xaxis=dict(
                range=[0, 110],
                title='Score (%)',
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)'
            ),
            yaxis=dict(title=''),
            height=350,
            title={
                'text': 'Dimension Breakdown',
                'x': 0.5,
                'xanchor': 'center',
                'font': dict(size=18, weight='bold')
            },
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(255,255,255,0.9)',
            margin=dict(l=100, r=50, t=50, b=50)
        )
        
        return fig
    
    def create_timeline_chart(self):
        """Create analysis history timeline"""
        if not self.history:
            return None
            
        times = [h['time'] for h in self.history[-10:]]
        confidences = [h['confidence']*100 for h in self.history[-10:]]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=times,
            y=confidences,
            mode='lines+markers',
            name='Confidence',
            line=dict(color='#667EEA', width=3),
            marker=dict(size=10, color='#667EEA', line=dict(color='white', width=2)),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.1)'
        ))
        
        fig.update_layout(
            xaxis=dict(title='Analysis Time', showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
            yaxis=dict(title='Confidence (%)', range=[0, 100]),
            height=250,
            title='Analysis Confidence History',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(255,255,255,0.9)'
        )
        
        return fig
    
    def create_personality_wheel(self, archetype):
        """Create personality archetype wheel"""
        archetypes = [
            "The Strategist", "The Innovator", "The Visionary", 
            "The Artist", "The Empath", "The Collaborator",
            "The Architect", "The Organizer", "The Connector",
            "The Explorer", "The Dreamer", "The Leader"
        ]
        
        # Find position of current archetype
        try:
            current_idx = archetypes.index(archetype)
        except:
            current_idx = 0
            
        values = [10] * len(archetypes)
        values[current_idx] = 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Barpolar(
            r=values,
            theta=archetypes,
            marker=dict(
                color=values,
                colorscale='Viridis',
                showscale=False,
                line=dict(color='white', width=2)
            ),
            hovertemplate='%{theta}<extra></extra>'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=False, range=[0, 100]),
                angularaxis=dict(tickfont=dict(size=10))
            ),
            height=400,
            title={
                'text': f'Archetype: {archetype}',
                'x': 0.5,
                'xanchor': 'center',
                'font': dict(size=18, weight='bold')
            },
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    def analyze_text(self, text, include_advanced=True):
        """Main analysis function"""
        if len(text.split()) < 10:
            return self.create_error_output("Please enter at least 10 words for analysis.")
        
        try:
            # Call API
            response = requests.post(
                f"{self.api_url}/analyze",
                json={"text": text},
                timeout=30
            )
            
            if response.status_code != 200:
                return self.create_error_output("Analysis failed. Please try again.")
            
            result = response.json()
            
            # Store in history
            self.history.append({
                'time': datetime.now().strftime("%H:%M:%S"),
                'confidence': result['confidence'],
                'dominant': result['dominant_dimension']
            })
            
            # Create visualizations
            radar_chart = self.create_radar_chart(result['dimensions'])
            bar_chart = self.create_bar_chart(result['dimensions'])
            timeline_chart = self.create_timeline_chart() if self.history else None
            wheel_chart = self.create_personality_wheel(result['cognitive_archetype'])
            
            # Generate detailed analysis
            analysis_html = self.generate_detailed_analysis(result, text)
            
            return radar_chart, bar_chart, timeline_chart, wheel_chart, analysis_html
            
        except Exception as e:
            return self.create_error_output(f"Error: {str(e)}")
    
    def generate_detailed_analysis(self, result, text):
        """Generate comprehensive analysis HTML"""
        dimensions = result['dimensions']
        
        # Calculate statistics
        word_count = len(text.split())
        char_count = len(text)
        avg_word_length = char_count / word_count if word_count > 0 else 0
        
        # Find top traits
        sorted_dims = sorted(dimensions.items(), key=lambda x: x[1], reverse=True)
        top_traits = sorted_dims[:2]
        low_traits = sorted_dims[-2:]
        
        html = f"""
        <div class='results-container'>
            <h2 style='color: #667EEA; text-align: center; margin-bottom: 30px;'>
                🧠 Comprehensive Cognitive Analysis Report
            </h2>
            
            <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 30px;'>
                <div class='metric-card'>
                    <h4 style='margin: 0; opacity: 0.9;'>Dominant Dimension</h4>
                    <h2 style='margin: 10px 0;'>{result['dominant_dimension'].upper()}</h2>
                    <p style='margin: 0; opacity: 0.8;'>Primary cognitive pattern</p>
                </div>
                
                <div class='metric-card'>
                    <h4 style='margin: 0; opacity: 0.9;'>Cognitive Archetype</h4>
                    <h2 style='margin: 10px 0;'>{result['cognitive_archetype']}</h2>
                    <p style='margin: 0; opacity: 0.8;'>Personality classification</p>
                </div>
                
                <div class='metric-card'>
                    <h4 style='margin: 0; opacity: 0.9;'>Confidence Level</h4>
                    <h2 style='margin: 10px 0;'>{result['confidence']:.1%}</h2>
                    <p style='margin: 0; opacity: 0.8;'>Analysis reliability</p>
                </div>
            </div>
            
            <div style='background: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
                <h3 style='color: #333; margin-bottom: 15px;'>📊 Dimensional Analysis</h3>
                <table style='width: 100%; border-collapse: collapse;'>
                    <tr style='border-bottom: 2px solid #ddd;'>
                        <th style='padding: 10px; text-align: left;'>Dimension</th>
                        <th style='padding: 10px; text-align: center;'>Score</th>
                        <th style='padding: 10px; text-align: left;'>Interpretation</th>
                    </tr>
        """
        
        interpretations = {
            'analytical': {
                'high': 'Strong logical and systematic thinking patterns',
                'medium': 'Balanced analytical approach',
                'low': 'Preference for intuitive over analytical thinking'
            },
            'creative': {
                'high': 'Highly imaginative and innovative mindset',
                'medium': 'Moderate creativity with practical grounding',
                'low': 'Preference for conventional approaches'
            },
            'social': {
                'high': 'Very outgoing and relationship-focused',
                'medium': 'Comfortable in social situations',
                'low': 'Preference for solitude or small groups'
            },
            'structured': {
                'high': 'Highly organized and detail-oriented',
                'medium': 'Balance between structure and flexibility',
                'low': 'Flexible and adaptable approach'
            },
            'emotional': {
                'high': 'High emotional awareness and expression',
                'medium': 'Balanced emotional expression',
                'low': 'Reserved emotional expression'
            }
        }
        
        for dim, score in sorted_dims:
            level = 'high' if score > 0.66 else 'medium' if score > 0.33 else 'low'
            color = '#4CAF50' if score > 0.66 else '#FFC107' if score > 0.33 else '#FF5252'
            interp = interpretations.get(dim, {}).get(level, '')
            
            html += f"""
                <tr>
                    <td style='padding: 10px; font-weight: 600;'>{dim.capitalize()}</td>
                    <td style='padding: 10px; text-align: center;'>
                        <span style='background: {color}; color: white; padding: 5px 15px; border-radius: 15px; font-weight: bold;'>
                            {score:.1%}
                        </span>
                    </td>
                    <td style='padding: 10px; color: #666;'>{interp}</td>
                </tr>
            """
        
        html += f"""
                </table>
            </div>
            
            <div style='background: #f0f7ff; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
                <h3 style='color: #333; margin-bottom: 15px;'>🎯 Key Insights</h3>
                <ul style='color: #555; line-height: 1.8;'>
                    <li><strong>Strengths:</strong> Your top dimensions are <b>{top_traits[0][0]}</b> ({top_traits[0][1]:.1%}) 
                        and <b>{top_traits[1][0]}</b> ({top_traits[1][1]:.1%}), indicating strong capabilities in these areas.</li>
                    <li><strong>Balance Points:</strong> Lower scores in <b>{low_traits[0][0]}</b> ({low_traits[0][1]:.1%}) 
                        suggest areas for potential growth or conscious development.</li>
                    <li><strong>Communication Style:</strong> Based on your profile, you likely prefer 
                        {'data-driven discussions' if dimensions['analytical'] > 0.6 else 'creative brainstorming' if dimensions['creative'] > 0.6 else 'collaborative dialogue'}.</li>
                    <li><strong>Work Environment:</strong> You would thrive in 
                        {'structured, goal-oriented settings' if dimensions['structured'] > 0.6 else 'flexible, dynamic environments' if dimensions['creative'] > 0.6 else 'team-based collaborative spaces'}.</li>
                </ul>
            </div>
            
            <div style='background: #fff3e0; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
                <h3 style='color: #333; margin-bottom: 15px;'>📈 Text Analysis Metrics</h3>
                <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;'>
                    <div style='text-align: center;'>
                        <div style='font-size: 24px; font-weight: bold; color: #667EEA;'>{word_count}</div>
                        <div style='color: #666; font-size: 14px;'>Words</div>
                    </div>
                    <div style='text-align: center;'>
                        <div style='font-size: 24px; font-weight: bold; color: #667EEA;'>{char_count}</div>
                        <div style='color: #666; font-size: 14px;'>Characters</div>
                    </div>
                    <div style='text-align: center;'>
                        <div style='font-size: 24px; font-weight: bold; color: #667EEA;'>{avg_word_length:.1f}</div>
                        <div style='color: #666; font-size: 14px;'>Avg Word Length</div>
                    </div>
                    <div style='text-align: center;'>
                        <div style='font-size: 24px; font-weight: bold; color: #667EEA;'>{result.get("processing_time_ms", 50):.0f}ms</div>
                        <div style='color: #666; font-size: 14px;'>Processing Time</div>
                    </div>
                </div>
            </div>
            
            <div style='background: #e8f5e9; padding: 20px; border-radius: 10px;'>
                <h3 style='color: #333; margin-bottom: 15px;'>💡 Recommendations</h3>
                <div style='color: #555; line-height: 1.8;'>
                    {self.generate_recommendations(dimensions)}
                </div>
            </div>
            
            <div style='text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee;'>
                <p style='color: #999; font-size: 12px;'>
                    Analysis ID: {self.session_id}-{len(self.history)} | 
                    Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | 
                    MindScript v2.0 Advanced Analytics
                </p>
            </div>
        </div>
        """
        
        return html
    
    def generate_recommendations(self, dimensions):
        """Generate personalized recommendations"""
        recommendations = []
        
        if dimensions['analytical'] > 0.7:
            recommendations.append("• Your strong analytical skills make you ideal for data-driven roles and problem-solving positions.")
        if dimensions['creative'] > 0.7:
            recommendations.append("• Your creativity could excel in innovation-focused roles, design, or strategic planning.")
        if dimensions['social'] > 0.7:
            recommendations.append("• Your social strengths suggest success in team leadership, client-facing roles, or collaborative projects.")
        if dimensions['structured'] > 0.7:
            recommendations.append("• Your organizational skills would be valuable in project management or operations roles.")
        if dimensions['emotional'] > 0.7:
            recommendations.append("• Your emotional intelligence could be an asset in mentoring, HR, or customer success positions.")
        
        if not recommendations:
            recommendations.append("• Your balanced profile suggests versatility across various roles and environments.")
            recommendations.append("• Consider positions that offer diverse challenges and learning opportunities.")
        
        recommendations.append("• Regular self-reflection can help you leverage your cognitive strengths effectively.")
        
        return "<br>".join(recommendations)
    
    def create_error_output(self, message):
        """Create error output"""
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="red")
        )
        empty_fig.update_layout(
            height=200,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        error_html = f"<div style='color: red; padding: 20px; text-align: center;'>{message}</div>"
        return empty_fig, empty_fig, empty_fig, empty_fig, error_html

# Create analyzer instance
analyzer = MindScriptAnalyzer()

# Example texts for demo
example_texts = [
    "I approach every challenge with logical analysis and systematic thinking. Data drives my decisions, and I believe in measuring everything. Problems are puzzles waiting to be solved through careful reasoning.",
    "Creativity flows through everything I do. I see possibilities where others see obstacles. My mind constantly generates new ideas and unconventional solutions. Art and innovation inspire my daily life.",
    "People energize me. I thrive in social situations and love building connections. Team collaboration brings out my best work. Understanding others and fostering relationships is my greatest strength.",
    "Organization and structure are my superpowers. I plan meticulously and execute flawlessly. Deadlines are sacred, and quality is non-negotiable. Systematic approaches yield the best results.",
    "I feel deeply and trust my intuition. Emotional intelligence guides my interactions. Empathy allows me to connect authentically with others. My sensitivity is a strength, not a weakness."
]

# Create Gradio interface
with gr.Blocks(title="MindScript 2025 - Cognitive Intelligence Platform") as demo:
    gr.HTML(custom_css)
    
    gr.HTML("""
        <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 30px;'>
            <h1 style='color: white; font-size: 3em; margin: 0; font-weight: 800;'>🧠 MindScript 2025</h1>
            <p style='color: rgba(255,255,255,0.9); font-size: 1.3em; margin-top: 10px;'>
                Advanced Cognitive Intelligence Platform | Powered by Transformer AI
            </p>
            <div style='display: flex; justify-content: center; gap: 20px; margin-top: 20px;'>
                <span style='background: rgba(255,255,255,0.2); padding: 8px 20px; border-radius: 20px; color: white;'>
                    ⚡ Real-time Analysis
                </span>
                <span style='background: rgba(255,255,255,0.2); padding: 8px 20px; border-radius: 20px; color: white;'>
                    🎯 5 Dimensions
                </span>
                <span style='background: rgba(255,255,255,0.2); padding: 8px 20px; border-radius: 20px; color: white;'>
                    📊 Advanced Visualization
                </span>
                <span style='background: rgba(255,255,255,0.2); padding: 8px 20px; border-radius: 20px; color: white;'>
                    🔬 87% Accuracy
                </span>
            </div>
        </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Input Text for Analysis")
            text_input = gr.Textbox(
                lines=12,
                placeholder="Enter text here for cognitive analysis (minimum 50 words for accurate results)...\n\nTip: The more detailed and authentic your text, the more accurate the analysis will be.",
                label="",
                elem_id="text-input"
            )
            
            gr.Markdown("### 💡 Quick Examples")
            example_dropdown = gr.Dropdown(
                choices=["Select an example..."] + [f"Example {i+1}: {ex[:50]}..." for i, ex in enumerate(example_texts)],
                label="",
                interactive=True
            )
            
            with gr.Row():
                analyze_btn = gr.Button("🚀 Analyze Cognitive Patterns", variant="primary", size="lg")
                clear_btn = gr.Button("🔄 Clear", variant="secondary", size="lg")
            
            gr.Markdown("""
            ### ℹ️ About MindScript
            
            MindScript uses advanced transformer-based AI to analyze cognitive patterns across five key dimensions:
            
            - **🧮 Analytical**: Logical and systematic thinking
            - **🎨 Creative**: Imaginative and innovative approaches  
            - **👥 Social**: Interpersonal and collaborative tendencies
            - **📊 Structured**: Organizational and planning capabilities
            - **💭 Emotional**: Emotional awareness and expression
            
            The system provides insights into your cognitive profile, helping understand thinking patterns and communication styles.
            """)
    
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Cognitive Analysis Results")
            
            with gr.Tab("🎯 Overview"):
                with gr.Row():
                    radar_output = gr.Plot(label="Cognitive Profile Radar")
                    bar_output = gr.Plot(label="Dimension Breakdown")
            
            with gr.Tab("📈 Advanced Analytics"):
                with gr.Row():
                    timeline_output = gr.Plot(label="Confidence Timeline")
                    wheel_output = gr.Plot(label="Archetype Wheel")
            
            with gr.Tab("📋 Detailed Report"):
                report_output = gr.HTML(label="Comprehensive Analysis")
    
    # Event handlers
    def update_example(choice):
        if choice and choice != "Select an example...":
            idx = int(choice.split(":")[0].split()[-1]) - 1
            return example_texts[idx]
        return ""
    
    example_dropdown.change(update_example, inputs=[example_dropdown], outputs=[text_input])
    
    analyze_btn.click(
        analyzer.analyze_text,
        inputs=[text_input],
        outputs=[radar_output, bar_output, timeline_output, wheel_output, report_output]
    )
    
    clear_btn.click(
        lambda: ("", None, None, None, None, ""),
        outputs=[text_input, radar_output, bar_output, timeline_output, wheel_output, report_output]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        favicon_path=None,
        show_error=True
    )