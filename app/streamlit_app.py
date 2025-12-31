"""
MindScript Streamlit Demo
=========================
Interactive web interface for cognitive pattern analysis.

Author: [Your Name]
"""

import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict
import time

# Page config
st.set_page_config(
    page_title="MindScript - Cognitive Pattern Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .dimension-card {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# API URL
API_URL = "http://localhost:8000"


def analyze_text(text: str) -> Dict:
    """Call API to analyze text"""
    try:
        response = requests.post(
            f"{API_URL}/analyze",
            json={"text": text, "include_confidence": True},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.text}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API. Please start the API server."}
    except Exception as e:
        return {"error": str(e)}


def create_radar_chart(dimensions: Dict[str, float]) -> go.Figure:
    """Create radar chart for dimension scores"""
    categories = list(dimensions.keys())
    values = list(dimensions.values())
    
    # Close the radar chart
    categories.append(categories[0])
    values.append(values[0])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=[c.title() for c in categories],
        fill='toself',
        fillcolor='rgba(99, 110, 250, 0.3)',
        line=dict(color='rgb(99, 110, 250)', width=2),
        name='Cognitive Profile'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(size=10)
            ),
            angularaxis=dict(
                tickfont=dict(size=12)
            )
        ),
        showlegend=False,
        height=400,
        margin=dict(l=80, r=80, t=40, b=40)
    )
    
    return fig


def create_bar_chart(dimensions: Dict[str, float]) -> go.Figure:
    """Create horizontal bar chart"""
    df = pd.DataFrame({
        'Dimension': [d.title() for d in dimensions.keys()],
        'Score': list(dimensions.values())
    })
    df = df.sort_values('Score', ascending=True)
    
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['Score'],
        y=df['Dimension'],
        orientation='h',
        marker=dict(
            color=colors[:len(df)],
            line=dict(color='white', width=1)
        ),
        text=[f"{s:.1%}" for s in df['Score']],
        textposition='outside'
    ))
    
    fig.update_layout(
        xaxis=dict(range=[0, 1.1], title='Score'),
        yaxis=dict(title=''),
        height=300,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    return fig


def get_dimension_description(dimension: str, score: float) -> str:
    """Get description for dimension and score"""
    descriptions = {
        "analytical": {
            "high": "Strong logical reasoning and systematic thinking",
            "medium": "Balanced approach to analysis",
            "low": "Prefers intuitive over analytical approaches"
        },
        "creative": {
            "high": "Highly imaginative and innovative thinker",
            "medium": "Creative when needed, practical otherwise",
            "low": "Prefers conventional approaches"
        },
        "social": {
            "high": "Highly sociable and relationship-oriented",
            "medium": "Comfortable in social settings",
            "low": "Prefers solitary or small group settings"
        },
        "structured": {
            "high": "Very organized and detail-oriented",
            "medium": "Moderately organized",
            "low": "Flexible and adaptable"
        },
        "emotional": {
            "high": "Emotionally expressive and empathetic",
            "medium": "Balanced emotional expression",
            "low": "Reserved emotional expression"
        }
    }
    
    level = "high" if score > 0.66 else ("medium" if score > 0.33 else "low")
    return descriptions.get(dimension, {}).get(level, "")


# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 MindScript</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center; font-size: 1.2rem; color: gray;">'
        'Decode Cognitive Patterns from Text'
        '</p>',
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        st.markdown("### About")
        st.info(
            "MindScript analyzes text to identify cognitive patterns "
            "across five dimensions: Analytical, Creative, Social, "
            "Structured, and Emotional."
        )
        
        st.markdown("### Sample Texts")
        sample_texts = {
            "Analytical": "I believe that every problem has a logical solution. "
                         "By breaking down complex issues into smaller components, "
                         "we can systematically address each part.",
            "Creative": "The world is a canvas of infinite possibilities. "
                       "I love exploring new ideas and pushing the boundaries "
                       "of what's considered possible.",
            "Social": "There's nothing I enjoy more than connecting with others. "
                     "Building relationships and understanding different perspectives "
                     "is what makes life meaningful.",
            "Structured": "Organization is key to success. I always plan my day "
                         "in advance and follow a systematic approach to achieve "
                         "my goals efficiently.",
            "Emotional": "I feel deeply connected to the experiences around me. "
                        "Every moment carries significance, and I believe in "
                        "expressing emotions authentically."
        }
        
        selected_sample = st.selectbox(
            "Load sample text:",
            ["Custom"] + list(sample_texts.keys())
        )
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Enter Text")
        
        if selected_sample != "Custom":
            default_text = sample_texts[selected_sample]
        else:
            default_text = ""
        
        text_input = st.text_area(
            "Text to analyze:",
            value=default_text,
            height=200,
            placeholder="Enter at least 50 words for accurate analysis..."
        )
        
        word_count = len(text_input.split()) if text_input else 0
        st.caption(f"Word count: {word_count}")
        
        analyze_button = st.button(
            "🔍 Analyze",
            type="primary",
            use_container_width=True
        )
    
    with col2:
        st.subheader("📊 Results")
        
        if analyze_button and text_input:
            if word_count < 10:
                st.warning("Please enter at least 10 words for analysis.")
            else:
                with st.spinner("Analyzing cognitive patterns..."):
                    result = analyze_text(text_input)
                
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    # Display results
                    dimensions = result["dimensions"]
                    
                    # Metrics row
                    m1, m2, m3 = st.columns(3)
                    
                    with m1:
                        st.metric(
                            "Dominant",
                            result["dominant_dimension"].title()
                        )
                    
                    with m2:
                        st.metric(
                            "Archetype",
                            result["cognitive_archetype"]
                        )
                    
                    with m3:
                        st.metric(
                            "Confidence",
                            f"{result['confidence']:.1%}"
                        )
    
    # Results visualization (below both columns)
    if analyze_button and text_input and "error" not in (result if 'result' in dir() else {"error": True}):
        st.markdown("---")
        st.subheader("📈 Cognitive Profile")
        
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            st.markdown("#### Radar Chart")
            fig_radar = create_radar_chart(dimensions)
            st.plotly_chart(fig_radar, use_container_width=True)
        
        with viz_col2:
            st.markdown("#### Dimension Breakdown")
            fig_bar = create_bar_chart(dimensions)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Dimension details
        st.markdown("---")
        st.subheader("🔍 Dimension Analysis")
        
        for dim_name, score in dimensions.items():
            with st.expander(f"{dim_name.title()}: {score:.1%}"):
                description = get_dimension_description(dim_name, score)
                st.write(description)
                st.progress(score)
        
        # Processing info
        st.markdown("---")
        st.caption(
            f"Processed in {result['processing_time_ms']:.0f}ms | "
            f"Text length: {result['text_length']} characters | "
            f"Timestamp: {result['timestamp']}"
        )


if __name__ == "__main__":
    main()