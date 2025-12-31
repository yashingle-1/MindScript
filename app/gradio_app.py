"""
MindScript Gradio Interface
"""

import gradio as gr
import requests
import json

def analyze_text(text):
    """Call API to analyze text"""
    try:
        response = requests.post(
            "http://localhost:8000/analyze",
            json={"text": text}
        )
        if response.status_code == 200:
            result = response.json()
            
            # Format output
            output = f"""
            📊 **Cognitive Profile Analysis**
            
            **Dimensions:**
            - Analytical: {result['dimensions']['analytical']:.2%}
            - Creative: {result['dimensions']['creative']:.2%}
            - Social: {result['dimensions']['social']:.2%}
            - Structured: {result['dimensions']['structured']:.2%}
            - Emotional: {result['dimensions']['emotional']:.2%}
            
            **Dominant Dimension:** {result['dominant_dimension']}
            **Cognitive Archetype:** {result['cognitive_archetype']}
            **Confidence:** {result['confidence']:.2%}
            """
            return output
        else:
            return "Error: Could not analyze text"
    except Exception as e:
        return f"Error: API not running. Start the API first.\nDetails: {str(e)}"

# Create Gradio interface (WITHOUT theme parameter)
iface = gr.Interface(
    fn=analyze_text,
    inputs=gr.Textbox(
        lines=10,
        placeholder="Enter text to analyze (at least 50 words)...",
        label="Input Text"
    ),
    outputs=gr.Markdown(label="Analysis Results"),
    title="🧠 MindScript - Cognitive Pattern Analysis",
    description="Analyze text to identify cognitive patterns across five dimensions",
    examples=[
        ["I believe that every problem can be solved through logical analysis. Breaking down complex issues into smaller components allows for systematic solutions."],
        ["The world is full of endless possibilities waiting to be explored. I love pushing boundaries and challenging conventional thinking."],
        ["Nothing energizes me more than spending time with friends and meeting new people. Social gatherings are my favorite way to recharge."],
    ]
    # REMOVED theme="soft" - this was causing the error
)

if __name__ == "__main__":
    iface.launch(share=False, server_name="0.0.0.0", server_port=7860)