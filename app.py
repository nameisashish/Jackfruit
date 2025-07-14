import streamlit as st
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
import io
import time
from ultralytics import YOLO
import plotly.graph_objects as go
import plotly.express as px
import json
import os
from huggingface_hub import hf_hub_download

# Page configuration
st.set_page_config(
    page_title="🍈 Jackfruit AI Detector",
    page_icon="🍈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []

# Theme definitions (always dark mode)
themes = {
    'dark': {
        'bg_primary': '#1a1a1a',
        'bg_secondary': '#2d2d2d',
        'text_primary': '#ffffff',
        'text_secondary': '#b0b0b0',
        'accent_1': '#ff6b6b',
        'accent_2': '#4ecdc4',
        'accent_3': '#45b7d1',
        'gradient_1': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'gradient_2': 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
        'gradient_3': 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
        'card_shadow': '0 8px 32px rgba(0,0,0,0.3)',
        'border_color': '#404040'
    }
}

current_theme = themes['dark']

# Enhanced CSS with theme support and better animations
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@300;400;500;600;700&display=swap');
    
    :root {{
        --bg-primary: {current_theme['bg_primary']};
        --bg-secondary: {current_theme['bg_secondary']};
        --text-primary: {current_theme['text_primary']};
        --text-secondary: {current_theme['text_secondary']};
        --accent-1: {current_theme['accent_1']};
        --accent-2: {current_theme['accent_2']};
        --accent-3: {current_theme['accent_3']};
        --gradient-1: {current_theme['gradient_1']};
        --gradient-2: {current_theme['gradient_2']};
        --gradient-3: {current_theme['gradient_3']};
        --card-shadow: {current_theme['card_shadow']};
        --border-color: {current_theme['border_color']};
    }}
    
    .stApp {{
        background: var(--bg-primary);
        color: var(--text-primary);
        transition: all 0.3s ease;
    }}
    
    * {{
        font-family: 'Inter', 'Poppins', sans-serif;
        transition: all 0.3s ease;
    }}
    
    .main-header {{
        background: var(--gradient-1);
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: var(--card-shadow);
        animation: slideInDown 1s ease-out;
        position: relative;
        overflow: hidden;
    }}
    
    .main-header::before {{
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: float 6s ease-in-out infinite;
    }}
    
    .main-header h1 {{
        color: white;
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
    }}
    
    .main-header p {{
        color: rgba(255,255,255,0.9);
        font-size: 1.3rem;
        margin: 0;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }}
    
    @keyframes slideInDown {{
        from {{
            transform: translateY(-100px);
            opacity: 0;
        }}
        to {{
            transform: translateY(0);
            opacity: 1;
        }}
    }}
    
    @keyframes float {{
        0%, 100% {{ transform: translateY(0px) rotate(0deg); }}
        50% {{ transform: translateY(-20px) rotate(180deg); }}
    }}
    
    .upload-section {{
        background: var(--gradient-2);
        padding: 2.5rem;
        border-radius: 20px;
        margin: 2rem 0;
        box-shadow: var(--card-shadow);
        animation: fadeInUp 1s ease-out;
        position: relative;
        overflow: hidden;
    }}
    
    .upload-section::before {{
        content: '🍈';
        position: absolute;
        top: -20px;
        right: -20px;
        font-size: 8rem;
        opacity: 0.1;
        animation: rotate 20s linear infinite;
    }}
    
    @keyframes rotate {{
        from {{ transform: rotate(0deg); }}
        to {{ transform: rotate(360deg); }}
    }}
    
    @keyframes fadeInUp {{
        from {{
            transform: translateY(50px);
            opacity: 0;
        }}
        to {{
            transform: translateY(0);
            opacity: 1;
        }}
    }}
    
    .stats-card {{
        background: var(--bg-secondary);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: var(--card-shadow);
        border: 1px solid var(--border-color);
        margin: 1rem 0;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
    }}
    
    .stats-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        transition: left 0.5s;
    }}
    
    .stats-card:hover {{
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
    }}
    
    .stats-card:hover::before {{
        left: 100%;
    }}
    
    .result-container {{
        background: var(--gradient-3);
        padding: 2.5rem;
        border-radius: 20px;
        margin: 2rem 0;
        box-shadow: var(--card-shadow);
        animation: zoomIn 1s ease-out;
        position: relative;
    }}
    
    @keyframes zoomIn {{
        from {{
            transform: scale(0.8);
            opacity: 0;
        }}
        to {{
            transform: scale(1);
            opacity: 1;
        }}
    }}
    
    .detection-badge {{
        background: var(--gradient-1);
        color: white;
        padding: 1rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.2rem;
        display: inline-block;
        margin: 1rem 0;
        box-shadow: var(--card-shadow);
        animation: pulse 2s infinite;
        position: relative;
        overflow: hidden;
    }}
    
    .detection-badge::before {{
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        background: rgba(255,255,255,0.2);
        border-radius: 50%;
        transition: all 0.6s ease;
        transform: translate(-50%, -50%);
    }}
    
    .detection-badge:hover::before {{
        width: 300px;
        height: 300px;
    }}
    
    @keyframes pulse {{
        0% {{ transform: scale(1); box-shadow: 0 0 0 0 rgba(255, 107, 107, 0.7); }}
        70% {{ transform: scale(1.05); box-shadow: 0 0 0 10px rgba(255, 107, 107, 0); }}
        100% {{ transform: scale(1); box-shadow: 0 0 0 0 rgba(255, 107, 107, 0); }}
    }}
    
    .sidebar-content {{
        background: var(--gradient-1);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        animation: slideInLeft 0.8s ease-out;
    }}
    
    @keyframes slideInLeft {{
        from {{
            transform: translateX(-100px);
            opacity: 0;
        }}
        to {{
            transform: translateX(0);
            opacity: 1;
        }}
    }}
    
    .sidebar-content h3 {{
        color: white;
        margin-bottom: 1rem;
        font-weight: 600;
    }}
    
    .stButton > button {{
        background: var(--gradient-1) !important;
        color: white !important;
        border: none !important;
        padding: 1rem 2.5rem !important;
        border-radius: 50px !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        box-shadow: var(--card-shadow) !important;
        position: relative !important;
        overflow: hidden !important;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-3px) scale(1.05) !important;
        box-shadow: 0 15px 35px rgba(0,0,0,0.2) !important;
    }}
    
    .stButton > button:active {{
        transform: translateY(0) scale(0.95) !important;
    }}
    
    .metric-card {{
        background: var(--bg-secondary);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: var(--card-shadow);
        border: 1px solid var(--border-color);
        margin: 0.5rem;
        animation: fadeIn 1s ease-out;
        transition: all 0.3s ease;
    }}
    
    .metric-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.15);
    }}
    
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    .loading-container {{
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        padding: 3rem;
        background: var(--bg-secondary);
        border-radius: 20px;
        margin: 2rem 0;
    }}
    
    .jackfruit-loader {{
        width: 80px;
        height: 80px;
        border-radius: 50%;
        background: var(--gradient-1);
        animation: jackfruitSpin 2s linear infinite;
        position: relative;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
        margin-bottom: 1rem;
    }}
    
    @keyframes jackfruitSpin {{
        0% {{ transform: rotate(0deg) scale(1); }}
        50% {{ transform: rotate(180deg) scale(1.1); }}
        100% {{ transform: rotate(360deg) scale(1); }}
    }}
    
    .feature-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 2rem;
        margin: 2rem 0;
    }}
    
    .feature-card {{
        background: var(--bg-secondary);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: var(--card-shadow);
        border: 1px solid var(--border-color);
        text-align: center;
        animation: slideInUp 0.8s ease-out;
        transition: all 0.3s ease;
    }}
    
    .feature-card:hover {{
        transform: translateY(-10px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
    }}
    
    @keyframes slideInUp {{
        from {{
            transform: translateY(100px);
            opacity: 0;
        }}
        to {{
            transform: translateY(0);
            opacity: 1;
        }}
    }}
    
    .feature-icon {{
        font-size: 3rem;
        margin-bottom: 1rem;
        display: block;
        animation: bounce 2s infinite;
    }}
    
    @keyframes bounce {{
        0%, 20%, 50%, 80%, 100% {{ transform: translateY(0); }}
        40% {{ transform: translateY(-10px); }}
        60% {{ transform: translateY(-5px); }}
    }}
    
    .footer {{
        background: var(--gradient-1);
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-top: 3rem;
        text-align: center;
        animation: fadeIn 1s ease-out;
        position: relative;
        overflow: hidden;
    }}
    
    .footer::before {{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        animation: shine 3s infinite;
    }}
    
    @keyframes shine {{
        0% {{ left: -100%; }}
        100% {{ left: 100%; }}
    }}
    
    .stSelectbox > div > div {{
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important;
    }}
    
    .stSlider > div > div {{
        background: var(--gradient-1) !important;
    }}
    
    .stProgress > div > div {{
        background: var(--gradient-1);
    }}
    
    .annotated-image {{
        animation: fadeInScale 1s ease-out;
    }}
    
    @keyframes fadeInScale {{
        from {{
            opacity: 0;
            transform: scale(0.95);
        }}
        to {{
            opacity: 1;
            transform: scale(1);
        }}
    }}
    
    /* Responsive design */
    @media (max-width: 768px) {{
        .main-header h1 {{
            font-size: 2.5rem;
        }}
        
        .main-header p {{
            font-size: 1.1rem;
        }}
        
        .feature-grid {{
            grid-template-columns: 1fr;
        }}
    }}
</style>
""", unsafe_allow_html=True)

# Header with enhanced styling
st.markdown("""
<div class="main-header">
    <h1>🍈 Jackfruit AI Detector</h1>
    <p>Advanced Computer Vision for Intelligent Fruit Detection & Analysis</p>
</div>
""", unsafe_allow_html=True)

# Features section
st.markdown("""
<div class="feature-grid">
    <div class="feature-card">
        <span class="feature-icon">🎯</span>
        <h3>High Precision Detection</h3>
        <p>Advanced YOLOv8 model with 95%+ accuracy for jackfruit identification</p>
    </div>
    <div class="feature-card">
        <span class="feature-icon">⚡</span>
        <h3>Real-time Processing</h3>
        <p>Lightning-fast inference with optimized neural network architecture</p>
    </div>
    <div class="feature-card">
        <span class="feature-icon">📊</span>
        <h3>Smart Analytics</h3>
        <p>Comprehensive statistics and trend analysis for your detections</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Enhanced Sidebar
with st.sidebar:
    st.markdown("""
    <div class="sidebar-content">
        <h3>🔧 Detection Settings</h3>
    </div>
    """, unsafe_allow_html=True)
    
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Minimum confidence score for detection"
    )
    
    st.markdown("---")
    
    st.markdown("""
    <div class="sidebar-content">
        <h3>📊 Analytics Dashboard</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.detection_history:
        total_images = len(st.session_state.detection_history)
        total_jackfruits = sum(st.session_state.detection_history)
        avg_per_image = total_jackfruits / total_images if total_images > 0 else 0
        max_detected = max(st.session_state.detection_history) if st.session_state.detection_history else 0
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📸 Images", total_images, delta=1 if total_images > 0 else 0)
            st.metric("🍈 Total Fruits", total_jackfruits)
        with col2:
            st.metric("📈 Average", f"{avg_per_image:.1f}")
            st.metric("🏆 Max Found", max_detected)
    else:
        st.info("Upload images to see analytics")
    
    st.markdown("---")
    
    st.markdown("""
    <div class="sidebar-content">
        <h3>🎛️ Controls</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🗑️ Clear History"):
        st.session_state.detection_history = []
        st.success("History cleared successfully!")
        st.rerun()

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    <div class="upload-section">
        <h2 style="color: white; text-align: center; margin-bottom: 1rem; font-weight: 600;">📸 Upload Your Image</h2>
        <p style="color: rgba(255,255,255,0.9); text-align: center; font-size: 1.1rem;">Drag & drop or click to upload jackfruit images</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Upload images in JPG, PNG, BMP, or TIFF format"
    )
    
    if uploaded_file is not None:
        # Display uploaded image with enhanced styling
        image = Image.open(uploaded_file)
        st.image(image, caption="📁 Uploaded Image", use_column_width=True)
        
        # Image info
        st.markdown(f"""
        <div class="stats-card">
            <h4>📋 Image Information</h4>
            <p><strong>Format:</strong> {image.format}</p>
            <p><strong>Size:</strong> {image.size[0]} x {image.size[1]} pixels</p>
            <p><strong>Mode:</strong> {image.mode}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Convert PIL image to OpenCV format
        image_array = np.array(image)
        if len(image_array.shape) == 3:
            image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        else:
            image_cv = image_array

with col2:
    if uploaded_file is not None:
        st.markdown("""
        <div class="result-container">
            <h2 style="color: white; text-align: center; margin-bottom: 1rem; font-weight: 600;">🔍 Detection Results</h2>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Analyze Image", key="analyze_btn"):
            # Enhanced loading animation
            st.markdown("""
            <div class="loading-container">
                <div class="jackfruit-loader">🍈</div>
                <h3 style="color: var(--text-primary); margin-bottom: 0.5rem;">Analyzing Image...</h3>
                <p style="color: var(--text-secondary);">Please wait while our AI processes your image</p>
            </div>
            """, unsafe_allow_html=True)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate processing stages
            stages = [
                "🔍 Loading AI model...",
                "🎯 Detecting objects...",
                "📊 Analyzing results...",
                "✅ Finalizing detection..."
            ]
            
            for i, stage in enumerate(stages):
                status_text.text(stage)
                for j in range(25):
                    progress_bar.progress((i * 25) + j + 1)
                    time.sleep(0.01)
            
            try:
                # Download the model from Hugging Face
                model_path = hf_hub_download(repo_id="theashish03/jackfruit", filename="best.pt")
                model = YOLO(model_path)
                
                # Use confidence threshold directly
                adjusted_conf = confidence_threshold
                
                # Save uploaded image temporarily
                temp_image_path = "temp_image.jpg"
                cv2.imwrite(temp_image_path, image_cv)
                
                # Run inference
                results = model(temp_image_path, conf=adjusted_conf, save=False)
                result = results[0]
                boxes = result.boxes
                
                # Clear loading elements
                progress_bar.empty()
                status_text.empty()
                
                # Count jackfruits and collect detection data
                jackfruit_count = 0
                confidences = []
                
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        conf_score = float(box.conf[0])
                        if class_id == 0 and conf_score >= adjusted_conf:
                            jackfruit_count += 1
                            confidences.append(conf_score)
                
                # Add to history
                st.session_state.detection_history.append(jackfruit_count)
                
                # Calculate summary
                if confidences:
                    avg_conf = np.mean(confidences)
                    min_conf = np.min(confidences)
                    max_conf = np.max(confidences)
                    avg_conf_pct = avg_conf * 100
                    min_conf_pct = min_conf * 100
                    max_conf_pct = max_conf * 100
                else:
                    avg_conf = min_conf = max_conf = 0
                    avg_conf_pct = min_conf_pct = max_conf_pct = 0
                
                # Display results with enhanced styling
                st.markdown(f"""
                <div class="detection-badge">
                    🍈 {jackfruit_count} Jackfruit{'s' if jackfruit_count != 1 else ''} Detected
                </div>
                """, unsafe_allow_html=True)
                
                # Show annotated image with animation
                annotated_img = result.plot()
                annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                st.image(annotated_img_rgb, caption=f"✅ Analysis Complete: {jackfruit_count} jackfruits detected", use_column_width=True)
                
                # Summarized detection details
                if jackfruit_count > 0:
                    st.markdown("""
                    <div class="stats-card">
                        <h4>📋 Detection Summary</h4>
                        <p><strong>Average Confidence:</strong> {avg_conf:.3f} ({avg_conf_pct:.1f}%)</p>
                        <p><strong>Min Confidence:</strong> {min_conf:.3f} ({min_conf_pct:.1f}%)</p>
                        <p><strong>Max Confidence:</strong> {max_conf:.3f} ({max_conf_pct:.1f}%)</p>
                    </div>
                    """.format(avg_conf=avg_conf, avg_conf_pct=avg_conf_pct, min_conf=min_conf, min_conf_pct=min_conf_pct, max_conf=max_conf, max_conf_pct=max_conf_pct), unsafe_allow_html=True)
                
                # Success message
                st.success(f"🎉 Analysis completed successfully! Found {jackfruit_count} jackfruit{'s' if jackfruit_count != 1 else ''} in your image.")
                
                # Cleanup
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
                        
            except Exception as e:
                st.error(f"❌ Error during detection: {str(e)}")
                st.info("💡 Please ensure the model file path is correct and the model is accessible.")

# Enhanced History visualization
if st.session_state.detection_history:
    st.markdown("---")
    st.markdown("""
    <div class="result-container">
        <h2 style="color: white; text-align: center; margin-bottom: 2rem; font-weight: 600;">📈 Detection Analytics</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Enhanced line chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, len(st.session_state.detection_history) + 1)),
            y=st.session_state.detection_history,
            mode='lines+markers',
            name='Jackfruits Detected',
            line=dict(color='#FF6B6B', width=4, shape='spline'),
            marker=dict(size=10, color='#4ECDC4', line=dict(color='white', width=2)),
            fill='tozeroy',
            fillcolor='rgba(255, 107, 107, 0.1)'
        ))
        fig.update_layout(
            title="🍈 Detection Trend Over Time",
            xaxis_title="Image Number",
            yaxis_title="Jackfruits Count",
            template="plotly_dark",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Enhanced distribution chart
        if len(st.session_state.detection_history) > 1:
            fig = px.histogram(
                x=st.session_state.detection_history,
                nbins=max(2, len(set(st.session_state.detection_history))),
                title="📊 Detection Distribution",
                labels={'x': 'Jackfruits Detected'},
                template="plotly_dark",
            )
            fig.update_layout(
                xaxis_title="Jackfruits Count",
                yaxis_title="Frequency",
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Upload more images to see distribution chart")

# Footer
st.markdown("""
<div class="footer">
    <p style="color: white; font-size: 1.2rem; margin: 0;">Jackfruit AI Detector | Powered by YOLOv8 & Streamlit</p>
</div>
""", unsafe_allow_html=True)
