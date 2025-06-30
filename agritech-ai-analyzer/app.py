import streamlit as st
import os
import gdown
from PIL import Image
from agricultural_ai_system.crop_monitor import get_crop_monitor
from agricultural_ai_system.disease_detector import get_disease_detector
from agricultural_ai_system.nutrient_analyser import get_nutrient_analyzer
from agricultural_ai_system.pest_detector import get_pest_detector
from agricultural_ai_system.soil_monitor import get_soil_analyzer
from agricultural_ai_system.weed_detector import get_weed_detector

# Page configuration
st.set_page_config(
    page_title="AgriTech AI Analyzer",
    page_icon="🌱",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .health-status {
        padding: 0.25em 0.5em;
        border-radius: 0.25em;
        font-weight: bold;
    }
    .healthy {
        background-color: #d1e7dd;
        color: #0f5132;
    }
    .moderate {
        background-color: #fff3cd;
        color: #664d03;
    }
    .unhealthy {
        background-color: #f8d7da;
        color: #842029;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .result-card {
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        background-color: #f8f9fa;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .result-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .result-value {
        font-size: 1rem;
        color: #34495e;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_analyzers():
    """Load all AI models with caching"""
    analyzers = {
        "crop": get_crop_monitor(),
        "disease": get_disease_detector(),
        "nutrient": get_nutrient_analyzer(),
        "pest": get_pest_detector(),
        "soil": get_soil_analyzer(),
        "weed": get_weed_detector()
    }
    return {k: v for k, v in analyzers.items() if v is not None}

def download_models():
    os.makedirs("models", exist_ok=True)
    
    model_urls = {
        "crop_classifier.pth": "https://drive.google.com/uc?id=1uN3mKO5QpMf6Lwqfy65d8GhkD6OPZcfh",
        "plant_disease_model.weights.h5": "https://drive.google.com/uc?id=1pS8COvQ_6S8QGk1aojolRQNZ68iLcz4M",
        "yolov8x-seg.pt": "https://drive.google.com/uc?id=1_Xku-efANU3yT9oMbA9J_gnOKlG6zz1S",
        "soil_scaler.pkl": "https://drive.google.com/uc?id=1X0dEHvZtXbPRRj1qyTNOmAx61AbFc1gN",
        "soil_label_encoder.pkl": "https://drive.google.com/uc?id=1uBWep6TKSKfzZz_-weivsMr0SbGz5ZTi",
        "soil_classifier_rf.pkl": "https://drive.google.com/uc?id=1iIw_xSSOERp5gG2dAoQizkoA7GbdcT4E",
        "soil_classifier_cnn.h5": "https://drive.google.com/uc?id=1eyI5GnpWEY6WQqDGySy4iFS5hE6_8yRd",
        "plant_disease_model.keras": "https://drive.google.com/uc?id=14Tt26X3R7Rdo0jyD6wmXHnhfaH2CCBgh",
        "pest_detection_model.keras": "https://drive.google.com/uc?id=1HW7JZQRh3ssfVRhZq_s-JQAZOMLjSAVB",
        "pest_class_names.npy": "https://drive.google.com/uc?id=1cqjAaSLCcaqPPrv_X3pX6xIcRnXmJTrd",
        "nutrient_model.keras": "https://drive.google.com/uc?id=1OUTCmm2_RlqgB0v4lHMGI3aoixhoR0Pt",
        "nutrient_class_names.npy": "https://drive.google.com/uc?id=1lAuOSck1iqU0XA5372V5AMG1bnuasDjz",
        "disease_class_names.txt": "https://drive.google.com/uc?id=1426wpNbUixwZvL3zk7VF1J7tX9qLW5u_",
        "class_names.txt": "https://drive.google.com/uc?id=1r4BfsJzcm0Xc5Do6V1EUz7LJA88cKw0k"
    }

    for filename, url in model_urls.items():
        output_path = f"models/{filename}"
        if not os.path.exists(output_path):
            try:
                gdown.download(url, output_path, quiet=False)
                # Verify the download wasn't corrupted (should be >1KB)
                if os.path.getsize(output_path) < 1024:
                    os.remove(output_path)
                    raise ValueError(f"Downloaded file {filename} is too small (likely corrupted)")
                print(f"Successfully downloaded {filename}")
            except Exception as e:
                print(f"Failed to download {filename}: {str(e)}")
                # Remove potentially corrupted file
                if os.path.exists(output_path):
                    os.remove(output_path)
                    
def main():
    st.title("🌱 AgriTech AI Analyzer")
    st.markdown("Upload an agricultural image for comprehensive analysis")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False
    )
    
    # Analysis type selection
    analysis_type = st.selectbox(
        "Select analysis type",
        [
            "Crop Health Monitoring",
            "Disease Detection", 
            "Nutrient Deficiency Analysis",
            "Pest Detection", 
            "Soil Classification", 
            "Weed Detection"
        ]
    )
    
    if uploaded_file and analysis_type:
        try:
            # Display original image
            image = Image.open(uploaded_file)
            if image.mode == 'RGBA':
                image = image.convert('RGB')
                
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Analyze Image", type="primary"):
                with st.spinner("Analyzing..."):
                    analyzers = load_analyzers()
                    img_bytes = uploaded_file.getvalue()
                    
                    # Perform analysis based on type
                    analyzer_map = {
                        "Crop Health Monitoring": ("crop", "analyze_crop_from_bytes"),  # Changed from analyze_crop
                        "Disease Detection": ("disease", "analyze_disease_from_bytes"),
                        "Nutrient Deficiency Analysis": ("nutrient", "analyze_leaf_from_bytes"),
                        "Pest Detection": ("pest", "analyze_pest_from_bytes"), 
                        "Soil Classification": ("soil", "analyze_soil_image_from_bytes"),
                        "Weed Detection": ("weed", "detect_weed_from_bytes")
                    }
                    
                    analyzer_key, method_name = analyzer_map[analysis_type]
                    analyzer = analyzers.get(analyzer_key)
                    
                    if not analyzer:
                        st.error(f"{analysis_type} analyzer not available")
                        return
                    
                    analysis_method = getattr(analyzer, method_name)
                    result = analysis_method(img_bytes)
                    
                    # Display results
                    display_results(result, analysis_type)
                    
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

def display_results(result, analysis_type):
    if result.get("status") == "error":
        st.error(f"Error: {result.get('error', 'Unknown error')}")
        return
    
    st.success("Analysis Completed Successfully!")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Analysis Results")
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        
        if analysis_type == "Crop Health Monitoring":
            if "crop_class" in result:
                st.markdown('<p class="result-title">Crop Class</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="result-value">{result["crop_class"]}</p>', unsafe_allow_html=True)
            
            if "health_status" in result:
                health_status = result["health_status"]
                status_class = health_status.lower()
                st.markdown('<p class="result-title">Health Status</p>', unsafe_allow_html=True)
                st.markdown(
                    f'<p class="result-value"><span class="health-status {status_class}">{health_status}</span></p>',
                    unsafe_allow_html=True
                )
            
            if "color_analysis" in result:
                st.markdown('<p class="result-title">Green Ratio</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="result-value">{result["color_analysis"].get("green_ratio", "N/A")}</p>', unsafe_allow_html=True)
        
        elif analysis_type == "Disease Detection":
            st.markdown('<p class="result-title">Disease Detected</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="result-value">{"Yes" if result.get("disease_detected") else "No"}</p>', unsafe_allow_html=True)
            
            if result.get('disease_type'):
                st.markdown('<p class="result-title">Disease Type</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="result-value">{result["disease_type"]}</p>', unsafe_allow_html=True)
            
            st.markdown('<p class="result-title">Confidence</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="result-value">{result.get("confidence", "N/A")}%</p>', unsafe_allow_html=True)
            
            if "recommendations" in result:
                st.markdown('<p class="result-title">Recommendations</p>', unsafe_allow_html=True)
                for rec in result["recommendations"]:
                    st.markdown(f'<p class="result-value">- {rec}</p>', unsafe_allow_html=True)
        
        elif analysis_type == "Nutrient Deficiency Analysis":
            if "primary_deficiency" in result:
                st.markdown('<p class="result-title">Primary Deficiency</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="result-value">{result["primary_deficiency"]}</p>', unsafe_allow_html=True)
            
            if "confidence" in result:
                st.markdown('<p class="result-title">Confidence</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="result-value">{result["confidence"]}%</p>', unsafe_allow_html=True)
            
            if "recommendations" in result:
                st.markdown('<p class="result-title">Recommendations</p>', unsafe_allow_html=True)
                for rec in result["recommendations"]:
                    st.markdown(f'<p class="result-value">- {rec}</p>', unsafe_allow_html=True)
        
        elif analysis_type == "Pest Detection":
            if "pest_detected" in result:
                st.markdown('<p class="result-title">Pest Detected</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="result-value">{"Yes" if result["pest_detected"] else "No"}</p>', unsafe_allow_html=True)
            
            if "pest_type" in result:
                st.markdown('<p class="result-title">Pest Type</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="result-value">{result["pest_type"]}</p>', unsafe_allow_html=True)
            
            if "confidence" in result:
                st.markdown('<p class="result-title">Confidence</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="result-value">{result["confidence"]}%</p>', unsafe_allow_html=True)
        
        elif analysis_type == "Soil Classification":
            if "classification_prediction" in result:
                st.markdown('<p class="result-title">RF Prediction</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="result-value">{result["classification_prediction"]}</p>', unsafe_allow_html=True)
            
            if "deep_learning_prediction" in result:
                st.markdown('<p class="result-title">CNN Prediction</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="result-value">{result["deep_learning_prediction"]}</p>', unsafe_allow_html=True)
        
        elif analysis_type == "Weed Detection":
            if "weed_present" in result:
                st.markdown('<p class="result-title">Weed Detected</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="result-value">{"Yes" if result["weed_present"] else "No"}</p>', unsafe_allow_html=True)
            
            if "confidence" in result:
                st.markdown('<p class="result-title">Confidence</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="result-value">{result["confidence"]}%</p>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.subheader("📈 Visualization")
        if "visualization" in result and result["visualization"]:
            if result["visualization"].startswith("data:image"):
                st.image(result["visualization"], use_column_width=True)
            else:
                st.warning("Visualization not available in expected format")
        else:
            st.info("No visualization available for this analysis")
if __name__ == "__main__":
    download_models()
    main()
