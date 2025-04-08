import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageEnhance
import io

# Configure Streamlit page
st.set_page_config(page_title="Text-to-Image Generator", page_icon=":art:")

# Load Stable Diffusion and Anime models with CPU optimizations
@st.cache_resource
def load_models(device):
    models = {
        "Standard": StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device),
        "Anime": StableDiffusionPipeline.from_pretrained("hakurei/waifu-diffusion").to(device),
    }
    return models

# Device selection
st.sidebar.header("Device Options")
device_choice = st.sidebar.radio("Choose the device for processing", ["cpu", "cuda"], index=0)

# Load models based on selected device
models = load_models(device_choice)

# Title and Description with styling
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.5em;
            color: #444DAD;
            font-family: 'Arial', sans-serif;
            font-weight: bold;
        }
        .sub-title {
            font-size: 1.2em;
            color: white;
            font-family: 'Verdana', sans-serif;
        }
        .generate-button {
            background-color: #4CAF50;
            color: white;
            font-size: 1em;
            font-weight: bold;
            border-radius: 5px;
            padding: 10px;
        }
    </style>
    <h1 class="main-title">🎨Text-to-Image Generator</h1>
    <p class="sub-title">Generate stunning images with multiple models, styles, and advanced preferences!</p>
    """,
    unsafe_allow_html=True
)

# Sidebar Configuration
st.sidebar.header("🔧 Model Settings")
model_choice = st.sidebar.selectbox(
    "Choose a Model", 
    list(models.keys()) + ["Black & White", "Enhance", "Realistic"], 
    index=0
)

# Performance Settings
st.sidebar.header("⚡ Performance")
speed_quality = st.sidebar.radio("Generation Preference", ["Speed", "Quality"], index=0)

# Image Dimension Controls
st.sidebar.header("📐 Image Dimensions")
image_width = st.sidebar.slider("Image Width (px)", 256, 1024, 512, step=64)
image_height = st.sidebar.slider("Image Height (px)", 256, 1024, 512, step=64)

# Shape Selection
st.sidebar.header("🖼️ Image Shape")
shapes = {
    "Square": (512, 512), 
    "Portrait": (512, 768), 
    "Landscape": (768, 512)
}
selected_shape = st.sidebar.selectbox("Choose Shape", list(shapes.keys()))
image_width, image_height = shapes[selected_shape]

# Style Options
st.sidebar.header("🌈 Style Options")
styles = ["Default", "Cyberpunk", "Anime", "Fantasy", "Realistic", "Black & White"]
selected_style = st.sidebar.selectbox("Choose Style", styles, index=0)

# Additional Options
download_option = st.sidebar.checkbox("Enable Image Download")

# Image Post-Processing Functions
def apply_black_and_white(image):
    return image.convert("L")

def enhance_image(image):
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(2.0)

def make_realistic(image):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(1.5)

# Main Input Section
prompt = st.text_input("✍️ Enter your image description", placeholder="A fantasy castle on a hill during sunset")
generate_button = st.button("🖌️ Generate Image", key="generate", use_container_width=True, help="Click to generate the image", args=("class", "generate-button"))

# Image Generation
if generate_button:
    if prompt.strip():
        with st.spinner("✨ Generating your image..."):
            try:
                # Select appropriate pipeline
                pipe = models.get(model_choice, models.get("Standard"))
                
                # Apply speed/quality optimization
                if speed_quality == "Speed":
                    pipe.enable_attention_slicing()
                else:
                    pipe.disable_attention_slicing()

                # Append style to prompt
                full_prompt = (f"{prompt}, {selected_style} style" 
                               if selected_style != "Default" 
                               else prompt)

                # Generate image
                generated_image = pipe(
                    full_prompt, 
                    height=image_height, 
                    width=image_width
                ).images[0]

                # Apply post-processing
                if model_choice == "Black & White":
                    generated_image = apply_black_and_white(generated_image)
                elif model_choice == "Enhance":
                    generated_image = enhance_image(generated_image)
                elif model_choice == "Realistic":
                    generated_image = make_realistic(generated_image)

                # Display generated image
                st.image(generated_image, caption="Generated Image", use_container_width=True)

                # Download option
                if download_option:
                    buf = io.BytesIO()
                    generated_image.save(buf, format="PNG")
                    buf.seek(0)
                    st.download_button(
                        label="Download Image",
                        data=buf,
                        file_name="generated_image.png",
                        mime="image/png",
                    )
            
            except Exception as e:
                st.error(f"Image generation failed: {e}")
    else:
        st.error("Please enter a valid text description.")

# Image History Feature
if "image_history" not in st.session_state:
    st.session_state.image_history = []

if generate_button and prompt.strip():
    st.session_state.image_history.append((prompt, generated_image))

# Display Image History
if st.session_state.image_history:
    st.subheader("🖼️ Image Generation History")
    for i, (hist_prompt, hist_image) in enumerate(st.session_state.image_history):
        with st.expander(f"Prompt: {hist_prompt} (Image {i+1})"):
            st.image(hist_image, use_container_width=True)

# Additional Information
st.sidebar.info("""
### 💡 Generation Tips
- Use descriptive, specific prompts
- Experiment with different styles
- Lower inference steps for faster generation
""")
