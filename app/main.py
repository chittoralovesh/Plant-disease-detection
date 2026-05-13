import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st
import streamlit.components.v1 as components


working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# loading the class names
class_indices = json.load(open(f"{working_dir}/class_indices.json"))


# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Ensure image is in RGB format (drops alpha channel for PNGs)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array


# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img, verbose=0)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name


# Streamlit App
st.title('Early Disease Detection In Plants Using Deep Learning')

# Custom CSS for Classic and 3D Look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');

    /* Custom Futuristic Solarpunk Cursors */
    * {
        cursor: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"><path d="M2 2L9.5 22L13 14L21 11L2 2Z" fill="%231ce872" stroke="%23041208" stroke-width="1.5"/></svg>'), auto !important;
    }
    
    a, button, .stButton > button, [role="button"], [data-testid="stFileUploader"] {
        cursor: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"><circle cx="12" cy="12" r="8" fill="rgba(28,232,114,0.2)" stroke="%231ce872" stroke-width="2"/><circle cx="12" cy="12" r="2" fill="%231ce872"/></svg>') 12 12, pointer !important;
    }

    /* Futuristic Solarpunk/Nature Background */
    .stApp {
        background-color: transparent !important;
        background-image: none !important;
        font-family: 'Space Grotesk', sans-serif !important;
        color: #d1e8d5;
    }
    
    /* Bioluminescent Main Title (h1) */
    h1 {
        color: #1ce872;
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 3px;
        text-shadow: 0 0 15px rgba(28, 232, 114, 0.5), 0 0 30px rgba(28, 232, 114, 0.3);
    }

    /* Distinct Subtitles (h2, h3, etc) */
    h2, h3, h4, h5, h6 {
        color: #84e296;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600;
        text-transform: none;
        letter-spacing: 1px;
        text-shadow: 0 0 10px rgba(132, 226, 150, 0.4);
    }
    
    /* Dark Glass Nature Sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(4, 18, 8, 0.85);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border-right: 1px solid rgba(28, 232, 114, 0.2);
        box-shadow: 5px 0 20px rgba(28, 232, 114, 0.05);
    }
    
    /* Solarpunk Outline Buttons */
    .stButton > button {
        background: transparent !important;
        border: 1px solid #1ce872 !important;
        border-radius: 4px !important;
        box-shadow: 0 0 10px rgba(28, 232, 114, 0.2), inset 0 0 5px rgba(28, 232, 114, 0.1) !important;
        color: #1ce872 !important;
        font-weight: 700 !important;
        font-family: 'Rajdhani', sans-serif !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
        padding: 10px 26px !important;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button:hover {
        background: rgba(28, 232, 114, 0.15) !important;
        box-shadow: 0 0 20px rgba(28, 232, 114, 0.5), inset 0 0 10px rgba(28, 232, 114, 0.3) !important;
        transform: translateY(-2px) scale(1.02) !important;
        color: #ffffff !important;
    }
    
    .stButton > button:active {
        transform: translateY(1px) scale(0.98) !important;
        box-shadow: 0 0 10px rgba(28, 232, 114, 0.3) !important;
    }
    
    /* Organic High-tech File Uploader */
    [data-testid="stFileUploader"] {
        background-color: rgba(8, 25, 13, 0.6);
        backdrop-filter: blur(10px);
        border: 1px dashed #1ce872;
        border-radius: 8px;
        box-shadow: inset 0 0 15px rgba(28, 232, 114, 0.05);
        padding: 24px;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        background-color: rgba(12, 35, 18, 0.8);
        border: 1px solid #1ce872;
        box-shadow: inset 0 0 25px rgba(28, 232, 114, 0.2), 0 0 15px rgba(28, 232, 114, 0.1);
        transform: translateY(-2px);
    }

    /* Solarpunk Holographic Images */
    img {
        padding: 4px;
        background: rgba(28, 232, 114, 0.05);
        backdrop-filter: blur(5px);
        border-radius: 4px;
        box-shadow: 0 0 20px rgba(28, 232, 114, 0.1);
        border: 1px solid rgba(28, 232, 114, 0.3);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        filter: contrast(1.1) brightness(1.1) saturate(1.1);
    }
    
    img:hover {
        transform: translateY(-8px) scale(1.03);
        box-shadow: 0 0 30px rgba(28, 232, 114, 0.3);
        border-color: rgba(28, 232, 114, 0.7);
        background: rgba(28, 232, 114, 0.1);
    }
    
    /* Markdown Text container */
    [data-testid="stMarkdownContainer"] {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #d1e8d5;
        font-size: 1.05rem;
        line-height: 1.7;
    }
    
    /* Make bold text bright emerald */
    [data-testid="stMarkdownContainer"] strong {
        color: #84e296;
        text-shadow: 0 0 5px rgba(132, 226, 150, 0.3);
    }

    /* Cards for main layout items */
    div.element-container {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# JavaScript injection to create a 3D animated background using Three.js
components.html(
    """
    <script>
    const parentDoc = window.parent.document;
    
    // Only initialize if not already done
    if (!parentDoc.getElementById('three-js-script')) {
        const script = parentDoc.createElement('script');
        script.id = 'three-js-script';
        script.src = 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js';
        script.onload = initThreeJS;
        parentDoc.head.appendChild(script);
    } else if (!parentDoc.getElementById('bg-canvas')) {
        initThreeJS();
    }

    function initThreeJS() {
        if (parentDoc.getElementById('bg-canvas')) return;

        const canvas = parentDoc.createElement('canvas');
        canvas.id = 'bg-canvas';
        canvas.style.position = 'fixed';
        canvas.style.top = '0';
        canvas.style.left = '0';
        canvas.style.width = '100vw';
        canvas.style.height = '100vh';
        canvas.style.zIndex = '-1';
        parentDoc.body.appendChild(canvas);

        const THREE = window.parent.THREE;
        if (!THREE) return;

        const scene = new THREE.Scene();
        scene.fog = new THREE.FogExp2(0x041208, 0.002);
        
        const camera = new THREE.PerspectiveCamera(75, window.parent.innerWidth / window.parent.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ canvas: canvas, alpha: true, antialias: true });
        
        renderer.setSize(window.parent.innerWidth, window.parent.innerHeight);
        renderer.setPixelRatio(window.parent.devicePixelRatio);
        renderer.setClearColor(0x041208, 1);

        // Solarpunk Bioluminescent Particle System
        const geometry = new THREE.BufferGeometry();
        const particlesCount = 2500;
        const posArray = new Float32Array(particlesCount * 3);
        const colorArray = new Float32Array(particlesCount * 3);
        
        const color1 = new THREE.Color(0x1ce872); // Bright Emerald
        const color2 = new THREE.Color(0x84e296); // Soft Green
        
        for(let i = 0; i < particlesCount * 3; i++) {
            // Spread particles over a large volume
            posArray[i] = (Math.random() - 0.5) * 120;
            
            // Assign random color mix
            if (i % 3 === 0) {
                const mixColor = Math.random() > 0.4 ? color1 : color2;
                colorArray[i] = mixColor.r;
                colorArray[i+1] = mixColor.g;
                colorArray[i+2] = mixColor.b;
            }
        }
        
        geometry.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colorArray, 3));

        const material = new THREE.PointsMaterial({
            size: 0.15,
            vertexColors: true,
            transparent: true,
            opacity: 0.8,
            blending: THREE.AdditiveBlending
        });

        const particlesMesh = new THREE.Points(geometry, material);
        scene.add(particlesMesh);

        camera.position.z = 30;

        let mouseX = 0;
        let mouseY = 0;
        const mouse = new THREE.Vector2(9999, 9999);
        const raycaster = new THREE.Raycaster();
        const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
        const target = new THREE.Vector3();

        parentDoc.addEventListener('mousemove', (event) => {
            // Normalized device coordinates for raycaster
            mouse.x = (event.clientX / window.parent.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.parent.innerHeight) * 2 + 1;
        });

        window.parent.addEventListener('resize', () => {
            camera.aspect = window.parent.innerWidth / window.parent.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.parent.innerWidth, window.parent.innerHeight);
        });

        const clock = new THREE.Clock();

        // Store original positions for spring back effect
        const originalPositions = new Float32Array(posArray);

        function animate() {
            requestAnimationFrame(animate);
            const elapsedTime = clock.getElapsedTime();

            // Continuous slow rotation of the entire system
            particlesMesh.rotation.y = elapsedTime * 0.02;
            particlesMesh.rotation.x = elapsedTime * 0.01;
            
            // Repulsion interaction
            raycaster.setFromCamera(mouse, camera);
            raycaster.ray.intersectPlane(plane, target);

            const positions = geometry.attributes.position.array;

            // Since the mesh is rotating, we need to convert mouse world position 
            // into local space of the mesh to interact with particles correctly
            const localTarget = target.clone();
            particlesMesh.worldToLocal(localTarget);

            for (let i = 0; i < particlesCount; i++) {
                const i3 = i * 3;
                
                const px = positions[i3];
                const py = positions[i3 + 1];
                const pz = positions[i3 + 2];
                
                // Distance to cursor
                const dx = px - localTarget.x;
                const dy = py - localTarget.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                
                const origX = originalPositions[i3];
                const origY = originalPositions[i3 + 1];
                
                const repelRadius = 15;
                if (dist < repelRadius) {
                    const force = (repelRadius - dist) / repelRadius;
                    // Push particles away
                    positions[i3] += dx * force * 0.05;
                    positions[i3 + 1] += dy * force * 0.05;
                } else {
                    // Spring back to original position smoothly
                    positions[i3] += (origX - px) * 0.02;
                    positions[i3 + 1] += (origY - py) * 0.02;
                }
            }
            geometry.attributes.position.needsUpdate = true;

            // Gentle floating breath on Z axis for the whole mesh
            particlesMesh.position.z = Math.sin(elapsedTime * 0.5) * 2;

            renderer.render(scene, camera);
        }
        animate();
    }
    </script>
    """,
    height=0,
    width=0,
)

#TRYING
# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition","Cures"])

# Main Page
if (app_mode == "Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = f"{working_dir}/home_banner.png"
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(Image.open(image_path), use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍 ~ By Lovesh

    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases.

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.
    4. **For Cures** Later open the cures page from the bar and you'll get cures for the disease.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

# About Project
elif (app_mode == "About"):
    st.header("About")
    about_image_path = f"{working_dir}/about_page.png"
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(Image.open(about_image_path), use_column_width=True)
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

                """)

elif (app_mode == "Cures"):
    st.header("Cures")
    cures_image_path = f"{working_dir}/cures_page.png"
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(Image.open(cures_image_path), use_column_width=True)
    st.markdown("""
                #### The Cures for the disease:
                The cures are:
                1. Remove and destroy infected plant parts.
                2. Apply fungicides or bactericides.
                3. Prune affected areas to improve air circulation.
                4. Adjust irrigation to avoid waterlogged conditions.
                5. Use biological control agents.
                6. Apply compost or organic mulch.
                7. Improve soil drainage.
                8. Implement crop rotation.
                9. Use resistant plant varieties.
                10. Apply neem oil as a natural pesticide.
                11. Provide adequate nutrition through fertilization.
                12. Introduce beneficial insects.
                13. Quarantine infected plants.
                14. Use reflective mulch to deter pests.
                15. Apply horticultural oils.
                16. Implement pheromone traps.
                17. Use row covers to protect plants.
                18. Apply copper-based fungicides.
                19. Adjust pH levels in soil.
                20. Encourage natural predators like ladybugs.
                21. Use disease-free seeds or transplants.
                22. Apply biopesticides derived from plants.
                23. Ensure proper spacing between plants.
                24. Rotate herbicides to prevent resistance.
                25. Apply foliar sprays of seaweed extract.
                26. Use sterile growing media.
                27. Encourage biodiversity in the garden.
                28. Practice deep watering to promote root health.
                29. Apply beneficial nematodes to the soil.
                30. Introduce companion plants for pest control.
                31. Apply insecticidal soaps.
                32. Provide adequate sunlight exposure.
                33. Use sticky traps for flying pests.
                34. Apply sulfur-based fungicides.
                35. Introduce mycorrhizal fungi to enhance nutrient uptake.
                36. Practice proper sanitation in the garden.
                37. Apply diatomaceous earth for pest control.
                38. Implement drip irrigation to reduce foliage wetness.
                39. Use physical barriers to exclude pests.
                40. Apply hot water treatment to seeds.
                41. Use resistant rootstocks for grafting.
                42. Apply potassium bicarbonate to control powdery mildew.
                43. Practice proper pruning techniques.
                44. Use botanical extracts with antifungal properties.
                45. Apply bacillus thuringiensis (Bt) for caterpillar control.
                46. Introduce predatory mites for pest control.
                47. Practice seed treatment with fungicides.
                48. Apply reflective barriers to repel insects.
                49. Use beneficial fungi for soil health.
                50. Seek professional advice for specific NBSP disease.
                """)


elif (app_mode == "Disease Recognition"):
    st.header("Disease Recognition")
    st.markdown("Upload an image of a plant leaf to identify potential diseases.")
    
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        col1, col2 = st.columns(2)

        with col1:
            resized_img = image.resize((150, 150))
            st.image(resized_img)

        with col2:
            if st.button('Classify'):
                # Rewind the file pointer before reading again
                uploaded_image.seek(0)
                # Preprocess the uploaded image and predict the class
                prediction = predict_image_class(model, uploaded_image, class_indices)
                st.success(f'Prediction: {str(prediction)}')