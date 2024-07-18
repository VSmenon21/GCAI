import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
import os
import joblib
from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
import io
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Configure GenkiCheck AI Doctor Assistant
API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])

instruction = """Your name is GenkiCheck AI Personal assistant🤖🩺. You are a healthcare chatbot that can help in answering patients' queries. Only reply to health-related questions; if the question is not related to healthcare, then reply as 'I cannot answer if it's not healthcare related.'"""

def chatbot(message):
    response = chat.send_message(instruction + message)
    result = response.text
    return result

# Load your models
skin_model_path = 'models/skin.h5'
alzheimer_model_path = 'models/svm_alzheimer_model.joblib'

skin_model = load_model(skin_model_path)
alzheimer_model = joblib.load(alzheimer_model_path)

# Define categories for Alzheimer's prediction
alzheimer_categories = ["Non_Demented", "Very_Mild_Demented", "Mild_Demented", "Moderate_Demented"]

# Function to preprocess the skin image
def preprocess_skin_image(img):
    img = img.resize((224, 224))  # Adjust size as per your model's input_shape
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.  # Normalize image data
    return img_array

# Function to preprocess the MRI image
def preprocess_alzheimer_image(image):
    width, height = 256, 256
    img = image.resize((width, height))
    img_array = np.array(img)
    img_wide = img_array.reshape(1, width * height)
    scaler = StandardScaler()
    img_wide = scaler.fit_transform(img_wide)
    return img_wide

# Function to predict skin cancer
def predict_skin(img):
    img_array = preprocess_skin_image(img)
    prediction = skin_model.predict(img_array)
    return prediction
# Function to predict Alzheimer's
def predict_alzheimer(image):
    img_array = preprocess_alzheimer_image(image)
    prediction = alzheimer_model.predict(img_array)
    return alzheimer_categories[prediction[0]]

# Function to display skin cancer prediction section
def display_skin_cancer_prediction():
    st.title('Skin Cancer Prediction')
    st.markdown('Upload a skin lesion image for prediction.📷')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Predict button
        if st.button('Predict'):
            # Read the uploaded file as PIL image
            pil_img = Image.open(uploaded_file)
            
            # Make prediction
            prediction = predict_skin(pil_img)
            prediction_label = "Benign" if prediction[0][0] > 0.5 else "Malignant"
            prediction_confidence = prediction[0][0] if prediction_label == "Benign" else 1 - prediction[0][0]

            col1, col2 = st.columns(2)

            with col1:
                st.image(pil_img, caption='Uploaded Image', use_column_width=False, width=200)

            with col2:
                st.markdown(f'### ***Prediction:*** {prediction_label}')
                st.markdown(f'### ***Confidence:*** {prediction_confidence:.2%}')

            # Get advice from GenkiCheck AI Doctor Assistant
            if prediction_label == 'Benign':
                prompt = """
                    This is a diagnosis of Benign skin lesion. Assume that you are an experienced doctor. Provide guidance on managing and monitoring the condition.
                    1. Reasons
                    2. Prevention
                    3. Regular Exercises and medicines to prevent it
                    Give the advice in this format and give a headling h2 to the bullet points in the prompt!
                """
            else:
                prompt = """
                    This is a diagnosis of Malignant skin lesion. Assume that you are an experienced doctor. Provide guidance on managing and treating the condition.
                    1. Reasons
                    2. Prevention
                    3. Regular Exercises and medicines to prevent it
                    Give the advice in this format and give a headling h2 to the bullet points in the prompt!
                """

            advice = chatbot(prompt)
            st.markdown(f'### ***Advice:*** {advice}')

            # Suggest a day-to-day plan for recovery
            daily_plan = chatbot(f"Suggest a day-to-day recovery plan for a patient diagnosed with {prediction_label} skin cancer.")
            st.markdown(f'### ***Recovery Plan:*** {daily_plan}')

        # User query input
        user_query = st.text_input("Ask GenkiCheck AI Personal Assistant 🤖🩺 a health-related question:")
        if st.button("Get Advice"):
            if user_query.strip() != "":
                advice = chatbot(user_query)
                title_color = 'pink'
                with st.container():
                    st.markdown(f"<h1 style='color: {title_color}; text-align: center;'>GenkiCheck AI Personal Assistant 🤖🩺</h1>", unsafe_allow_html=True)
                    st.markdown(f'{advice}')
            else:
                st.write("Please enter a valid question.")

# Function to display Alzheimer's prediction section
def display_alzheimer_prediction():
    st.title("Alzheimer's Disease Prediction")
    st.markdown('Upload an MRI scan image for prediction.🧠')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)

        if st.button('Predict'):
            prediction = predict_alzheimer(img)

            # Display prediction in two columns
            col1, col2 = st.columns(2)

            with col1:
                st.image(img, caption='Uploaded Image', use_column_width=False, width=200)

            with col2:
                st.markdown(f'### ***Prediction:*** {prediction}')

            # Get advice from GenkiCheck AI Doctor Assistant
            if prediction == 'Non_Demented':
                prompt = """
                    This is a healthcare related question. This is an MRI scan of a Non-Demented patient. Assume that you are an experienced doctor. Analyze it for potential signs of Alzheimer's disease (cognitive decline, Memory loss, Behavioral changes)
                    1. Reasons
                    2. Prevention
                    3. Regular Exercises and medicines to prevent it
                    Give the advice in this format and give a headling h2 to the bullet points in the prompt!
                """
            elif prediction == 'Very_Mild_Demented':
                prompt = """
                    This is a healthcare related question. This is an MRI scan of a Very_Mild_Demented patient. Assume that you are an experienced doctor. Analyze it for potential signs of Alzheimer's disease (cognitive decline, Memory loss, Behavioral changes)
                    1. Reasons
                    2. Prevention
                    3. Regular Exercises and medicines to prevent it
                    Give the advice in this format and give a headling h2 to the bullet points in the prompt!
                """
            elif prediction == 'Mild_Demented':
                prompt = """
                    This is a healthcare related question. This is an MRI scan of a Mild_Demented patient. Assume that you are an experienced doctor. Analyze it for potential signs of Alzheimer's disease (cognitive decline, Memory loss, Behavioral changes)
                    1. Reasons
                    2. Prevention
                    3. Regular Exercises and medicines to prevent it
                    Give the advice in this format and give a headling h2 to the bullet points in the prompt!
                """
            elif prediction == 'Moderate_Demented':
                prompt = """
                    This is a healthcare related question. This is an MRI scan of a Moderate_Demented patient. Assume that you are an experienced doctor. Analyze it for potential signs of Alzheimer's disease.
                    1. Reasons
                    2. Prevention
                    3. Regular Exercises and medicines to prevent it
                    Give the advice in this format and give a headling h2 to the bullet points in the prompt!
                """
            else:
                prompt = "No specific insights available for this prediction."

            advice = chatbot(prompt)
            title_color = 'pink'

            centered_container = st.container()

            with centered_container:
                st.markdown(f"<h1 style='color: {title_color}; text-align: center;'>GenkiCheck AI Personal Assistant 🤖🩺</h1>", unsafe_allow_html=True)            
            st.markdown(f'### ***Advice:*** {advice}')

            # Suggest a day-to-day plan for recovery
            daily_plan = chatbot(f"Suggest a day-to-day recovery plan for a patient diagnosed with {prediction}.")
            st.markdown(f'### ***Recovery Plan:*** {daily_plan}')

        user_query = st.text_input("Ask GenkiCheck AI Personal Assistant 🤖🩺 a health-related question:")
        if st.button("Get Advice"):
            if user_query.strip() != "":
                advice = chatbot(user_query)
                title_color = 'pink'
                with st.container():
                    st.markdown(f"<h1 style='color: {title_color}; text-align: center;'>GenkiCheck AI Personal Assistant 🤖🩺</h1>", unsafe_allow_html=True)
                    st.markdown(f'{advice}')
            else:
                st.write("Please enter a valid question.")

# Load Xception model with original top layers
model_builder = keras.applications.xception.Xception
img_size = (299, 299)
preprocess_input = keras.applications.xception.preprocess_input
last_conv_layer_name = "block14_sepconv2_act"

# Load model
base_model = model_builder(weights='models/xception_weights_tf_dim_ordering_tf_kernels.h5', include_top=True)

# Modify the last layer to have 4 output units
x = base_model.layers[-2].output  # Get the second last layer's output
output = keras.layers.Dense(4, activation='softmax')(x)
model = keras.models.Model(inputs=base_model.input, outputs=output)

# Function to preprocess the image
def preprocess_image(image_bytes, target_size):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert('RGB')  # Ensure the image has 3 channels (RGB)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Function to create Grad-CAM heatmap
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Function to display Grad-CAM output
def save_and_display_gradcam(img_bytes, heatmap, cam_path="cam.jpg", alpha=0.4):
    img = Image.open(io.BytesIO(img_bytes))
    img = img.convert('RGB')  # Ensure the image has 3 channels (RGB)
    img = np.array(img)

    heatmap = np.uint8(255 * heatmap)
    jet = plt.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    superimposed_img.save(cam_path)
    return superimposed_img

# Function to display X-ray disease prediction with Grad-CAM visualization and GenkiCheck AI Personal Assistant
def display_xray_prediction():
    st.title('X-ray Disease Prediction with Grad-CAM')
    st.markdown('Upload a chest X-ray image for prediction and Grad-CAM visualization. 🩻')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        if st.button('Predict and Visualize'):
            img_bytes = uploaded_file.read()
            img_array = preprocess_image(img_bytes, target_size=img_size)
            preds = model.predict(img_array)
            heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
            superimposed_img = save_and_display_gradcam(img_bytes, heatmap, cam_path="cam.jpg")

            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.image(img_bytes, caption='Uploaded Image', use_column_width=False, width=200)
            with col2:
                st.image(superimposed_img, caption='Grad-CAM', use_column_width=False, width=200)

            # Display prediction
            classes = ['Normal', 'COVID', 'Lung_Opacity', 'Pneumonia']
            prediction_label = classes[np.argmax(preds[0])]
            prediction_confidence = np.max(preds[0])
            st.markdown(f'### ***Prediction:*** {prediction_label}')
            # st.markdown(f'**Confidence:** {prediction_confidence:.2%}')

            # Get advice from GenkiCheck AI Personal Assistant based on prediction
            prompt = f"""This is a healthcare related question. This is a chest X-ray scan classified as {prediction_label}. Assume that you are an experienced doctor. Analyze it for potential signs(must) and provide relevant medical insights.
            1. Reasons
            2. Prevention
            3. Regular Exercises and medicines to prevent it
            Give the advice in this format and give a headling h2 in display to the bullet points in the prompt!"""
            advice = chatbot(prompt)
            title_color = 'pink'

            with st.container():
                st.markdown(f"<h1 style='color: {title_color}; text-align: center;'>GenkiCheck AI Personal Assistant 🤖🩺</h1>", unsafe_allow_html=True)
                st.markdown(f'### ***Advice:*** {advice}')

        # Allow user to ask additional health-related questions
        user_query = st.text_input("Ask GenkiCheck AI Personal Assistant 🤖🩺 a health-related question:")
        if st.button("Get Advice"):
            if user_query.strip() != "":
                advice = chatbot(user_query)
                title_color = 'pink'
                with st.container():
                    st.markdown(f"<h1 style='color: {title_color}; text-align: center;'>GenkiCheck AI Personal Assistant 🤖🩺</h1>", unsafe_allow_html=True)
                    st.markdown(f'{advice}')
            else:
                st.write("Please enter a valid question.")


def main():
    st.set_page_config(page_title='GenkiCheck🤖🩺', page_icon=':microscope:', layout='wide')

    # Custom CSS for aesthetics
    st.markdown("""
        <style>
        body {
            background-image: url('https://img.freepik.com/premium-photo/artificial-intelligence-doctor-concept-ai-medicine-ai-assisted-diagnostic_706554-13.jpg'); /* Replace with your image URL */
            background-size: cover;
            background-repeat: no-repeat;
            color: #FFFFFF; /* Change text color to white */
        }
        .stApp {
            background: rgba(0, 0, 0, 0.7); /* Semi-transparent black background */
            color: #FFFFFF;
            font-family: 'Helvetica Neue', sans-serif;
        }
        .stButton>button {
            background-color: #1f77b4 !important;
            color: white !important;
            font-size: 18px;
        }
        .stMarkdown p {
            font-size: 18px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar with logo and navigation
    st.sidebar.image("GenkiCheck logo.png", use_column_width=True)
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Skin Cancer Prediction", "Alzheimer's Prediction", "X-Ray Chest Disease Prediction"])

    # Main content based on selection
    if selection == "Skin Cancer Prediction":
        display_skin_cancer_prediction()
    elif selection == "Alzheimer's Prediction":
        display_alzheimer_prediction()
    elif selection == "X-Ray Chest Disease Prediction":
        display_xray_prediction()

# Run the app
if __name__ == '__main__':
    main()
