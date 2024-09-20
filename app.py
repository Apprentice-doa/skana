import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from keras.losses import SparseCategoricalCrossentropy
import numpy as np
from PIL import Image
import cv2
from camera_input_live import camera_input_live

# Load the pre-trained model
model = load_model('Code/imageclassifier_fixed.h5', compile = False)
# # Compile with a valid loss function and optimizer
# model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy())
# model.save('Code/imageclassifier_fixed.h5')

# print(model.get_config())

st.title("Counterfeit Drug Detector App")

# Function to preprocess an image
def preprocess_image(img):
    # Resize the image to match the input size of the model
    img = img.resize((256, 256))
    # Ensure 3 channels (RGB)
    img = img.convert('RGB')
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Function to make predictions
def predict(img):
    preprocessed_image = preprocess_image(img)
    prediction = model.predict(preprocessed_image)
    return prediction

# Upload image through Streamlit
uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)

    # Make a prediction when a button is clicked
    if st.button('Confirm Authenticity'):
        # Perform prediction``
        prediction = predict(img)
        if prediction > 0.5: 
            st.write(f'This Drug  is a Counterfeit Paracetamol.')
        else:
            st.write(f'This Drug is an Original Paracetamol.')
        
#image = camera_input_live()

st.subheader("Scan the QR code")

# Get camera input
image = st.camera_input("Place Camera on QR code")

if image is not None:
    st.image(image)

    # Convert the camera input to OpenCV format
    bytes_data = image.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # QR code detection and decoding
    detector = cv2.QRCodeDetector()
    data, bbox, straight_qrcode = detector.detectAndDecode(cv2_img)

    if data:
        st.write("# Found QR code")
        st.write("Data:", data)
        with st.expander("Show details"):
            st.write("BBox:", bbox)
            st.write("Straight QR code:", straight_qrcode)
    else:
        st.write('This Drug is a Counterfeit Paracetamol.')





