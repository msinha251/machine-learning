# streamlit app
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

st.title('Dino or Dragon?')
st.write('Upload an image of a dinosaur or a dragon and find out if it is a dinosaur or a dragon')

uploaded_file = st.file_uploader('Choose an image...', type='jpg')

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=False, width=150)
    st.write('')
    st.write('Classifying...')

    model = tf.keras.models.load_model('./models/base-dino-dragon.h5')
    img = np.array(image)
    img = img / 255.0
    img = tf.image.resize(img, (150, 150))
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img)
    
    if preds[0][0] > 0.5:
        st.write(f'This is a dragon with a {preds[0][0]*100:.2f}% probability')
    else:
        st.write(f'This is a dinosaur with a {preds[0][0]*100:.2f}% probability')
else:
    st.write('Please upload an image')