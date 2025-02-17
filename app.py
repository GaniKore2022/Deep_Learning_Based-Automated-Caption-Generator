import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from gtts import gTTS
import io

vgg_model = VGG16()
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

model = load_model('ICG_MODEL.h5')

with open('ICG_Tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)
    
st.set_page_config(page_title="Caption Generator App", page_icon="📷")

st.title("DEEP LEARNING-BASED AUTOMATED CAPTION GENERATOR")
st.markdown(
    "Upload an image, and this app will generate a caption for it using a trained LSTM model."
)

uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    with st.spinner("Generating caption..."):
        image = load_img(uploaded_image, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)

        image_features = vgg_model.predict(image, verbose=0)

        max_caption_length = 35
        
        def get_word_from_index(index, tokenizer):
            return next(
                (word for word, idx in tokenizer.word_index.items() if idx == index), None
            )

        def predict_caption(model, image_features, tokenizer, max_caption_length):
            caption = "startseq"
            for _ in range(max_caption_length):
                sequence = tokenizer.texts_to_sequences([caption])[0]
                sequence = pad_sequences([sequence], maxlen=max_caption_length)
                yhat = model.predict([image_features, sequence], verbose=0)
                predicted_index = np.argmax(yhat)
                predicted_word = get_word_from_index(predicted_index, tokenizer)
                caption += " " + predicted_word
                if predicted_word is None or predicted_word == "endseq":
                    break
            return caption

        generated_caption = predict_caption(model, image_features, tokenizer, max_caption_length)
        generated_caption = generated_caption.replace("startseq", "").replace("endseq", "")

        st.subheader(generated_caption)

        tts = gTTS(generated_caption, lang='en')
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)

        st.audio(audio_buffer, format='audio/mp3')

        st.download_button(
            label="Download Audio",
            data=audio_buffer,
            file_name="Caption_Audio.mp3",
            mime="audio/mp3"
        )

        st.markdown("Generated Caption")
        st.subheader("Uploaded Image")
        st.image(uploaded_image, use_container_width=True)
