import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import pathlib
import platform

plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath
st.title("Hayvonlarni klassifikatsiya qiluvchi model")


file = st.file_uploader('Rasm yuklash', type = ['png', 'jpeg', 'gif', 'svg', 'jpg'])
if file:
    st.image(file)

    img = PILImage.create(file)

    model = load_learner('animals_model.pkl')

    model.predict(img)

    pred, pred_id, prob = model.predict(img)
    st.success(f"Bashorat: {pred}")
    st.info(f"Ehtimollik: {prob[pred_id]*100:.1f}%")

    fig = px.bar(x=prob*100, y=model.dls.vocab)
    st.plotly_chart(fig)
