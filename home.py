import streamlit as st
import streamlit.components.v1 as stc 
import pandas as pd



from eda import run_eda_app
from prediction import run_predict_app

st.set_page_config(page_title="Iris ML",
		   page_icon="🚢",
		   layout="wide")

html_temp = """
		<div style="background-color:#3872fb;padding:5px;border-radius:10px">
		<h3 style="color:white;text-align:center;font-family:arial;">PJJ Data Analitik Spesialis III 2023</h3>
		<h3 style="color:white;text-align:center;font-family:arial;">Iris Machine Learning</h3>
		<h3 style="color:white;text-align:center;"></h3>
		</div>
		"""


def main():
    stc.html(html_temp)
    menu = ['Home','EDA','Prediction']
    choice = st.sidebar.selectbox("Menu", menu)

    if choice=="Home":
        st.subheader("Home")
        st.write('Aplikasi Sederhana Machine Learning Iris')    
        # iris= Image.open('iris.png')
        st.image('iris.png')
    
    elif choice=='EDA':
        run_eda_app()

    elif choice=='Prediction':
        run_predict_app()

if __name__ == '__main__':
    main()


