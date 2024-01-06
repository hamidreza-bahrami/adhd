import streamlit as st
import pandas as pd
import numpy as np
import pickle 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def load_model():
    with open('saved.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

model = data['model']
x = data['x']

def show_page():
    st.write("<h1 style='text-align: center; color: blue;'>مدل تشخیص بیش فعالی و نقص توجه</h1>", unsafe_allow_html=True)
    st.write("<h2 style='text-align: center; color: gray;'>علائم خود را وارد کنید</h2>", unsafe_allow_html=True)
    st.write("<h4 style='text-align: center; color: gray;'>True = بله , False = خیر</h4>", unsafe_allow_html=True)
    st.write("<h4 style='text-align: center; color: gray;'>Robo-Ai.ir طراحی شده توسط</h4>", unsafe_allow_html=True)
    st.link_button("Robo-Ai بازگشت به", "https://robo-ai.ir")

    restlessness = (True , False)
    restlessness = st.selectbox('همواره سرجای خود تکان می خورید و آرام ندارید', restlessness)

    hurry = (True , False)
    hurry = st.selectbox('همیشه برای رفتن عجله دارید و احساس فعال بودن می کنید', hurry)

    talkalot = (True , False)
    talkalot = st.selectbox('هنگام صحبت با دیگران زیاد از حد حرف می زنید', talkalot)

    moving = (True , False)
    moving = st.selectbox('در شرایطی مانند کلاس یا جلسه، از جا بلند می شوید و حرکت می کنید', moving)

    restlessness2 = (True , False)
    restlessness2 = st.selectbox('احساس بیقراری می کنید', restlessness2)

    control = (True , False)
    control = st.selectbox('به سختی نیاز های خود را کنترل می کنید', control)

    cantdo = (True , False)
    cantdo = st.selectbox('برای شروع یا ادامه تکالیف خود به مشکل بر می خورید', cantdo)

    mistake = (True , False)
    mistake = st.selectbox('در انجام امور اشتباهات سهوی زیادی دارید', mistake)

    focus = (True , False)
    focus = st.selectbox('در حفظ تمرکز هنگام انجام یک فعالیت خاص به مشکل برمی خورید', focus)

    motive = (True , False)
    motive = st.selectbox('محرک های محیط مانند صدای تلویزیون به راحتی حواس شما را پرت می کنند', motive)

    forget = (True , False)
    forget = st.selectbox('انجام فعالیت های روتین و روزمره را فراموش می کنید', forget)

    daily = (True , False)
    daily = st.selectbox('برای برنامه ریزی در امور به مشکل می خورید', daily)

    badword = (True , False)
    badword = st.selectbox('در شرایط نامناسب مانند جلسه، حرف نامربوط می زنید', badword)

    patience = (True , False)
    patience = st.selectbox('صبر کردن برایتان دشوار است و زود خشمگین می شوید', patience)

    stoptalk = (True , False)
    stoptalk = st.selectbox('اغلب حین مکالمه حرف دیگران را قطع می کنید', stoptalk)

    unlike = (True , False)
    unlike = st.selectbox('اغلب افکار شما با هنجارها و مرسومات اجتماعی در تضاد است', unlike)

    disturbed = (True , False)
    disturbed = st.selectbox('آیا موارد ذکر شده باعث اختلال عملکرد شغلی، اجتماعی و ... شما شده است؟', disturbed)

    button = st.button('معاینه و تشخیص')
    if button:
        with st.chat_message("assistant"):
                with st.spinner('''درحال بررسی لطفا صبور باشید'''):
                    time.sleep(3)
                    st.success(u'\u2713''بررسی انجام شد')
                    x = np.array([[restlessness, hurry, talkalot, moving, restlessness2, control, cantdo, mistake, focus,
                                   motive, forget, daily, badword, patience, stoptalk, unlike, disturbed]])

        y_prediction = model.predict(x)
        if y_prediction == True:
            st.write("<h4 style='text-align: right; color: gray;'>بر اساس داده های وارد شده، شما دارای بیش فعالی و نقص توجه هستید</h4>", unsafe_allow_html=True)
            st.write("<h5 style='text-align: right; color: gray;'>برای درمان به روانشناس مراجعه کنید</h5>", unsafe_allow_html=True)
        elif y_prediction == False:
            st.write("<h4 style='text-align: right; color: gray;'>بر اساس داده های وارد شده، شما در سلامتی کامل هستید</h4>", unsafe_allow_html=True)

show_page()
