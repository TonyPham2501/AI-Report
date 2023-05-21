

import streamlit as st
import tensorflow as tf
import time
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input
from streamlit_option_menu import option_menu

# Link emoji https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="Nhận diện các loài côn trùng", page_icon= "🤓",layout = 'wide')

# Navigation menu
with st.sidebar:
    selected = option_menu(
        menu_title = "",
        options = ["Home", "Guide", "Author", "Contact"],
        icons = ["house", "book", "file-earmark-person", "envelope"],
        default_index = 0,
        styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "blue", "font-size": "25px"},
                "nav-link": {
                    "font-size": "25px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "#DDE0E6"},
            }
    )

# Load model
model = tf.keras.models.load_model("Model_Insect2.h5")

# Select mode 
#----------------------------------Home---------------------------------------
if selected == "Home":
    st.title('Nhận diện các bé côn trùng')
    st.divider()
    uploaded_file = st.file_uploader("Chọn ảnh muốn nhận diện tại đây:", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
         # Convert file size
        imga = image.load_img(uploaded_file,target_size=(256,256))
        st.image(imga, channels="RGB")
        img = image.load_img(uploaded_file,target_size=(100,100))

        # Convert to array
        img = img_to_array(img)
        img = img.reshape(1,100,100,3)
        img = img.astype('float32')
        img = img/255
        st.divider()

        # Button detection
        Button_detect = st.button("Detect")
        st.divider()

        if Button_detect:
            with st.spinner("Please wait, Running!!!"):
                time.sleep(2)
            prediction = model.predict(img).argmax()
            y_pred = model.predict(img)
            # Phân loại bệnh
            if prediction == 1:
                st.write("**Đây là con bướm xinh**")
                a = y_pred.max()
                a = a*100
                st.write("**Accuracy:** ",a,"%")

            elif prediction == 2:
                st.write("**Chuồn chuồn nè**")
                a = y_pred.max()
                a = a*100
                st.write("**Accuracy:** ",a,"%")

            elif prediction == 3:
                st.write("**Châu chấu nè he**")
                a = y_pred.max()
                a = a*100
                st.write("**Accuracy:** ",a,"%")

            elif prediction == 4:
                st.write("**Này là con bọ cam nè**")
                a = y_pred.max()
                a = a*100
                st.write("**Accuracy:** ",a,"%")

            elif prediction == 5:
                st.write("**Còn này là em muỗi**")
                a = y_pred.max()
                a = a*100
                st.write("**Accuracy:** ",a,"%")

 
    
#----------------------------------Guide---------------------------------------
if selected == "Guide":
    st.title('Guide')

#----------------------------------About---------------------------------------
if selected == "Author":
    st.title('About the Author:')
    st.write('ㅤㅤㅤPhạm Thành Toàn')
    st.write('ㅤㅤㅤStudent ID:ㅤ20146146')
    st.write('ㅤㅤㅤMajor:ㅤMechatronics')
    st.write('ㅤㅤㅤFME, HCMC University of Technology and Education')
    st.caption('This is Project of Artificial Intelligence subject')
    st.divider()

#----------------------------------Contact---------------------------------------
if selected == "Contact":
    st.title('Contact with me:')
    st.write('ㅤㅤㅤ:label: Facebookㅤ:ㅤhttps://www.facebook.com/profile.php?id=100044297062770')
    st.write('ㅤㅤㅤ:envelope: Gmailㅤ:ㅤtonypham2501@gmail.com')
    st.write('ㅤㅤㅤ:telephone_receiver: Zaloㅤ:ㅤ03747488**')
    st.divider()
st.write('ㅤ')
st.write('ㅤ')
st.write('ㅤ')
st.write('ㅤ')
st.write('ㅤ')
st.write('ㅤ')
st.text('ㅤㅤㅤㅤㅤㅤCopyright © 2023 by Thanh Toan Pham ( Tony Pham )| All rights reserved!')