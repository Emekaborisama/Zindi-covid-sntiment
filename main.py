import streamlit as st
import sys
import time
import requests
import json






def endpoint1(texte):
    url = "https://ilistener2-emekaborisama.cloud.okteto.net/named_entity"
    payload={'text': texte}
    files=[

    ]
    headers = {}
    response = requests.request("POST", url, headers=headers, data=payload, files=files)
    return (response.text)




html_temp = """
<div style = "background.color:teal; padding:10px">
<h2 style = "color:white; text_align:center;"> demo/h2>
<p style = "color:white; text_align:center;"> demo </p>
</div>
"""
st.markdown(html_temp, unsafe_allow_html = True)





#st.cache()
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """


input_text = st.text_area("text")

if st.button("NER"):
    json_file = endpoint1(texte = input_text)
    st.code(json_file, language = 'python')




