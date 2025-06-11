import time
import pandas as pd
import numpy as np
import datetime as dt
from PIL import Image
import requests


import streamlit as st
import streamlit.components.v1 as components



import plotly.graph_objects as go
import plotly.express as px
px_templates = ['plotly', 'plotly_white', 'plotly_dark', 'ggplot2', 'seaborn', 'simple_white', 'presentation', 'none']

st.set_page_config(page_title="Kedro Streamlit App", layout='wide')


import streamlit as st
from PIL import Image

import json 
#set_page_config(layout="wide")
st.markdown('# Kedro App Manager')
st.sidebar.markdown('# Main page')
st.sidebar.markdown('This is the application\'s main page.')



st.markdown('### App Documentation')
st.markdown('''This streamlit application is intended manage Kedro assets.''')
st.markdown('##### How to use this app')
st.markdown('''To run this application you need to do the following:\n
1) Navigate to the project where this applicaion is located.\n 
2) Open a session with the "Editor" option chosen as "Streamlit(Run)".
Note: The "STREAMLIT_RUN" environmental variable needs to be set to
the main streamlit application file.\n
3) Now you can navigate through the different application pages.''')
st.markdown('##### How to contribute to this app')
st.markdown('''If you want to contribute to this application you can easily do so.
In case you want to simply modify one of the pages, just navigate to the folder where the 
application is at, enter the "pages" subdirectory and modify the .py file corresponding 
to the page.\n
In case you want to add a new page, navigate to the "pages" subdirectory and 
create a new python file. The first lines of this file should be set to
''')
st.code('''
import streamlit as st
st.sidebar.markdown(nameOfPage)
''')







