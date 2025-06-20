
# streamlit libray
import streamlit as st
import streamlit.components.v1 as components
import pickle
import sys
from PIL import Image
import pandas as pd
from kedro.io import DataCatalog
from streamlit_ydata_profiling import st_profile_report
import time
import pandas as pd
import requests
from pathlib import Path
from kedro.framework.project import configure_project

package_name = Path(__file__).parent.name
configure_project(package_name)

KEDRO_VIZ_SERVER_URL = 'http://127.0.0.1:4141/'

import subprocess
import sys

import os
from threading import Thread
from IPython import get_ipython
import multiprocessing as mp



st.set_page_config(layout='wide')
st.title('Run pipelines from Kedro')

# from IPython.terminal.interactiveshell import TerminalInteractiveShell
# shell = TerminalInteractiveShell.instance()
# shell.display_formatter.display_columns = 300

import subprocess

# command = ["kedro" ,"run","--pipeline=ingestion" ]
# process = subprocess.Popen(command, cwd="../", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

# while process.poll() is None:
#     line = process.stdout.readline()
#     if not line:
#         continue
#     st.write(line.strip())
from contextlib import redirect_stdout, redirect_stderr
import io
import sys
import subprocess
import traceback
import streamlit as st
import pandas as pd

# from kedro.framework.project import find_pipelines

# import sys
# import os

# # Get the directory of the current script
# current_dir = os.path.dirname(__file__)

# # Get the path to the root directory by navigating 3 levels up
# root_path = os.path.abspath(os.path.join(current_dir, '../'))


# # Add the src directory to sys.path
# sys.path.append(root_path)


from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

conf_path = str(Path('..') / settings.CONF_SOURCE)
conf_loader = OmegaConfigLoader(conf_source=conf_path)
param = conf_loader["parameters"]


st.write('Default parameters in the parameter.yml')
st.json(param)



existent_pipelines = ["ingestion",
                      "split_data",
                      "data_unit_tests",
                      "preprocess_train",
                      "split_train",
                      "model_selection",
                      "model_train",
                      "feature_selection",
                      "production_full_train_process",
                      "preprocess_batch",
                      "production_full_prediction_process"
                      ]

if 'button_model' not in st.session_state:
    st.session_state.button_model = False

def button_model():
    st.session_state.button_model = True 

def stateful_button(*args, key=None, **kwargs):
    if key is None:
        raise ValueError("Must pass key")

    if key not in st.session_state:
        st.session_state[key] = False

    if st.button(*args, **kwargs):
        st.session_state[key] = not st.session_state[key]

    return st.session_state[key]    

# Initialization
if 'use_case' not in st.session_state:
    st.session_state['use_case'] = ''


choice = st.radio('**Available Pipelines:**',existent_pipelines, index= 0)

if choice != '':
    if stateful_button(f'**Run Pipeline**', key=f'**Run Pipeline**'):

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"

        #st.header("Left: Std Out*********************Right: Std Err")
        st.header("Terminal display")
        stdout, stderr = st.columns(2)

        with redirect_stdout(io.StringIO()) as stdout_f, redirect_stderr(io.StringIO()) as stderr_f:
            try:
                cmd = [
                sys.executable,   
                "-X", "utf8",     # enable full UTF‑8 mode
                "-m", "kedro",    # run kedro as a module
                "run",
                f"--pipeline={choice}"
                ]
                good_process = subprocess.run(cmd, cwd="../",encoding="utf-8", capture_output=True, text=True)
                stdout_f.write(good_process.stdout)
                stderr_f.write(good_process.stderr)
            except Exception as e:
                traceback.print_exc()
                traceback.print_exc(file=sys.stdout) # or sys.stdout

        stdout_text = stdout_f.getvalue()
        stdout.text(stdout_text)
        stderr_text = stderr_f.getvalue()
        stderr.text(stderr_text)

        if choice == "data_unit_tests":
            df_validation = pd.read_csv("..//data//08_reporting//data_tests.csv")
            st.write(df_validation)
           
        elif choice == "model_train":
            st.image("..//data//08_reporting//shap_plot.png", caption='Systemic Explainability')
