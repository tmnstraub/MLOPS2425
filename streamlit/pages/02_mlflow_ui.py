# streamlit libray
import streamlit as st
import streamlit.components.v1 as components
from streamlit_ydata_profiling import st_profile_report
from pathlib import Path
from kedro.framework.project import configure_project
import time
import requests

package_name = Path(__file__).parent.name
configure_project(package_name)

KEDRO_VIZ_SERVER_URL = 'http://127.0.0.1:4141/'
MLFLOW_SERVER_URL = 'http://127.0.0.1:8080/'

if 'kedro_viz_started' not in st.session_state:
    st.session_state['kedro_viz_started'] = False


if 'mlflow' not in st.session_state:
    st.session_state['mlflow'] = False


def launch_mlflow_server(reporter):

    if not st.session_state['mlflow']:
        import os
        import subprocess
        import threading

        def _run_job(job):
            print (f"\nRunning job: {job}\n")
            proc = subprocess.Popen(job)
            proc.wait()
            return proc

        #job = ["kedro viz"]

        reporter.warning('Starting visualization server...')
        time.sleep(3)
        # server thread will remain active as long as streamlit thread is running, or is manually shutdown
        good_process = subprocess.run(["mlflow", "server", "--host=127.0.0.1", "--port=8080","--artifacts-destination=./ml_artifacts", "--default-artifact-root=./ml_artifacts"], cwd="C://Users//rosan//OneDrive//Desktop//MLOps_2025//MLOps_project//04_week", capture_output=True, text=True)
        good_process.kill()
        reporter.info('Waiting for server response...')
        time.sleep(3)

        retries = 5
        while True:
            reporter.info('Waiting for server response...')
            # give it time to start
            resp = None
            try:
                resp = requests.get(MLFLOW_SERVER_URL)
            except:
                pass
            if resp and resp.status_code == 200:
                st.session_state['mlflow'] = True
                reporter.empty()
                break
            else:
                time.sleep(1)
            retries -= 1
            if retries < 0:
                reporter.info('Right click on the empty iframe and select "Reload frame"')
                break

def show_mlflow():
    # Render the pipeline graph (cool demo here: https://demo.kedro.org/)
    st.subheader('MLFOW UI')
    
    reporter = st.empty()
    
    st.write("Visualization available at http://localhost:8080/")

    launch_mlflow_server(reporter)
    
    if st.session_state['mlflow']:
        st.caption(f'This interactive pipeline visualization.')
        components.iframe(MLFLOW_SERVER_URL, width=1500, height=800)
   
show_mlflow()