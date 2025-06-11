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

# KEDRO VIZ SERVER

def launch_kedro_viz_server(reporter):

    if not st.session_state['kedro_viz_started']:
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
        good_process = subprocess.run(["kedro", "viz","--no-browser"], cwd="../", capture_output=True, text=True)
        # good_process.start()
        #thread = threading.Thread(name='Kedro-Viz', target=_run_job, args=(job,), daemon=True,)
        #thread.start()
        reporter.info('Waiting for server response...')
        time.sleep(3)

        retries = 5
        while True:
            reporter.info('Waiting for server response...')
            # give it time to start
            resp = None
            try:
                resp = requests.get(KEDRO_VIZ_SERVER_URL)
            except:
                pass
            if resp and resp.status_code == 200:
                st.session_state['kedro_viz_started'] = True
                reporter.empty()
                break
            else:
                time.sleep(1)
            retries -= 1
            if retries < 0:
                reporter.info('Right click on the empty iframe and select "Reload frame"')
                break

def show_pipeline_viz():
    # Render the pipeline graph (cool demo here: https://demo.kedro.org/)
    st.subheader('KEDRO PIPELINE VISUALIZATION')
    
    reporter = st.empty()
    
    st.write("Visualization available at http://localhost:4141/")

    launch_kedro_viz_server(reporter)
    
    if st.session_state['kedro_viz_started']:
        st.caption(f'This interactive pipeline visualization.')
        components.iframe(KEDRO_VIZ_SERVER_URL, width=1500, height=800)
   
show_pipeline_viz()
