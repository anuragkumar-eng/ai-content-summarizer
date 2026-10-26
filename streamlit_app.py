"""
Streamlit Cloud entry point.
Forwards execution to app.py in the primary script context.
"""
import runpy

if __name__ == "__main__":
    runpy.run_path("app.py", run_name="__main__")
