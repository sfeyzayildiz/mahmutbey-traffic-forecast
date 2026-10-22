from pathlib import Path

from streamlit.testing.v1 import AppTest

APP=Path(__file__).resolve().parents[1]/'app'/'app.py'


def test_demo_app_runs():
    at=AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    assert len(at.metric)==3
