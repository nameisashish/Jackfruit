# Jackfruit

Streamlit app for detecting jackfruits with a YOLO model hosted on Hugging Face.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

The ML detector depends on patched PyTorch wheels. Use Python 3.12 on Streamlit
Community Cloud to enable YOLO inference. If the app is deployed on Python 3.13,
the UI will still install and load, but model inference will be disabled because
PyTorch wheels are not available there yet.

## Verify

```bash
python -m pip install -r requirements-dev.txt
pytest -q
bandit -q -c pyproject.toml -r .
pip-audit -r requirements.txt
```
