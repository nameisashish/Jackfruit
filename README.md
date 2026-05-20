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

The ML detector depends on patched PyTorch wheels. Current secure PyTorch wheels
are not published for Intel macOS, so Intel macOS installs still load the app UI
but should use the devcontainer or a Linux/Python 3.11 deployment for model
inference.

## Verify

```bash
python -m pip install -r requirements-dev.txt
pytest -q
bandit -q -c pyproject.toml -r .
pip-audit -r requirements.txt
```
