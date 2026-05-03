# Deployment

ValorPredict is a Streamlit application. The easiest public deployment path is Streamlit Community Cloud because it can run `app.py` directly from GitHub.

## Streamlit Community Cloud

1. Open Streamlit Community Cloud.
2. Connect the GitHub repository: `Ayush141910/valorpredict`.
3. Select branch: `main`.
4. Main file path: `app.py`.
5. Python version: `3.12`.
6. Deploy.

The repo includes committed model artifacts and curated data extracts, so the app does not need to retrain during deployment.

## Render

The repo includes a `Procfile`:

```text
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

Use a Python web service with:

- Build command: `pip install -r requirements.txt`
- Start command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`

## Docker

The included `Dockerfile` can run the app in any container host:

```bash
docker build -t valorpredict .
docker run -p 8501:8501 valorpredict
```

Then open `http://localhost:8501`.
