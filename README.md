# Excel Q&A Assistant

Ask plain-English questions about an uploaded Excel/CSV/PDF file, powered
by LangChain's pandas dataframe agent and Google Gemini (via its free
API tier).

The app uses a single Google API key configured by whoever runs/deploys
it (you) — visitors do not enter their own key or pick a model. Every
question anyone asks against the deployed app is billed against that
one key's quota, so keep this in mind before sharing a public link. The
Gemini free tier has rate limits (requests per minute/day); if you hit
them, wait a bit or upgrade to a paid Google AI plan.

## Get a free API key

1. Go to https://aistudio.google.com/apikey.
2. Sign in with a Google account and click "Create API key".
3. Copy the key — this is your `GOOGLE_API_KEY`.

## Run locally

```bash
pip install -r requirements.txt
export GOOGLE_API_KEY="your-key-here"   # Windows PowerShell: $env:GOOGLE_API_KEY="your-key-here"
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501),
upload a spreadsheet, and ask a question. If `GOOGLE_API_KEY` isn't set,
the app shows an error when you try to ask a question instead of
crashing.

## Deploy for free on Streamlit Community Cloud

1. Push this folder to a GitHub repository (must include `app.py` and
   `requirements.txt`).
2. Go to https://share.streamlit.io, sign in, and click "New app".
3. Pick the repo/branch and set the main file path to `app.py`.
4. Before (or after) deploying, go to **App settings -> Secrets** and add:

   ```toml
   GOOGLE_API_KEY = "your-key-here"
   ```

5. Click "Deploy" (or "Reboot" if you added the secret after deploying).

## Deploy for free on Hugging Face Spaces

1. Create a new Space at https://huggingface.co/new-space.
2. Choose the **Streamlit** SDK.
3. Upload/push `app.py`, `requirements.txt`, and this `README.md` to the
   Space repo (or `git push` if you cloned the Space).
4. Go to **Settings -> Repository secrets** on the Space and add:
   - Name: `GOOGLE_API_KEY`
   - Value: your Google API key
5. The Space will build automatically and expose the app at
   `https://huggingface.co/spaces/<your-username>/<space-name>`.

## Notes

- `allow_dangerous_code=True` lets the LangChain agent execute Python code
  it writes against your dataframe. Only run this with data/files you
  trust, since arbitrary code execution is inherently risky.
- There is no model picker — every question is answered with
  `gemini-3.6-flash` (set via `DEFAULT_MODEL` in `app.py`). Change that
  constant if you want a different model or run into free-tier rate
  limits.
- Since one API key serves every visitor, consider adding your own access
  control (e.g. a Streamlit Community Cloud private app, or an
  `st.text_input` app-wide passcode check) before sharing the link
  publicly, to avoid unexpectedly burning through the free-tier quota.
- The uploader also accepts PDFs. Tables are pulled out with `pdfplumber`,
  which works best on PDFs with a clean, gridded table layout or clearly
  aligned text columns; scanned images (no real text layer) won't extract.
  If a PDF has multiple tables/pages, pick which one to analyze from the
  sidebar, same as picking an Excel sheet.
