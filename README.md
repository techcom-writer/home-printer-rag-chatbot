# Home Printer Documentation RAG Chatbot

AI-powered chatbot for your documentation using free cloud services.

## 🚀 Quick Start

See [../RAG_QUICKSTART.md](../RAG_QUICKSTART.md) for 5-step setup (30 minutes).

## 📁 Files

### Application
- **`app.py`** - Streamlit chatbot application (the main app file)
- **`ingest.py`** - Document ingestion script (run once to populate vector DB)

### Configuration
- **`.env.example`** - Environment variables template (copy to `.env`)
- **`streamlit_secrets.toml.example`** - Streamlit Cloud secrets template
- **`requirements.txt`** - Python dependencies

## 🔄 Workflow

### 1. Local Development

```bash
# Setup
cp .env.example .env
# Edit .env with your API keys

pip install -r requirements.txt

# Ingest documents to Pinecone (one-time)
python ingest.py

# Run app locally
streamlit run app.py
```

### 2. Deploy to Streamlit Cloud

```bash
git add .
git commit -m "Add chatbot"
git push origin main
```

Then visit [share.streamlit.io](https://share.streamlit.io) and deploy.

## 📊 Architecture

```
User Question
    ↓
Streamlit App (app.py)
    ↓
Embedding (Hugging Face)
    ↓
Vector Search (Pinecone)
    ↓
Retrieve Context
    ↓
LLM Call (Groq)
    ↓
Generate Response
    ↓
Display to User
```

## 🛠️ Development

### Update Documents
1. Edit markdown files in `../docs/`
2. Re-run ingestion: `python ingest.py`
3. Restart app (Streamlit reloads automatically)

### Add Features
- Modify `app.py` for UI changes
- Modify `ingest.py` for embedding customization
- Update `requirements.txt` for new dependencies

### Test Locally
```bash
streamlit run app.py
# Visit http://localhost:8501
```

## 🔐 Security

- Never commit `.env` or secrets files
- Store API keys in environment variables locally
- Use Streamlit Secrets for cloud deployment
- Keep `.env` in `.gitignore` ✓

## 🆘 Troubleshooting

### "Module not found" errors
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Ingestion fails
- Check API keys in `.env`
- Verify Pinecone index exists
- Test with: `python ingest.py --dry-run`

### Streamlit app won't load
- Secrets not set? Go to Streamlit Settings → Secrets
- Missing API keys? Check all three are provided
- Rate limited? Wait and retry

## 📚 Documentation

- [RAG_QUICKSTART.md](../RAG_QUICKSTART.md) - 5-step setup
- [RAG_SETUP_GUIDE.md](../RAG_SETUP_GUIDE.md) - Complete guide
- [Streamlit Docs](https://docs.streamlit.io)
- [Groq API Docs](https://console.groq.com/docs)
- [Pinecone Docs](https://docs.pinecone.io)

## 📈 Monitoring

### Free Tier Usage
- Groq: [console.groq.com](https://console.groq.com) → Usage
- Pinecone: [app.pinecone.io](https://app.pinecone.io) → Metrics
- Streamlit: [share.streamlit.io](https://share.streamlit.io) → My app → Settings

### Check Ingestion Status
```bash
python ingest.py --dry-run
```

## 🎯 Next Steps

1. Get API keys (5 min)
2. Run `python ingest.py` (10 min)
3. Deploy to Streamlit Cloud (5 min)
4. Integrate into Docusaurus (5 min)
5. Test chatbot (5 min)

👉 See [RAG_QUICKSTART.md](../RAG_QUICKSTART.md) for detailed steps.

---

**Free-tier stack**: Groq + Pinecone + Hugging Face + Streamlit Cloud
**Cost**: $0/month
**Status**: Production-ready
