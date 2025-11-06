# Environment Variables Setup

This project uses environment variables to securely manage API keys and configuration. 

## Setup Instructions

1. Copy the example environment file:
   ```bash
   cp backend/.env.example backend/.env
   ```

2. Edit `backend/.env` and add your API keys:
   - For **Hugging Face**: Get your token from https://huggingface.co/settings/tokens
   - For **Google Gemini**: Get your API key from https://aistudio.google.com/app/apikey

3. Choose your LLM provider by setting `LLM_PROVIDER` to one of:
   - `ollama` (default, runs locally)
   - `hf` or `huggingface` (requires HUGGINGFACE_TOKEN)
   - `gemini` (requires GEMINI_API_KEY)

## Security

- The `.env` file is already included in `.gitignore` to prevent accidental commits
- Never commit API keys to version control
- Use the `.env.example` file as a template for new deployments

## Current Configuration

Your current setup uses:
- **LLM_PROVIDER**: ollama (local, no API key required)
- **OLLAMA_MODEL**: llama3.2:3b
- **OLLAMA_HOST**: http://127.0.0.1:11434