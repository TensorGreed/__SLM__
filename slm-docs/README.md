# BrewSLM Documentation (Docusaurus)

This directory contains the beginner-focused documentation portal for BrewSLM, built with Docusaurus.

## Run Locally

```bash
cd slm-docs
npm install
npm start
```

Docs are served at: `http://localhost:3001/`

## Build Static Site

```bash
cd slm-docs
npm run build
npm run serve
```

## Frontend Help Button Link

The frontend Help icon opens the URL from `VITE_DOCS_URL`.

If not set, it defaults to:

`http://localhost:3001/docs/getting-started/quickstart`

To override in frontend:

```bash
# frontend/.env.local
VITE_DOCS_URL=https://your-docs-host/docs/getting-started/quickstart
```
