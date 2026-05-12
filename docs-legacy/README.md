# docs-legacy

Archive of pre-2026 BrewSLM documentation assets. **The current docs site is at `slm-docs/`** (Docusaurus, served on port 3001 in dev). Everything here is kept for historical reference only.

## What's in here

| File | Notes |
|---|---|
| `slm_product_guide_v1.pdf` | Original product-guide PDF. TOC stops at "Export & Deployment" + an "Architecture Reference" appendix; predates Autopilot v3 (Wave A), Wave F deployment, Wave G observability, and Wave H extensions. Screenshots in this PDF do **not** match the current UI. |
| `gen_pdf.py` | Script that generated the old PDF. Kept for reference if you want to compare old vs new layout decisions. |

## Where to find the current docs

- **Local dev**: `cd slm-docs && npm run start` → http://localhost:3001/
- **In-app**: the `?` icon in the top-right TopBar links to this site.
- **Source**: `slm-docs/docs/` (Markdown). Sidebar is `slm-docs/sidebars.ts`.

## Why we kept the old PDF

- Captures what the product looked like at v1.0 / 2026-edition cut.
- Useful when reviewing what assumptions / abstractions changed between waves.
- Mostly: it's small, costs nothing to keep, and someone might ask for it.

If you need to refer to it for a customer hand-off, **always send the current Docusaurus site** instead. The PDF is out of date.
