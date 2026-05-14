# Deploying `website/` to brewslm.com

The local `website/` directory in this repo is the **source of truth** for
[brewslm.com](https://brewslm.com), which is served by GitHub Pages from a
separate repo: [`TensorGreed/__SLM__website`](https://github.com/TensorGreed/__SLM__website).

The deploy is a sync + push. Two repos, two pushes.

## One-time setup

Clone the public website repo next to this one:

```sh
cd ~/Desktop/GitHub  # or wherever this repo lives
git clone https://github.com/TensorGreed/__SLM__website.git
```

## Each deploy

From the root of THIS repo:

```sh
./website/sync-to-public.sh ../​__SLM__website
```

(The path argument is the local clone of `TensorGreed/__SLM__website`. Adjust
if you cloned elsewhere.)

The script will:

1. `rsync` every file under `website/` into the public repo's working tree,
   **excluding** this `DEPLOY.md` and the `sync-to-public.sh` script itself
   (those are repo-internal).
2. Delete files in the public repo that no longer exist locally (the 10
   removed SEO blog posts, for example).
3. Print a `git status` of the public repo so you can eyeball what changed.

Then, from the public repo:

```sh
cd ../​__SLM__website
git diff               # review the changes
git add -A
git commit -m "Sync website from __SLM__ main"
git push origin main
```

GitHub Pages picks up the push and redeploys in ~60 seconds. The CNAME is
already configured at `brewslm.com`; no DNS work needed.

## What gets deployed

- `index.html`, `creation-paths.html`, `workflow.html`, `capabilities.html`,
  `blog.html`, `faqs.html` — the five main pages.
- 3 blog posts in the engineering-blog rewrite
  (`blog-schema-introspector-vs-hand-written-converters.html`,
  `blog-task-aware-eval-handler-dispatcher.html`,
  `blog-gamifying-a-dev-tool.html`).
- `styles.css`, `script.js`, `assets/` — the shared chrome + brand assets.
- `sitemap.xml`, `robots.txt` — SEO basics.
- `.nojekyll` (top of public repo only — not present locally), `CNAME` (ditto).

## What does NOT get deployed

- `DEPLOY.md` (this file).
- `sync-to-public.sh` (the sync script).
- Anything outside `website/` in this repo.

## Updating going forward

Edit the files in `website/` as normal in the BrewSLM main repo, commit them
with the rest of your changes, then run the sync + push when you want them
live. The two commits (one in BrewSLM main, one in `__SLM__website`) are
separate by design — the public repo's history is exactly what's been
published.
