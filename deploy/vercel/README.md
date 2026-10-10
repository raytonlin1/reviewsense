# Star-rating API on Vercel

The Part 14 INT8 star model as a small FastAPI app: only FastAPI, ONNX Runtime and the tokenizer library (~180 MB
installed plus the 65 MB model, under Vercel's 500 MB limit for Python). The full ReviewSense app (chatbot, search,
speech) needs more memory than Vercel functions have (2-4 GB); see the main README.

## Deploy
From the repository root:
```bash
python -m reviewsense.serving.onnx_model export      # only if artifacts/stars-onnx doesn't exist yet
cp artifacts/stars-onnx/model-int8.onnx artifacts/stars-onnx/tokenizer.json deploy/vercel/model/
cd deploy/vercel
vercel login
vercel              # preview deployment: prints a URL
vercel --prod       # production
```
The model files are not in git (65 MB); the copy step puts them in this folder, and the CLI uploads them.
`vercel.json` sets the framework to FastAPI. Without it, a project created as "Other" deploys an empty static site:
the build takes a few seconds, no Python function is created, and every URL returns 404 NOT_FOUND.

If `vercel` fails with **Request Entity Too Large**, it is uploading the whole repository (models: ~12 GB; the Hobby
plan allows 100 MB per CLI upload). That happens when the CLI links the git repository instead of this folder
(a `.vercel/repo.json` at the repository root). Remove that link and link this folder only:
```bash
rm -rf ../../.vercel                                  # the repository-level link (local files only)
vercel link --project reviewsense-stars --yes         # run inside deploy/vercel
vercel --prod
```

## Use
```bash
curl -X POST https://YOUR-PROJECT.vercel.app/stars -H "Content-Type: application/json" \
     -d '{"texts": ["Best tacos in town!", "Cold pizza, rude manager."]}'
```
Interactive docs: `https://YOUR-PROJECT.vercel.app/docs`
