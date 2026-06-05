# GitHub Wiki setup (one-time)

Wiki content is maintained in the [`wiki/`](../wiki/) directory and published by the **Publish Wiki** GitHub Action.

GitHub does **not** create the `.wiki` git repository until the first page is created in the UI. After that one-time step, every push to `main` that changes `wiki/` will sync automatically.

## One-time: enable the wiki backend

1. Open https://github.com/TheMitchyBoy/GEX/wiki  
2. Click **Create the first page** (or **New Page** if empty).  
3. Set the title to **Home**, add any short placeholder text, and click **Save Page**.  
4. You can delete the placeholder body afterward — the next Action run will overwrite it from `wiki/Home.md`.

## Automatic publishing

Workflow: [`.github/workflows/publish-wiki.yml`](../.github/workflows/publish-wiki.yml)

- Triggers on pushes to `main` under `wiki/`  
- Force-pushes `wiki/` to `https://github.com/TheMitchyBoy/GEX.wiki.git`  

Manual trigger: **Actions → Publish Wiki → Run workflow**

## Edit wiki locally (optional)

After step 1 above:

```bash
git clone https://github.com/TheMitchyBoy/GEX.wiki.git
cd GEX.wiki
# edit pages, then:
git add -A && git commit -m "Update wiki" && git push
```

Prefer editing `wiki/` in the main repo so changes go through review and CI.

## Wiki index

| Page | File |
|------|------|
| Home | `wiki/Home.md` |
| What is GEX? | `wiki/What-is-GEX.md` |
| Architecture | `wiki/Architecture.md` |
| Getting Started | `wiki/Getting-Started.md` |
| Configuration | `wiki/Configuration.md` |
| Forecasting | `wiki/Forecasting.md` |
| Dashboard & API | `wiki/Dashboard-and-API.md` |
| Live Flow | `wiki/Live-Flow.md` |
| Data Quality | `wiki/Data-Quality.md` |
| Operations | `wiki/Operations.md` |
| Roadmap | `wiki/Roadmap.md` |

Sidebar: `wiki/_Sidebar.md`
