# CampaignLens — GitHub & Colab Workflow

## Setup (once per person)

1. Get added as a collaborator — ask Majid to add your GitHub username via **Settings → Collaborators**
2. Open [colab.research.google.com](https://colab.research.google.com)
3. Go to **File → Open notebook → GitHub tab**
4. Search for `CampaignLens` and open your notebook

---

## Daily Workflow

1. **Before starting** — always open the latest version from GitHub: **File → Open notebook → GitHub**
2. Do your work
3. **When done** — save back: **File → Save a copy in GitHub**
4. Pick the `CampaignLens` repo, choose the right branch, and write a clear commit message

---

## Commit Message Rules

- Describe what you actually did
- Good: `added sentiment classifier`, `fixed scraper pagination`, `cleaned preprocessing step`
- Bad: `update`, `changes`, `fix`

---

## Team Rules

- One notebook per person or per task — don't edit the same file at the same time
- Always open from GitHub before starting, never work from a saved local copy
- Commit after every working session, not just at the end of the project (the professor checks commit history)

---

## Repo Structure

```
CampaignLens/
├── data_collection/
│   └── scraper.ipynb
├── preprocessing/
│   └── preprocessing.ipynb
├── pipeline/
│   └── pipeline.ipynb
├── tools/
│   ├── chatbot.ipynb
│   └── dashboard.ipynb
├── data/
│   └── comments.csv
└── README.md
```

---

## If You Have a Conflict

If Colab says the file already exists or there's a version conflict:

1. Rename your current notebook with today's date e.g. `scraper_june1`
2. Save it to GitHub
3. Tell the team so someone can merge it manually

---

## Quick Reference

| Action | How |
|---|---|
| Open from GitHub | File → Open notebook → GitHub |
| Save to GitHub | File → Save a copy in GitHub |
| See commit history | github.com/MajidSharaf/CampaignLens → commits |
| Add a teammate | Settings → Collaborators → Add people |
