# AI Newsroom

AI Newsroom is an open-source educational project that demonstrates a small,
auditable workflow for discovering technology news and producing grounded
social-media drafts. Version 0.1 fetches RSS/Atom feeds, ranks stories with
deterministic rules, researches one candidate, verifies its claims, applies a
deterministic editorial gate, and prepares drafts for human review.

It does not publish content, operate social-media accounts, use autonomous web
search, or run an open-ended agent loop.

## Architecture

```text
RSS/Atom feeds
      |
      v
News Scout: fetch and normalize
      |
      v
Deterministic ranking: deduplicate, filter, score
      |
      v
Researcher: extract grounded claims from one retrieved article
      |
      v
Fact Checker: verify each claim against supplied article evidence
      |
      v
Deterministic editorial gate
      |
      v
Writer: produce LinkedIn and X drafts from verified facts only
      |
      v
Human review (no publishing integration)
```

The Researcher, Fact Checker, and Writer share a provider abstraction. The
current implementation uses GitHub Models, while the workflow logic remains
independent of that provider.

## Prerequisites

- Python 3.12 or newer
- Git
- Network access to configured RSS feeds and article pages
- A GitHub account with access to GitHub Models
- A GitHub token with `models: read` permission for local model calls

## Python setup

Create an isolated environment and install the project with its runtime dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --editable .
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Environment variables

Copy `.env.example` if you want a local reference:

```bash
cp .env.example .env
```

The project does not automatically load `.env`; export variables in the shell
or use your development environment's secret injection:

```bash
export GITHUB_TOKEN="your-token"
export AI_MODEL="openai/gpt-4.1-mini"
export RESEARCH_MAX_ARTICLE_CHARS="12000"
```

| Variable | Required | Purpose |
| --- | --- | --- |
| `GITHUB_TOKEN` | For model stages | GitHub Models authentication; use a token with `models: read`. |
| `AI_MODEL` | No | Model identifier. Defaults to `openai/gpt-4.1-mini`. |
| `RESEARCH_MAX_ARTICLE_CHARS` | No | Maximum grounded article characters sent to a model. Defaults to `12000`. |

Never commit a real token. `.env` and its variants are ignored by Git.

## Configure news sources

Edit `config/sources.yaml`. Version 0.1 supports sources with `type: rss`;
`feedparser` handles both RSS and Atom documents:

```yaml
sources:
  - name: Publisher name
    url: null  # Replace with the publisher's real RSS/Atom feed URL.
    type: rss
    enabled: true
```

Use a real feed URL supplied by the publisher. Disable a source with
`enabled: false`. One unavailable or malformed feed does not stop other
configured sources.

## Configure editorial topics

Edit `config/editorial.yaml` to control:

- Topic names and keyword variants
- Title and summary match weights
- Recency window and recency bonus
- Minimum relevance score
- Source-priority bonuses

Ranking is deterministic and happens before any model call.

## Run locally

Run only News Scout and deterministic ranking:

```bash
python src/main.py
```

Run all unit tests without making live model calls:

```bash
python -m unittest discover -s tests -v
```

Run one existing candidate through the Researcher:

```bash
python src/research_one.py
```

Fact-check one researched dossier:

```bash
python src/fact_check_one.py --story-id story_example
```

Write one fact-checked dossier:

```bash
python src/write_one.py --story-id story_example
```

## Run one end-to-end pipeline

After exporting `GITHUB_TOKEN`, run:

```bash
python src/run_pipeline.py
```

The pipeline processes only the highest-ranked candidate:

```text
fetch -> rank -> research -> fact-check -> editorial gate -> draft
```

A successful eligible story uses one model call per model-backed stage:
Researcher, Fact Checker, and Writer. The command never publishes its output.

## GitHub Actions

`.github/workflows/newsroom.yml` runs on Python 3.12 and:

1. Installs dependencies.
2. Runs the full unit-test suite.
3. Runs the existing one-candidate pipeline.

Start it manually from **Actions > AI Newsroom > Run workflow**.

A daily schedule is present at 06:00 UTC but is disabled by default. To enable
scheduled pipeline runs, create a repository Actions variable:

```text
ENABLE_DAILY_PIPELINE=true
```

Optional Actions variables:

```text
AI_MODEL=openai/gpt-4.1-mini
RESEARCH_MAX_ARTICLE_CHARS=12000
```

The workflow maps GitHub's automatically generated `secrets.GITHUB_TOKEN` to
the application's `GITHUB_TOKEN` environment variable. Workflow permissions
are limited to `contents: read` and `models: read`. No publishing step exists.

## Security notes

- Tokens are read only from the environment, stripped safely, and never logged.
- Provider diagnostics expose only token presence and length.
- Retrieved article content and model output are treated as untrusted input.
- Researcher and Fact Checker URLs must match URLs already supplied in dossiers.
- Fact Checker results are validated before being stored.
- The editorial gate is deterministic; blocked stories do not call the Writer.
- Writer input contains verified facts, not raw article HTML.
- Every model response is schema-validated.
- No stage has shell, publishing, database, or autonomous browsing authority.

## Generated data

Runtime data is local and intentionally ignored:

```text
data/news.json
data/candidates.json
data/research/*.json
data/output/*.json
```

Tracked `.gitkeep` files preserve the empty directories. GitHub Actions does
not commit or upload generated results. A synthetic draft shape is available
at `examples/output.example.json`; it contains no real news or research data.

## Current limitations

- RSS/Atom is the only source-ingestion type.
- The pipeline processes one candidate by default.
- Research and fact-checking use only the candidate's retrieved article.
- There is no independent web search or multi-source corroboration.
- Article extraction is intentionally lightweight, not a full scraper.
- Source trust tiers are configured but not yet used for automatic decisions.
- Model judgments can be wrong and all drafts require human review.
- No user interface, scheduler service, database, authentication, or publishing
  integration is included.

## Roadmap

Potential post-v0.1 work:

- Multi-source research and corroboration
- Human approval interfaces and audit history
- Stronger article extraction and provenance capture
- Additional model providers through the existing abstraction
- Configurable batch sizes and quota controls
- Optional draft export integrations after explicit human approval
- Evaluation datasets for grounding and fact-check quality

Publishing remains deliberately out of scope until approval, audit, and safety
controls are designed.
