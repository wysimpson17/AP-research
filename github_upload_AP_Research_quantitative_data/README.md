# Houston AI Job Trends

Project label: `AP Research quantitative data`

This project collects current job postings from public Houston-employer career boards, keeps only Houston, Texas roles, flags AI and machine-learning-related postings, and writes CSV outputs you can analyze in Python, Excel, or PyCharm.

## What it captures

- Posting title
- Company name
- Date posted
- Description text
- Location
- Department
- AI-related keyword counts

## Current employers

The starter configuration lives in `houston_employers.json`. It currently uses public Workday feeds for:

- Axiom Space
- PROS
- Sysco
- ConocoPhillips

You can add more Houston employers by appending entries to `houston_employers.json`.

## Outputs

Running `main.py` writes these files into `outputs/`:

- `current_postings.csv`: Houston postings collected in the current run
- `current_ai_related_postings.csv`: Houston postings that match at least one AI/ML keyword
- `job_postings_history.csv`: deduplicated Houston daily history across repeated runs
- `yearly_keyword_counts.csv`: Houston yearly counts grouped by posting year
- `snapshot_keyword_counts.csv`: Houston run-by-run counts grouped by collection date

The default location filter is `Houston, Texas`. Year filtering is optional with `--start-year` and `--end-year`.
If there are no Houston matches during a run, the CSVs will still be created with headers so they stay easy to use in pandas or Excel.

## Historical Trend Proxy

Running `archive_trend.py` builds an approximate `2005-2025` Houston trend from archived Houston job and career pages in the Wayback Machine. It writes:

- `archive_source_yearly_samples.csv`: one sampled archived page per source per year, with AI-term counts
- `houston_archive_yearly_trend.csv`: yearly Houston archive trend summary and a normalized trend index

This archive-based trend is a lower-confidence proxy, not a complete historical count of every Houston posting. It works best as a visible directional signal.

## Why there are two trend files

Most public job board APIs expose current openings, not a full historical archive. Because of that:

- `yearly_keyword_counts.csv` groups currently observed Houston postings by the year they were posted
- `snapshot_keyword_counts.csv` becomes more useful over time as you rerun the script and build your own history

If you want a stronger long-term trend dataset, rerun the script regularly.

## Run it

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
python main.py --location "Houston, Texas" --start-year 2005 --end-year 2025
python archive_trend.py --start-year 2005 --end-year 2025
```

## Keywords tracked

- AI / artificial intelligence
- machine learning
- data science / data scientist
- quantitative research / quant researcher
- ML engineer / machine learning engineer
- deep learning
- Python
- statistical modeling
