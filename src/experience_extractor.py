# src/experience_extractor.py
import re
from datetime import datetime

MONTHS = {
    'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
    'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12
}

_year_re = re.compile(r'(19|20)\d{2}')
_range_patterns = [
    # Examples: Jan 2018 - Mar 2020 or January 2018 - 2020
    re.compile(r'([A-Za-z]{3,9})\s*(19|20)\d{2}\s*[-–—]\s*([A-Za-z]{3,9})?\s*(19|20)\d{2}'),
    # 2018 - 2020
    re.compile(r'((?:19|20)\d{2})\s*[-–—]\s*((?:19|20)\d{2})'),
    # Jan 2018 to Present
    re.compile(r'([A-Za-z]{3,9})\s*(19|20)\d{2}\s*(?:to|until|-)\s*(present|now|current)', re.I)
]

def _parse_month_year(month_str: str, year_str: str):
    month = 1
    if month_str:
        m = month_str.strip()[:3].lower()
        month = MONTHS.get(m, 1)
    year = int(year_str)
    return datetime(year, month, 1)

def estimate_experience_years(text: str):
    """
    Extracts date ranges and returns approximate total years (float).
    Heuristic-based: finds explicit ranges; if none, uses earliest and latest year seen.
    """
    text = (text or "").lower()
    now = datetime.now()
    ranges = []

    # pattern matches
    for pat in _range_patterns:
        for m in pat.finditer(text):
            try:
                groups = m.groups()
                if len(groups) >= 2 and groups[0] and groups[1]:
                    # handle different groups depending on pattern
                    if pat is _range_patterns[0]:
                        start_month = groups[0]
                        start_year = groups[1]
                        end_month = groups[2] if groups[2] else None
                        end_year = groups[3]
                        start_dt = _parse_month_year(start_month, start_year)
                        end_dt = _parse_month_year(end_month or 'jan', end_year)
                    elif pat is _range_patterns[1]:
                        start_dt = _parse_month_year('jan', groups[0])
                        end_dt = _parse_month_year('dec', groups[1])
                    else:
                        start_month = groups[0]
                        start_year = groups[1]
                        start_dt = _parse_month_year(start_month, start_year)
                        end_dt = now
                    ranges.append((start_dt, end_dt))
            except Exception:
                continue

    # fallback: if no explicit ranges, use years found
    if not ranges:
        years = [int(y.group(0)) for y in _year_re.finditer(text)]
        if len(years) >= 2:
            start = min(years)
            end = max(years)
            ranges.append((datetime(start,1,1), datetime(end,12,31)))
        elif len(years) == 1:
            # single year -> assume 1 year experience
            return 1.0

    # merge ranges (simple)
    total_months = 0
    for s,e in ranges:
        if e < s:
            continue
        months = (e.year - s.year) * 12 + (e.month - s.month) + 1
        total_months += months

    years = round(total_months / 12.0, 2)
    return years
