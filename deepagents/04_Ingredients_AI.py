#!/usr/bin/env python3
"""
Price Intelligence / Renegotiation Agent

  STEP 1  Load ingredients from ../data/ingredients.txt
  STEP 2  Search (Google News RSS - free, no key)
  STEP 3  Scrape & clean source pages
  STEP 4  LLM analyst (Ollama locally, or OpenRouter)
  STEP 5  Renegotiation report (Markdown) + audit JSON

Usage:
  python 04_Ingredients_AI.py
  python 04_Ingredients_AI.py --backend openrouter --model openai/gpt-4o-mini -v

Env:
  OPENROUTER_API_KEY   (only if --backend openrouter)
  Optional: OPENROUTER_MODEL, OLLAMA_MODEL, MAX_WORKERS
"""

import argparse
import json
import logging
import os
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
INGREDIENTS_FILE = (BASE_DIR / ".." / "data" / "ingredients.txt").resolve()
OUT_DIR = (BASE_DIR / ".." / "reports").resolve()
CACHE_DIR = (BASE_DIR / ".." / "data" / "cache").resolve()

SOURCE_WINDOW_DAYS = 180      # max age of search results we consider
MAX_SOURCES = 4               # sources scraped + fed to the LLM per ingredient
JSON_RETRIES = 2              # attempts to get valid JSON out of the LLM

LLM_BACKENDS = ("ollama", "openrouter")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODELS = {
    "ollama": os.getenv("OLLAMA_MODEL") or "qwen2.5:7b-instruct",
    "openrouter": os.getenv("OPENROUTER_MODEL") or "nvidia/nemotron-3.5-super-vl:free",
}

VALID_TRENDS = {"UP", "DOWN", "STABLE", "UNKNOWN"}
VALID_CONFIDENCE = {"HIGH", "MEDIUM", "LOW"}
REQUIRED_KEYS = ("trend", "confidence", "specific_quotes", "summary")

log = logging.getLogger("price-agent")

_HTTP = requests.Session()
_HTTP.headers.update({
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
})

# ----------------------------------------------------------------------------
# STEP 1: Load Ingredients from File
# ----------------------------------------------------------------------------


def load_ingredients(filepath: Path) -> list[str]:
    """Load ingredient names from .txt (one per line or comma-separated) or .json."""
    content = filepath.read_text(encoding="utf-8").strip()

    if filepath.suffix.lower() == ".json":
        data = json.loads(content)
        if isinstance(data, dict) and "ingredients" in data:
            data = data["ingredients"]
        if isinstance(data, list):
            return [str(item).strip() for item in data if str(item).strip()]

    if "," in content and "\n" not in content:   # single comma-separated line
        return [part.strip() for part in content.split(",") if part.strip()]
    return [line.strip() for line in content.splitlines() if line.strip()]


# ----------------------------------------------------------------------------
# STEP 2: Web Search (Google News RSS - free, no API key)
# ----------------------------------------------------------------------------


def search_ingredient_price(ingredient: str) -> list[dict]:
    """Return list of {url, title, snippet, date} from Google News RSS."""
    since = (datetime.now(timezone.utc) - timedelta(days=SOURCE_WINDOW_DAYS)).strftime("%Y-%m")
    query = f"{ingredient} wholesale price trend since {since}"

    resp = _HTTP.get(
        "https://news.google.com/rss/search",
        params={"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"},
        timeout=15,
    )
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    results: list[dict] = []
    for item in root.iter("item"):
        raw_desc = item.findtext("description") or ""
        results.append({
            "url": (item.findtext("link") or "").strip(),
            "title": (item.findtext("title") or "").strip(),
            "snippet": BeautifulSoup(raw_desc, "html.parser").get_text(" ", strip=True),
            "date": item.findtext("pubDate"),
        })
        if len(results) >= 2 * MAX_SOURCES:   # headroom for de-duplication
            break
    return [r for r in results if r["url"]]


def parse_date(raw) -> datetime | None:
    """Parse RFC 822 (RSS pubDate) or ISO 8601 dates."""
    if not raw:
        return None
    text = str(raw).strip()
    try:
        return parsedate_to_datetime(text)
    except (TypeError, ValueError):
        pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def is_recent(published) -> bool:
    """True for recent items AND items whose date is missing/unparseable."""
    dt = parse_date(published)
    if dt is None:
        return True
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    cutoff = datetime.now(timezone.utc) - timedelta(days=SOURCE_WINDOW_DAYS)
    return dt >= cutoff


def _registrable_domain(url: str) -> str:
    """Coarse registrable domain: last two labels of the host."""
    host = urlparse(url).netloc.split(":")[0].lower()
    labels = [part for part in host.split(".") if part]
    return ".".join(labels[-2:]) if len(labels) >= 2 else host


def select_sources(results: list[dict], k: int = MAX_SOURCES) -> list[dict]:
    """Prefer recent results, de-duplicate URLs and publishers, keep up to k."""
    dated = [r for r in results if r.get("date") and is_recent(r["date"])]
    pool = dated if len(dated) >= k else results
    picked: list[dict] = []
    seen_urls: set[str] = set()
    seen_publishers: set[str] = set()
    for result in sorted(pool, key=lambda r: -int(is_recent(r.get("date")))):
        if result["url"] in seen_urls:
            continue
        # Google News wraps every article in a news.google.com redirect URL, so
        # all results share that domain - only de-dup real publishers.
        domain = _registrable_domain(result["url"])
        if domain != "news.google.com" and domain in seen_publishers:
            continue
        seen_publishers.add(domain)
        seen_urls.add(result["url"])
        picked.append(result)
        if len(picked) == k:
            break
    return picked


# ----------------------------------------------------------------------------
# STEP 3: Scrape and Clean Webpage Text
# ----------------------------------------------------------------------------


def fetch_page_text(url: str, max_chars: int = 3000) -> str | None:
    """Fetch and extract main text from a URL. None on failure."""
    try:
        resp = _HTTP.get(url, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside",
                         "form", "noscript"]):
            tag.decompose()
        text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
        return text[:max_chars] or None
    except Exception as exc:
        log.debug("Failed to fetch %s: %s", url, exc)
        return None


# ----------------------------------------------------------------------------
# STEP 4: LLM Analyst (Ollama locally, or OpenRouter)
# ----------------------------------------------------------------------------

ANALYST_SYSTEM_PROMPT = (
    "You are a meticulous commodity-pricing analyst. You only report prices and trends "
    "that are supported by the supplied context, and you answer strictly in the requested "
    "JSON format."
)


def call_llm(prompt: str, backend: str, model: str, json_mode: bool = True) -> str:
    messages = [
        {"role": "system", "content": ANALYST_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    if backend == "ollama":
        import ollama
        kwargs = {"model": model, "messages": messages, "options": {"temperature": 0}}
        if json_mode:
            kwargs["format"] = "json"
        return ollama.chat(**kwargs)["message"]["content"].strip()

    if backend == "openrouter":
        from openai import OpenAI
        client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"])
        if json_mode:
            try:
                completion = client.chat.completions.create(
                    model=model, messages=messages, temperature=0,
                    response_format={"type": "json_object"})
                return completion.choices[0].message.content.strip()
            except Exception as exc:
                # some OpenRouter models reject response_format; degrade gracefully
                log.debug("OpenRouter %s rejected json_object; retrying plain: %s", model, exc)
        completion = client.chat.completions.create(
            model=model, messages=messages, temperature=0)
        return completion.choices[0].message.content.strip()

    raise ValueError(f"Unknown LLM backend: {backend}")


def _validate_analysis(data) -> str | None:
    """Return an error description, or None if the analysis dict is well-formed."""
    if not isinstance(data, dict):
        return "not a JSON object"
    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        return f"missing keys: {', '.join(missing)}"
    if str(data["trend"]).strip().upper() not in VALID_TRENDS:
        return f"trend must be one of {sorted(VALID_TRENDS)}"
    if str(data["confidence"]).strip().upper() not in VALID_CONFIDENCE:
        return f"confidence must be one of {sorted(VALID_CONFIDENCE)}"
    if not isinstance(data["specific_quotes"], list):
        return "specific_quotes must be a list"
    return None


def _extract_json(raw: str) -> dict | None:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None


def analyze_price_trend(ingredient: str, results: list[dict],
                        backend: str, model: str) -> dict:
    """Scrape the selected sources and use the LLM to extract trend/quotes/summary."""
    selected = select_sources(results, k=MAX_SOURCES)

    # Scrape in parallel; fall back to the search snippet when a page blocks us
    with ThreadPoolExecutor(max_workers=min(3, max(1, len(selected)))) as pool:
        pages = list(pool.map(lambda res: fetch_page_text(res["url"]), selected))

    context_blocks: list[str] = []
    sources_meta: list[dict] = []
    for result, page in zip(selected, pages):
        content = page or (result.get("snippet") or "").strip()
        if not content:
            continue   # failed fetch AND no snippet -> don't pollute the LLM context
        note = "" if page else " (page fetch failed; using search snippet)"
        published = parse_date(result.get("date"))
        date_label = (published.strftime("%Y-%m-%d") if published
                      else str(result.get("date") or "unknown date"))
        context_blocks.append(
            f"--- SOURCE {len(context_blocks) + 1}{note}\n"
            f"Title: {result['title']}\n"
            f"URL: {result['url']}\n"
            f"Published: {date_label}\n"
            f"Content: {content[:1500]}"
        )
        sources_meta.append({"url": result["url"], "title": result["title"],
                             "date": date_label})

    if not context_blocks:
        return {"ingredient": ingredient, "trend": "UNKNOWN", "confidence": "LOW",
                "specific_quotes": [], "summary": "No usable search results or page content.",
                "sources": []}

    prompt = f"""You are a commodity pricing analyst preparing intelligence for a procurement team
before contract renegotiation. Below are web excerpts for "{ingredient}".

Using ONLY the information in the context above (never invent or guess prices), determine:
1. The overall price trend (UP, DOWN, STABLE, or UNKNOWN) over the last 3-6 months.
2. Specific price quotes found in the context, each with currency, unit, and date/date period.
3. A 2-3 sentence executive summary.

Rules:
- If the context contains no usable price information, return trend UNKNOWN, confidence LOW,
  an empty quote list, and a one-sentence explanation.
- Cite sources as SOURCE 1, SOURCE 2, ... inside the quotes.
- Quote format example: "215 USD per ton (Apr 2026, SOURCE 2)".

Return strictly this JSON (no markdown fences, no commentary):
{{
  "trend": "UP|DOWN|STABLE|UNKNOWN",
  "confidence": "HIGH|MEDIUM|LOW",
  "specific_quotes": ["quote with currency, unit and date", "..."],
  "summary": "your concise summary"
}}

Context ({len(context_blocks)} excerpt(s)):
{chr(10).join(context_blocks)}
"""

    raw = ""
    data = None
    last_error = None
    for attempt in range(1, JSON_RETRIES + 1):
        attempt_prompt = prompt
        if last_error:
            attempt_prompt += (f"\nYour previous reply was invalid ({last_error}). "
                               "Respond with ONLY the valid JSON object.")
        raw = call_llm(attempt_prompt, backend, model)
        candidate = _extract_json(raw)
        last_error = _validate_analysis(candidate)
        if last_error is None:
            data = candidate
            break
        log.debug("LLM JSON invalid for %r (attempt %d): %s", ingredient, attempt, last_error)

    if data is None:
        # Last resort: word-boundary heuristic so one bad parse doesn't kill the run
        log.warning("Falling back to heuristic trend extraction for %r", ingredient)
        match = re.search(r"\b(UP|DOWN|STABLE)\b", raw, re.IGNORECASE)
        data = {"trend": match.group(1).upper() if match else "UNKNOWN",
                "confidence": "LOW", "specific_quotes": [],
                "summary": raw.strip()[:300]}

    data["trend"] = str(data["trend"]).strip().upper()
    data["confidence"] = str(data["confidence"]).strip().upper()
    data["ingredient"] = ingredient
    data["sources"] = sources_meta
    return data


# ----------------------------------------------------------------------------
# STEP 5: Generate Renegotiation Report (Markdown)
# ----------------------------------------------------------------------------

LEVERAGE_GUIDANCE = {
    "DOWN": ("Strong leverage",
             "Prices are falling - demand a 5-10% reduction from current contracts, "
             "citing recent market drops."),
    "UP": ("Limited leverage",
           "Prices are rising - consider locking in long-term fixed pricing or "
           "exploring alternative suppliers."),
    "STABLE": ("Neutral",
               "Prices are stable - renegotiate on volume, service levels, and "
               "relationship rather than market pressure."),
    "UNKNOWN": ("Unknown",
                "No reliable market signal found - verify prices manually before negotiating."),
}


def generate_report(analyses: list[dict], meta: dict) -> str:
    counts: dict[str, int] = {}
    for analysis in analyses:
        trend = analysis.get("trend", "UNKNOWN")
        counts[trend] = counts.get(trend, 0) + 1

    lines = [
        "# Procurement Price Renegotiation Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"**Analyst:** {meta['backend']} ({meta['model']}) | "
        f"Search window: last {SOURCE_WINDOW_DAYS} days",
        "",
        "## Executive Summary",
        "",
        f"Market signal across {len(analyses)} ingredient(s): "
        + ", ".join(f"{value} {trend}" for trend, value in sorted(counts.items())) + ".",
        "",
        "| Ingredient | Trend | Confidence | Leverage |",
        "|---|---|---|---|",
    ]
    for analysis in analyses:
        short, _ = LEVERAGE_GUIDANCE.get(analysis.get("trend", "UNKNOWN"),
                                         LEVERAGE_GUIDANCE["UNKNOWN"])
        lines.append(
            f"| {analysis['ingredient']} | {analysis.get('trend', 'UNKNOWN')} "
            f"| {analysis.get('confidence', 'LOW')} | {short} |"
        )

    lines += ["", "---", ""]
    for analysis in analyses:
        short, detail = LEVERAGE_GUIDANCE.get(analysis.get("trend", "UNKNOWN"),
                                              LEVERAGE_GUIDANCE["UNKNOWN"])
        quotes = [q for q in analysis.get("specific_quotes", []) if q]
        lines += [
            f"## {analysis['ingredient'].upper()}",
            "",
            f"**Trend:** {analysis.get('trend', 'UNKNOWN')} | "
            f"**Confidence:** {analysis.get('confidence', 'LOW')}",
            "",
            "**Price quotes:** " + ("; ".join(quotes) if quotes else "_none found in sources_"),
            "",
            f"**Analysis:** {analysis.get('summary', 'N/A')}",
            "",
            f"**Leverage:** **{short}:** {detail}",
        ]

        sources = analysis.get("sources") or []
        if sources:
            lines += ["", "**Sources:**"]
            lines += [f"- [{source['title']}]({source['url']}) - {source['date']}"
                      for source in sources]

        lines += ["", "---", ""]

    lines.append("*Generated automatically by the Price Intelligence Agent. "
                 "Verify prices against official sources before negotiating.*")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# MAIN PIPELINE
# ----------------------------------------------------------------------------


def cache_path(ingredient: str) -> Path:
    """Weekly per-ingredient cache file, so prompt-tweaking re-runs are instant."""
    iso = datetime.now(timezone.utc).isocalendar()
    slug = re.sub(r"[^a-z0-9]+", "-", ingredient.lower()).strip("-") or "ingredient"
    return CACHE_DIR / f"{iso[0]}-W{iso[1]:02d}_{slug}.json"


def process_ingredient(ingredient: str, args) -> dict:
    """Search + analyze one ingredient, with a weekly result cache."""
    cache = cache_path(ingredient)
    if not args.no_cache:
        try:
            cached = json.loads(cache.read_text(encoding="utf-8"))
            if isinstance(cached, dict) and cached.get("trend"):
                log.info("[cache] Using cached analysis for %r (--no-cache to refresh)",
                         ingredient)
                return cached
        except (OSError, ValueError):
            pass

    log.info("Processing %r ...", ingredient)
    results = search_ingredient_price(ingredient)
    log.info("  %d search result(s)", len(results))
    analysis = analyze_price_trend(ingredient, results, args.backend,
                                   args.model or DEFAULT_MODELS.get(args.backend, ""))
    log.info("  Trend: %s (confidence %s)",
             analysis.get("trend"), analysis.get("confidence"))

    if not args.no_cache:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(analysis, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        except OSError as exc:
            log.debug("Could not write cache for %r: %s", ingredient, exc)
    return analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingredient price intelligence & renegotiation leverage agent")
    parser.add_argument("--backend", choices=LLM_BACKENDS,
                        default=os.getenv("LLM_BACKEND") or "ollama")
    parser.add_argument("--model", default=None, help="override the backend's model name")
    parser.add_argument("--workers", type=int,
                        default=int(os.getenv("MAX_WORKERS") or "3"),
                        help="parallel ingredient workers (default 3)")
    parser.add_argument("--no-cache", action="store_true",
                        help="ignore the weekly per-ingredient result cache")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    log.info("=== Price Renegotiation Agent ===")

    ingredients = load_ingredients(INGREDIENTS_FILE)
    if not ingredients:
        log.warning("No ingredients found in %s", INGREDIENTS_FILE)
        return
    preview = ", ".join(ingredients[:10]) + (" ..." if len(ingredients) > 10 else "")
    model = args.model or DEFAULT_MODELS.get(args.backend, "")
    log.info("Loaded %d ingredient(s): %s", len(ingredients), preview)
    log.info("Backend: %s (%s) | search window: %d days | workers: %d",
             args.backend, model, SOURCE_WINDOW_DAYS, args.workers)

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            analyses = list(pool.map(lambda i: process_ingredient(ingredients[i], args),
                                     range(len(ingredients))))
    else:
        analyses = [process_ingredient(ingredient, args) for ingredient in ingredients]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / "price_negotiation_report.md"
    report_path.write_text(
        generate_report(analyses, {"backend": args.backend, "model": model}),
        encoding="utf-8")
    log.info("Report saved: %s", report_path)

    (OUT_DIR / "analyses.json").write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend, "model": model,
        "ingredients_file": str(INGREDIENTS_FILE),
        "analyses": analyses,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Audit data saved: %s", OUT_DIR / "analyses.json")


if __name__ == "__main__":
    main()
