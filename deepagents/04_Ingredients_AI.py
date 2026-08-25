import os
import json
import re
from typing import List, Dict, Optional
from datetime import datetime
import requests
from bs4 import BeautifulSoup

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------
INGREDIENTS_FILE = "../data/ingredients.txt"  # or .json / .md
OUTPUT_REPORT = "../reports/price_negotiation_report.md"

# Use "openai", "ollama", or "transformers_local"
LLM_BACKEND = "ollama"   # "ollama" is free and runs locally (pull llama3)
OLLAMA_MODEL = "llama3"
CLOUD_API_KEY = os.getenv("GOOGLE_API_KEY")  # set if using Google AI

# Tavily API
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")  # set to None to use free scraping fallback

# ----------------------------------------------------------------------------
# STEP 1: Load Ingredients from File
# ----------------------------------------------------------------------------
def load_ingredients(filepath: str) -> List[str]:
    ext = os.path.splitext(filepath)[1].lower()
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    if ext == '.json':
        data = json.loads(content)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'ingredients' in data:
            return data['ingredients']
    elif ext == '.md':
        # Markdown: assume bullet list or comma-separated
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        items = []
        for line in lines:
            if line.startswith('- ') or line.startswith('* '):
                items.append(line[2:].strip())
            elif line.startswith('1. ') or re.match(r'^\d+\.', line):
                items.append(line.split('.', 1)[1].strip())
        if items:
            return items
    # Fallback: treat as simple text lines or comma-separated
    if ',' in content and '\n' not in content:
        return [i.strip() for i in content.split(',') if i.strip()]
    return [line.strip() for line in content.splitlines() if line.strip()]

# ----------------------------------------------------------------------------
# STEP 2: Web Search (Tavily preferred, fallback with Google News RSS)
# ----------------------------------------------------------------------------
def search_ingredient_price(ingredient: str) -> List[Dict]:
    """Return list of {url, title, snippet}"""
    query = f"{ingredient} wholesale price trend 2024 2025 site:gov OR market report"
    
    # Try Tavily first
    if TAVILY_API_KEY:
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=TAVILY_API_KEY)
            response = client.search(query, max_results=5, include_raw_content=False)
            results = []
            for r in response['results']:
                results.append({
                    'url': r['url'],
                    'title': r['title'],
                    'snippet': r.get('content', '')
                })
            if results:
                return results
        except Exception as e:
            print(f"Tavily error: {e}. Falling back.")

    # Fallback: Google News RSS (free, no key)
    print(f"Using fallback search for {ingredient}...")
    try:
        rss_url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
        resp = requests.get(rss_url, timeout=10)
        soup = BeautifulSoup(resp.content, 'xml')
        items = soup.find_all('item')[:5]
        results = []
        for item in items:
            title = item.title.text if item.title else ""
            link = item.link.text if item.link else ""
            desc = item.description.text if item.description else ""
            results.append({'url': link, 'title': title, 'snippet': desc})
        return results
    except Exception as e:
        print(f"Fallback search failed: {e}")
        return []

# ----------------------------------------------------------------------------
# STEP 3: Scrape and Clean Webpage Text
# ----------------------------------------------------------------------------
def fetch_page_text(url: str, max_chars: int = 3000) -> str:
    """Fetch and extract main text from a URL."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = requests.get(url, timeout=15, headers=headers)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, 'html.parser')
        # Remove script/style tags
        for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
            tag.decompose()
        text = soup.get_text(separator=' ', strip=True)
        # Collapse whitespace and truncate
        text = re.sub(r'\s+', ' ', text)
        return text[:max_chars]
    except Exception as e:
        return f"Error fetching: {e}"

# ----------------------------------------------------------------------------
# STEP 4: LLM Analyst (Local with Ollama, or OpenAI)
# ----------------------------------------------------------------------------
def call_llm(prompt: str) -> str:
    if LLM_BACKEND == "ollama":
        import ollama
        response = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
        return response['message']['content'].strip()
    
    elif LLM_BACKEND == "openai":
        import openai
        openai.api_key = CLOUD_API_KEY
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    
    else:  # local transformers (requires GPU)
        from transformers import pipeline
        pipe = pipeline("text-generation", model="microsoft/Phi-3-mini-4k-instruct", device=0)
        messages = [{"role": "user", "content": prompt}]
        return pipe(messages, max_new_tokens=500)[0]['generated_text'][-1]['content']

def analyze_price_trend(ingredient: str, search_results: List[Dict]) -> Dict:
    """Use LLM to extract price direction, specific numbers, and evidence."""
    if not search_results:
        return {
            "ingredient": ingredient,
            "trend": "UNKNOWN",
            "price_quotes": [],
            "summary": "No search results found.",
            "sources": []
        }
    
    # Build context
    context = ""
    sources = []
    for i, res in enumerate(search_results[:4]):
        text = fetch_page_text(res['url'])
        snippet = res.get('snippet', '')
        sources.append(res['url'])
        context += f"\n--- SOURCE {i+1} ({res['url']}) ---\nTitle: {res['title']}\nContent: {text[:1500]}\n"
    
    prompt = f"""
You are a commodity pricing analyst. Given the following web excerpts for "{ingredient}", determine:
1. The overall price trend (UP, DOWN, or STABLE) over the last 3-6 months.
2. Specific price quotes (with currency and date) if mentioned.
3. A 2-3 sentence executive summary for a procurement team.

Return your answer strictly in this JSON format:
{{
  "trend": "UP/DOWN/STABLE",
  "confidence": "HIGH/MEDIUM/LOW",
  "specific_quotes": ["quote1", "quote2"],
  "summary": "Your concise summary."
}}

Context:
{context}
"""
    
    raw = call_llm(prompt)
    try:
        # Extract JSON from response (LLM might wrap in markdown)
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
        else:
            raise ValueError("No JSON found")
    except:
        # Fallback: manual parsing
        trend = "UNKNOWN"
        if "UP" in raw.upper(): trend = "UP"
        elif "DOWN" in raw.upper(): trend = "DOWN"
        elif "STABLE" in raw.upper(): trend = "STABLE"
        data = {
            "trend": trend,
            "confidence": "LOW",
            "specific_quotes": ["Not extracted"],
            "summary": raw[:300]
        }
    
    data["sources"] = sources[:3]
    data["ingredient"] = ingredient
    return data

# ----------------------------------------------------------------------------
# STEP 5: Generate Renegotiation Report (Markdown + optional PDF)
# ----------------------------------------------------------------------------
def generate_report(analyses: List[Dict]) -> str:
    report = f"# Procurement Price Renegotiation Report\n"
    report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    report += "## Executive Summary\n"
    report += "This report provides actionable intelligence on key ingredient price trends to support supplier contract renegotiations.\n\n"

    for item in analyses:
        report += f"---\n"
        report += f"## {item['ingredient'].upper()}\n"
        report += f"- **Trend:** {item.get('trend', 'UNKNOWN')}\n"
        report += f"- **Confidence:** {item.get('confidence', 'LOW')}\n"
        report += f"- **Price Quotes:** {', '.join(item.get('specific_quotes', ['None found']))}\n"
        report += f"- **Summary:** {item.get('summary', 'N/A')}\n"
        
        # Renegotiation Leverage
        trend = item.get('trend', 'UNKNOWN')
        if trend == "DOWN":
            leverage = "✅ **Strong Renegotiation Leverage:** Prices are falling. Demand a 5-10% reduction from current contracts, citing recent market drops."
        elif trend == "UP":
            leverage = "⚠️ **Limited Leverage:** Prices are rising. Consider securing long-term fixed pricing or exploring alternative suppliers."
        else:
            leverage = "⚖️ **Neutral:** Prices are stable. Renegotiate based on volume/relationship rather than market pressure."
        report += f"- **Leverage:** {leverage}\n"
        
        if item.get('sources'):
            report += f"- **Sources:**\n"
            for url in item['sources']:
                report += f"  - {url}\n"
        report += "\n"

    report += "\n---\n"
    report += "*This report was generated automatically by the Price Intelligence Agent.*"
    return report

# ----------------------------------------------------------------------------
# MAIN PIPELINE
# ----------------------------------------------------------------------------
def main():
    print("=== Price Renegotiation Agent ===\n")
    
    # 1. Load ingredients
    ingredients = load_ingredients(INGREDIENTS_FILE)
    print(f"Loaded {len(ingredients)} ingredient(s): {', '.join(ingredients)}\n")
    
    if not ingredients:
        print("No ingredients found. Check your input file.")
        return

    # 2. Analyze each ingredient
    all_analyses = []
    for ing in ingredients:
        print(f"Processing: {ing} ...")
        results = search_ingredient_price(ing)
        print(f"  Found {len(results)} sources.")
        analysis = analyze_price_trend(ing, results)
        all_analyses.append(analysis)
        print(f"  Trend: {analysis.get('trend')}\n")

    # 3. Generate report
    report_md = generate_report(all_analyses)
    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        f.write(report_md)
    print(f"\n✅ Report saved to: {OUTPUT_REPORT}")

    # 4. Optional: Convert to PDF (requires weasyprint)
    try:
        from weasyprint import HTML
        html_content = f"<html><body>{report_md.replace('\n', '<br>')}</body></html>"
        HTML(string=html_content).write_pdf("price_negotiation_report.pdf")
        print("✅ PDF generated: price_negotiation_report.pdf")
    except:
        print("PDF generation skipped (weasyprint not installed).")

if __name__ == "__main__":
    main()