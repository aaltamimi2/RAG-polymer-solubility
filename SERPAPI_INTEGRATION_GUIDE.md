# Google Scholar Integration Guide (SerpAPI)

Complete guide for integrating Google Scholar article search into the Polymer Solubility App using SerpAPI.

## Overview

SerpAPI provides programmatic access to Google Scholar search results without violating terms of service. This integration enables automated extraction of academic articles related to polymer solubility research.

**Authentication:** Simple API Key
**Endpoint:** https://serpapi.com/search
**Free Tier:** 100 searches/month

## Setup Instructions

### 1. Get SerpAPI Key

1. Visit: https://serpapi.com/
2. Sign up using your email: **aaltamimi2@wisc.edu**
3. Choose a plan:
   - **Free:** 100 searches/month (great for testing)
   - **Developer:** $50/month, 5,000 searches
   - **Production:** Custom pricing
4. Get your API key from the dashboard

### 2. Configure Environment Variable

Add your API key to the `.env` file:

```bash
# Edit .env
nano .env

# Add this line:
SERPAPI_KEY=your-api-key-here
```

### 3. Install Dependencies

SerpAPI client uses `requests` (already in your project):

```bash
pip install requests python-dotenv
```

## Quick Start

### Test Your Setup

```bash
python test_scholar_integration.py
```

Expected output:
```
✅ Connection successful!
📊 Account Information:
   Email: aaltamimi2@wisc.edu
   Plan: Free
   Searches this month: 5
   Total searches left: 95
```

### Basic Usage

```python
from serpapi_scholar_client import GoogleScholarClient

# Initialize client (uses SERPAPI_KEY from .env)
client = GoogleScholarClient()

# Simple search
results = client.search(query="polymer solubility", num_results=10)

# Polymer-specific search
articles = client.search_polymer_articles(
    polymer_name="polyethylene",
    solvent_name="toluene",
    year_low=2020,
    year_high=2024,
    max_results=10
)

# Print results
for article in articles:
    print(f"{article['title']} ({article['year']})")
    print(f"Citations: {article['cited_by_count']}")
    if article['pdf_link']:
        print(f"PDF: {article['pdf_link']}")
    print()
```

## Search Query Syntax

Google Scholar search supports various operators:

### Basic Operators

```python
# Exact phrase
query = '"Hansen solubility parameters"'

# Multiple terms (AND)
query = 'polymer solubility dissolution'

# OR operator
query = 'polyethylene OR polypropylene'

# Exclude terms
query = 'polymer -plastic'
```

### Field-Specific Searches

```python
# Author search
query = 'author:"Charles Hansen"'

# Title search
query = 'intitle:"Hansen parameters"'

# Source/journal search
query = 'source:"Journal of Polymer Science"'
```

### Advanced Examples

```python
# Exact phrase + year filter
articles = client.search(
    query='"polymer solubility"',
    year_low=2020,
    year_high=2024
)

# Multiple polymers
query = '(polyethylene OR polypropylene) AND solubility'
results = client.search(query=query)

# Specific research area
query = '"Hansen solubility parameters" polymer coatings'
results = client.search(query=query)
```

## API Methods

### `GoogleScholarClient` Class

#### Constructor
```python
client = GoogleScholarClient(
    api_key=None  # Optional, uses SERPAPI_KEY env var
)
```

#### Core Methods

**`search(query, num_results=10, year_low=None, year_high=None, sort_by=None, include_patents=False, include_citations=False)`**
- General Google Scholar search
- Max 20 results per page
- Returns: Full SerpAPI response dictionary

**`search_polymer_articles(polymer_name=None, solvent_name=None, year_low=None, year_high=None, max_results=10)`**
- Specialized search for polymer solubility articles
- Builds intelligent queries automatically
- Returns: List of parsed article dictionaries

**`search_hansen_parameters(polymer_name=None, year_low=None, year_high=None, max_results=10)`**
- Search specifically for Hansen solubility parameter research
- Optional polymer filter
- Returns: List of parsed article dictionaries

**`get_author_articles(author_name, max_results=10)`**
- Find articles by specific author
- Example: `"Charles Hansen"` or `"Hansen CM"`
- Returns: List of parsed article dictionaries

**`test_connection()`**
- Test API key validity
- Returns: `True` if successful

**`get_account_info()`**
- Get account details and usage statistics
- Returns: Dictionary with plan info, searches remaining, etc.

## Article Record Format

```python
{
    'title': 'Article Title',
    'authors': ['Author One', 'Author Two'],
    'year': '2024',
    'snippet': 'Brief summary...',
    'link': 'https://...',
    'pdf_link': 'https://...pdf' or None,
    'cited_by_count': 123,
    'cited_by_link': 'https://scholar.google.com/...',
    'versions_count': 5,
    'publication_info': 'Authors - Journal, Year'
}
```

## Rate Limits and Best Practices

### Rate Limits (Free Tier)
- **100 searches/month**
- No per-second rate limit
- Cached results are free (don't count toward quota)

### Best Practices

1. **Use Specific Queries**
   - Narrow searches return better results
   - Use exact phrases with quotes
   - Filter by year range when possible

2. **Cache Results**
   - Store search results locally
   - Avoid repeating identical queries
   - SerpAPI caches searches automatically

3. **Monitor Usage**
   ```python
   account = client.get_account_info()
   print(f"Searches left: {account['total_searches_left']}")
   ```

4. **Batch Processing**
   - Process multiple articles efficiently
   - Extract all needed info in one search
   - Use pagination for large result sets

5. **Error Handling**
   ```python
   try:
       articles = client.search_polymer_articles(...)
   except Exception as e:
       print(f"Search failed: {e}")
   ```

## Comparison: SerpAPI vs Web of Science

| Feature | SerpAPI (Google Scholar) | Web of Science |
|---------|-------------------------|----------------|
| **Setup** | Simple (API key) | Complex (OAuth 2.0) |
| **Free Tier** | 100 searches/month | Varies by institution |
| **Coverage** | Broad (all sources) | Curated journals only |
| **Citations** | Yes (Google Scholar) | Yes (WoS) |
| **PDF Links** | Often available | Not directly |
| **Metadata** | Good | Excellent |
| **Best For** | Quick searches, testing | Comprehensive research |

## Example Use Cases

### 1. Find Recent Articles on Polymer Solubility

```python
articles = client.search_polymer_articles(
    year_low=2024,
    year_high=2024,
    max_results=20
)

for article in articles:
    print(f"{article['title']}")
    print(f"Citations: {article['cited_by_count']}")
```

### 2. Research Specific Polymer-Solvent Pair

```python
articles = client.search_polymer_articles(
    polymer_name="cellulose acetate",
    solvent_name="acetone",
    year_low=2015,
    max_results=15
)
```

### 3. Track Hansen Parameters Research

```python
articles = client.search_hansen_parameters(
    year_low=2020,
    max_results=20
)

# Find highly cited papers
top_cited = sorted(articles, key=lambda x: x['cited_by_count'], reverse=True)
for article in top_cited[:5]:
    print(f"{article['title']} - {article['cited_by_count']} citations")
```

### 4. Author Publication History

```python
articles = client.get_author_articles(
    author_name="Charles Hansen",
    max_results=20
)
```

### 5. Find Papers with PDFs

```python
articles = client.search_polymer_articles(
    polymer_name="polyethylene",
    max_results=20
)

pdf_articles = [a for a in articles if a['pdf_link']]
print(f"Found {len(pdf_articles)} articles with PDFs")
```

## Error Handling

### Common Errors

**Invalid API Key (401)**
```
Error: Invalid API key
```
**Solution:** Check SERPAPI_KEY in `.env` file

**Rate Limit Exceeded (429)**
```
Error: You have reached your monthly search limit
```
**Solution:** Wait until next month or upgrade plan

**No Results Found**
```
organic_results: []
```
**Solution:** Broaden search query, remove year filters

## Integration with Polymer Solubility Agent

### Add Scholar Search Tool

```python
from serpapi_scholar_client import GoogleScholarClient

# Initialize client
scholar_client = GoogleScholarClient()

@tool
def search_literature(
    query: str,
    max_results: int = 5
) -> str:
    """
    Search Google Scholar for academic articles

    Args:
        query: Search query
        max_results: Max results to return

    Returns:
        Formatted article list
    """
    articles = scholar_client.search_polymer_articles(
        polymer_name=query if "polymer" not in query.lower() else None,
        max_results=max_results
    )

    output = [f"Found {len(articles)} articles:\n"]

    for i, article in enumerate(articles, 1):
        output.append(f"{i}. {article['title']}")
        output.append(f"   Authors: {', '.join(article['authors'][:3])}")
        output.append(f"   Year: {article['year']} | Citations: {article['cited_by_count']}")
        if article['pdf_link']:
            output.append(f"   📄 PDF: {article['pdf_link']}")
        output.append(f"   🔗 {article['link']}\n")

    return "\n".join(output)
```

## Troubleshooting

### Import Error
```bash
# Ensure you're in the correct directory
cd /home/aaltamimi2/polymer-solubility-app
python test_scholar_integration.py
```

### Connection Timeout
- Check internet connectivity
- Verify SerpAPI status: https://status.serpapi.com/
- Increase timeout in `_make_request()`

### Unexpected Results
- Review query syntax
- Check for typos in polymer/solvent names
- Use exact phrases with quotes for better matches

### No PDF Links
- Not all articles have public PDFs
- Check institutional access
- Use `cited_by_link` to find related papers

## Advanced Features

### Pagination

```python
# Get first page
results_page1 = client.search(query="polymer", num_results=20)

# Get second page (not directly supported by method, use raw params)
# Note: This counts as an additional search
```

### Sort by Date

```python
articles = client.search(
    query="polymer solubility",
    num_results=20,
    sort_by='date'  # Most recent first
)
```

### Include Patents

```python
results = client.search(
    query="polymer coating",
    num_results=10,
    include_patents=True
)
```

## Resources

- **SerpAPI Website:** https://serpapi.com/
- **Documentation:** https://serpapi.com/google-scholar-api
- **Pricing:** https://serpapi.com/pricing
- **Dashboard:** https://serpapi.com/dashboard
- **Status Page:** https://status.serpapi.com/

## Cost Optimization Tips

1. **Use Free Tier Wisely**
   - Cache results locally
   - Avoid duplicate searches
   - Test queries carefully before running

2. **Leverage Cache**
   - SerpAPI caches results automatically
   - Cached searches are FREE
   - Cache expires after ~1 hour

3. **Optimize Queries**
   - Use specific terms to get fewer, better results
   - Filter by year to reduce result count
   - Exact phrases reduce irrelevant matches

4. **Monitor Usage**
   ```python
   # Check before each search
   account = client.get_account_info()
   if account['total_searches_left'] < 10:
       print("⚠️ Low on searches!")
   ```

## Next Steps

1. ✅ Sign up at https://serpapi.com/ with aaltamimi2@wisc.edu
2. ✅ Get your API key from the dashboard
3. ✅ Add SERPAPI_KEY to .env file
4. Test with: `python test_scholar_integration.py`
5. Integrate with agent tools
6. Create UI for article browsing
7. Combine with Web of Science when available

---

**Questions?**
- Review test script: `test_scholar_integration.py`
- Check SerpAPI documentation
- Contact SerpAPI support for technical issues
