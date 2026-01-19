# Web of Science API Integration Guide

Complete guide for integrating Web of Science article extraction into the Polymer Solubility App.

## Overview

The Web of Science (WoS) API integration enables automated extraction of scientific articles related to polymer solubility, Hansen solubility parameters, and related research topics.

**Authentication Method:** OAuth 2.0 Client Credentials Flow

## Setup Instructions

### 1. Get Web of Science API Credentials

1. Visit the Clarivate Developer Portal: https://developer.clarivate.com/
2. Create an account or sign in
3. Subscribe to the **Web of Science API**
4. Create a new application
5. Note your **Client ID** and **Client Secret**

### 2. Configure Environment Variables

Add your credentials to the `.env` file:

```bash
# Copy example if you haven't already
cp .env.example .env

# Edit .env and add:
WOS_CLIENT_ID=your-client-id-here
WOS_CLIENT_SECRET=your-client-secret-here
```

**Important:** Never commit `.env` to version control!

### 3. Install Dependencies

The WoS client requires `requests` (already in your project):

```bash
pip install requests python-dotenv
```

## Usage

### Quick Test

Run the test script to verify your credentials:

```bash
python test_wos_integration.py
```

Expected output:
```
✅ Access token obtained
✅ Web of Science API connection successful!
📊 Total results: 1234
```

### Basic Usage in Python

```python
from wos_api_client import WebOfScienceClient

# Initialize client (uses env vars)
client = WebOfScienceClient()

# Search for articles
results = client.search_articles(
    query='TS=(polymer solubility)',
    count=10
)

# Search polymer-specific articles
articles = client.search_polymer_solubility_articles(
    polymer_name="polyethylene",
    solvent_name="toluene",
    year_range="2020-2024",
    max_results=10
)

# Print results
for article in articles:
    print(f"{article['title']}")
    print(f"Authors: {', '.join(article['authors'])}")
    print(f"Year: {article['year']}")
    print(f"DOI: {article['doi']}")
    print()
```

## Query Syntax

Web of Science uses a specialized query language:

### Field Codes

- `TS=` - Topic (searches title, abstract, keywords)
- `TI=` - Title
- `AU=` - Author
- `PY=` - Publication Year
- `SO=` - Source (journal name)
- `DO=` - DOI

### Example Queries

```python
# Topic search
query = 'TS=(polymer solubility)'

# Multiple topics (AND)
query = 'TS=(polymer) AND TS=(Hansen parameters)'

# Multiple topics (OR)
query = 'TS=(polyethylene OR polypropylene)'

# Year range
query = 'TS=(polymer) AND PY=(2020-2024)'

# Specific author
query = 'AU=(Hansen C) AND TS=(solubility parameters)'

# Combined search
query = 'TS=(polymer solubility) AND PY=(2020-2024) AND SO=(Polymer)'
```

## API Methods

### `WebOfScienceClient` Class

#### Constructor
```python
client = WebOfScienceClient(
    client_id=None,        # Optional, uses WOS_CLIENT_ID env var
    client_secret=None,    # Optional, uses WOS_CLIENT_SECRET env var
    auth_method="realms/api",
    api_name="wos"
)
```

#### Methods

**`authenticate()`**
- Obtains OAuth 2.0 access token
- Token automatically refreshed when expired
- Returns: `WoSToken` object

**`search_articles(query, database='WOS', count=10, first_record=1, sort_field=None)`**
- Search for articles
- Returns: Full WoS API response dictionary

**`get_article_by_uid(uid)`**
- Retrieve single article by WoS UID
- Example UID: `'WOS:000123456789'`
- Returns: Article metadata dictionary

**`search_polymer_solubility_articles(polymer_name=None, solvent_name=None, year_range=None, max_results=10)`**
- Specialized search for polymer solubility articles
- Builds intelligent queries based on parameters
- Returns: List of parsed article dictionaries

**`test_connection()`**
- Test API connection and authentication
- Returns: `True` if successful, `False` otherwise

## Article Record Format

Parsed articles have the following structure:

```python
{
    'uid': 'WOS:000123456789',
    'title': 'Article Title',
    'authors': ['Author One', 'Author Two'],
    'year': '2024',
    'source': 'Journal Name',
    'doi': '10.1234/example',
    'abstract': 'Article abstract text...'
}
```

## Rate Limits and Best Practices

### Rate Limits
- Varies by subscription plan
- Typical: 5-10 requests per second
- Token expiry: 3600 seconds (1 hour)

### Best Practices
1. **Cache results** - Don't repeat identical queries
2. **Batch requests** - Process multiple articles efficiently
3. **Handle errors gracefully** - Implement retry logic
4. **Monitor quotas** - Track API usage
5. **Use specific queries** - Narrow searches for better results

## Error Handling

Common errors and solutions:

### 401 Unauthorized
```
Message: "No API key found in request"
```
**Solution:** Check WOS_CLIENT_ID and WOS_CLIENT_SECRET in `.env`

### 403 Forbidden
```
Message: "Insufficient permissions"
```
**Solution:** Verify your API subscription includes the endpoint you're accessing

### 429 Too Many Requests
```
Message: "Rate limit exceeded"
```
**Solution:** Implement exponential backoff, reduce request frequency

### Token Expiry
Tokens expire after 1 hour. The client automatically refreshes expired tokens.

## Integration with Polymer Solubility Agent

### Example: Add WoS Tool to Agent

```python
from wos_api_client import WebOfScienceClient

# Initialize WoS client
wos_client = WebOfScienceClient()

# Create agent tool
@tool
def search_polymer_literature(
    polymer_name: str,
    solvent_name: str = None,
    max_results: int = 5
) -> str:
    """
    Search Web of Science for polymer solubility literature

    Args:
        polymer_name: Name of polymer
        solvent_name: Optional solvent name
        max_results: Max articles to return

    Returns:
        Formatted article list
    """
    articles = wos_client.search_polymer_solubility_articles(
        polymer_name=polymer_name,
        solvent_name=solvent_name,
        max_results=max_results
    )

    output = [f"Found {len(articles)} articles:\n"]

    for i, article in enumerate(articles, 1):
        output.append(f"{i}. {article['title']}")
        output.append(f"   Authors: {', '.join(article['authors'][:3])}")
        output.append(f"   Year: {article['year']} | DOI: {article['doi']}\n")

    return "\n".join(output)
```

## Troubleshooting

### Import Error: `No module named 'wos_api_client'`
Ensure you're in the project directory:
```bash
cd /home/aaltamimi2/polymer-solubility-app
python test_wos_integration.py
```

### Connection Timeout
- Check internet connectivity
- Verify Clarivate API status: https://status.clarivate.com/
- Increase timeout in `_make_request()` method

### Invalid Query Syntax
- Review WoS query language documentation
- Test queries in Web of Science web interface first
- Use proper field codes (TS=, AU=, PY=, etc.)

## Resources

- **Clarivate Developer Portal:** https://developer.clarivate.com/
- **WoS API Documentation:** https://developer.clarivate.com/apis/wos
- **Query Language Guide:** https://webofscience.help.clarivate.com/
- **API Status Page:** https://status.clarivate.com/

## Example Use Cases

### 1. Find Articles for Specific Polymer-Solvent Pair
```python
articles = client.search_polymer_solubility_articles(
    polymer_name="cellulose acetate",
    solvent_name="acetone",
    year_range="2020-2024"
)
```

### 2. Hansen Parameters Research
```python
results = client.search_articles(
    query='TS=("Hansen solubility parameters")',
    count=20,
    sort_field='TC+D'  # Sort by times cited, descending
)
```

### 3. Track Recent Publications
```python
# Get latest polymer solubility papers
articles = client.search_polymer_solubility_articles(
    year_range="2024-2024",
    max_results=50
)
```

## Next Steps

1. ✅ Test the integration with `python test_wos_integration.py`
2. Add WoS tools to the agent (`agent_sql_final_1212_patched.py`)
3. Create UI components for article browsing
4. Implement citation export (BibTeX, RIS)
5. Add article recommendation based on current query

---

**Need Help?**
- Check the test script: `test_wos_integration.py`
- Review API logs for detailed error messages
- Consult Clarivate support for subscription issues
