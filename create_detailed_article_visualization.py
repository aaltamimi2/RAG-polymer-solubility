"""
Create detailed article visualization with titles, years, journals, and keywords
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
from datetime import datetime

# Read the saved JSON data
with open('./data/polymer_dissolution_search_20260118_153343.json', 'r') as f:
    data = json.load(f)

# Extract articles
articles = data.get('articles', [])[:10]  # Top 10 articles

# Function to extract journal from publication_info
def extract_journal(pub_info):
    """Extract journal name from publication info string"""
    if not pub_info or pub_info == 'N/A':
        return 'Unknown Journal'

    # Try to extract journal name (usually after dash and before year)
    # Format: "Authors - Journal, Year"
    parts = pub_info.split('-')
    if len(parts) > 1:
        journal_part = parts[-1].strip()
        # Remove year
        journal_part = re.sub(r',?\s*\d{4}\s*-?\s*\w*$', '', journal_part)
        # Truncate if too long
        if len(journal_part) > 40:
            journal_part = journal_part[:40] + '...'
        return journal_part if journal_part else 'Unknown Journal'

    return 'Unknown Journal'

# Function to extract keywords from snippet
def extract_keywords(snippet, title):
    """Extract potential keywords from snippet and title"""
    if not snippet or snippet == 'N/A':
        return []

    # Common polymer science keywords to look for
    keyword_patterns = [
        'polymer', 'dissolution', 'solvent', 'solubility', 'Hansen',
        'precipitation', 'crystallization', 'molecular', 'synthesis',
        'characterization', 'membrane', 'recycling', 'electrolyte',
        'PVC', 'polyethylene', 'polypropylene', 'cellulose', 'chitosan',
        'bio-aerogel', 'electrospinning', 'kinetics', 'thermodynamics',
        'rheology', 'DSC', 'parameters', 'compatibility', 'miscible'
    ]

    found_keywords = []
    text_lower = (snippet + ' ' + title).lower()

    for keyword in keyword_patterns:
        if keyword.lower() in text_lower:
            found_keywords.append(keyword.capitalize())

    # Remove duplicates and limit to 5
    found_keywords = list(dict.fromkeys(found_keywords))[:5]

    return found_keywords

# Create figure
fig = plt.figure(figsize=(18, 12))
fig.suptitle('Polymer Dissolution Literature: Detailed Article Breakdown (2020-2025)',
             fontsize=16, fontweight='bold', y=0.98)

# Main content area
ax = plt.subplot(1, 1, 1)
ax.axis('off')

# Build detailed article list
y_position = 0.95
x_left = 0.02
line_spacing = 0.094

# Header
header_text = "Article #    Title | Year | Journal | Keywords"
ax.text(x_left, y_position, header_text,
        transform=ax.transAxes,
        fontsize=11, fontweight='bold',
        family='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5, edgecolor='black', linewidth=2))

y_position -= 0.03

for i, article in enumerate(articles, 1):
    # Extract information
    title = article.get('title', 'N/A')
    if len(title) > 85:
        title = title[:85] + '...'

    year = article.get('year', 'N/A')
    pub_info = article.get('publication_info', '')
    journal = extract_journal(pub_info)
    keywords = extract_keywords(article.get('snippet', ''), title)
    keywords_str = ', '.join(keywords) if keywords else 'No keywords identified'

    citations = article.get('cited_by_count', 0)
    has_pdf = article.get('pdf_link') is not None

    # Color coding based on citations
    if citations >= 2:
        bg_color = '#d5f4e6'  # Green for highly cited
        edge_color = '#27ae60'
    elif has_pdf:
        bg_color = '#fff9e6'  # Yellow for PDF available
        edge_color = '#f39c12'
    else:
        bg_color = '#e8f4f8'  # Blue for others
        edge_color = '#3498db'

    # Create article box
    article_text = f"""
{i:2d}.  TITLE: {title}

     YEAR: {year}
     JOURNAL: {journal}
     KEYWORDS: {keywords_str}
     CITATIONS: {citations}  |  PDF: {'Available' if has_pdf else 'Not Available'}
"""

    y_position -= line_spacing

    ax.text(x_left, y_position, article_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.8', facecolor=bg_color,
                     alpha=0.6, edgecolor=edge_color, linewidth=1.5),
            family='monospace')

# Legend
legend_y = y_position - 0.05
legend_text = """
Legend:  Green box = Highly cited (2+ citations)  |  Yellow box = PDF Available  |  Blue box = Other articles
"""

ax.text(x_left, legend_y, legend_text,
        transform=ax.transAxes,
        fontsize=9, style='italic',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lavender', alpha=0.4))

# Footer
footer_text = f"""Source: Google Scholar via SerpAPI  |  Search: "polymer dissolution" (2020-2025)  |  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"""

fig.text(0.5, 0.01, footer_text, ha='center', fontsize=8, style='italic',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.2))

plt.tight_layout(rect=[0, 0.02, 1, 0.97])

# Save figure
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"./plots/polymer_dissolution_detailed_{timestamp}.png"
plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')

print("=" * 80)
print("✅ DETAILED ARTICLE VISUALIZATION CREATED")
print("=" * 80)
print(f"\n📊 File saved: {filename}")
print(f"\n✨ This visualization includes:")
print("   • Article titles (truncated for readability)")
print("   • Publication year")
print("   • Journal/source information")
print("   • Extracted keywords from abstracts")
print("   • Citation counts")
print("   • PDF availability")
print("   • Color-coded by relevance (citations/PDF access)")
print("\n📚 Showing top 10 articles from the search results")
print("\n🔍 View the detailed visualization in the plots directory")
