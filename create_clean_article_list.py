"""
Create clean, simple article list visualization
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
articles = data.get('articles', [])[:15]  # Top 15 articles

# Function to extract journal from publication_info
def extract_journal(pub_info):
    """Extract journal name from publication info string"""
    if not pub_info or pub_info == 'N/A':
        return 'Unknown Journal'

    # Try to extract journal name (usually after dash and before year)
    parts = pub_info.split('-')
    if len(parts) > 1:
        journal_part = parts[-1].strip()
        # Remove year and publisher
        journal_part = re.sub(r',?\s*\d{4}\s*-?\s*.*$', '', journal_part)
        # Truncate if too long
        if len(journal_part) > 50:
            journal_part = journal_part[:50] + '...'
        return journal_part if journal_part else 'Unknown'

    return 'Unknown'

# Create figure
fig, ax = plt.subplots(figsize=(16, 11))
ax.axis('off')

# Title
plt.title('Polymer Dissolution: Recent Research Articles (2020-2025)',
          fontsize=16, fontweight='bold', pad=20)

# Starting position
y_pos = 0.95

# Add each article
for i, article in enumerate(articles, 1):
    title = article.get('title', 'N/A')
    if len(title) > 100:
        title = title[:100] + '...'

    pub_info = article.get('publication_info', '')
    journal = extract_journal(pub_info)
    year = article.get('year', 'N/A')

    # Format: "Article Name" - Journal Name (Year)
    article_text = f"{i:2d}.  {title}\n      {journal} ({year})"

    # Color alternate rows for readability
    bg_color = '#f0f8ff' if i % 2 == 0 else '#ffffff'

    ax.text(0.05, y_pos, article_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.6', facecolor=bg_color,
                     edgecolor='#cccccc', linewidth=1))

    y_pos -= 0.062

# Footer
footer = f"Source: Google Scholar | Search: 'polymer dissolution' | {datetime.now().strftime('%Y-%m-%d')}"
ax.text(0.5, 0.01, footer, ha='center', fontsize=9, style='italic',
        transform=ax.transAxes, color='gray')

plt.tight_layout()

# Save
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"./plots/polymer_dissolution_clean_{timestamp}.png"
plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')

print("=" * 70)
print("✅ CLEAN ARTICLE LIST CREATED")
print("=" * 70)
print(f"\n📊 File: {filename}")
print(f"\n✨ Simple format:")
print("   • Article title")
print("   • Journal name")
print("   • Year")
print("\n🔍 Easy to read and scan!")
