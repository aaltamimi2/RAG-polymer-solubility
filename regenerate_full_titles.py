"""
Regenerate visualization with full article titles (no API call needed)
"""

import json
import re
import textwrap
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Read the saved JSON data
with open('./data/polymer_solvent_20years_20260118_154211.json', 'r') as f:
    data = json.load(f)

articles = data.get('articles', [])

def extract_journal(pub_info):
    """Extract journal name from publication info string"""
    if not pub_info or pub_info == 'N/A':
        return 'Unknown Journal'

    parts = pub_info.split('-')
    if len(parts) > 1:
        journal_part = parts[-1].strip()
        journal_part = re.sub(r',?\s*\d{4}\s*-?\s*.*$', '', journal_part)
        if len(journal_part) > 50:
            journal_part = journal_part[:50] + '...'
        return journal_part if journal_part else 'Unknown'

    return 'Unknown'

# Create figure
fig, ax = plt.subplots(figsize=(17, 15))
ax.axis('off')

# Title
plt.title('Polymer and Solvent Research: Last 20 Years (2005-2025)',
          fontsize=16, fontweight='bold', pad=20)

# Starting position
y_pos = 0.97

# Add each article with FULL titles
for i, article in enumerate(articles, 1):
    title = article.get('title', 'N/A')
    # Wrap title instead of truncating - FULL TITLE
    wrapped_title = textwrap.fill(title, width=130)

    pub_info = article.get('publication_info', '')
    journal = extract_journal(pub_info)
    year = article.get('year', 'N/A')
    citations = article.get('cited_by_count', 0)

    # Format: "Article Name" - Journal Name (Year) [Citations]
    article_text = f"{i:2d}.  {wrapped_title}\n      {journal} ({year})"

    if citations > 0:
        article_text += f" - {citations} citations"

    # Color based on citations
    if citations >= 100:
        bg_color = '#d5f4e6'  # Green for highly cited
    elif citations >= 20:
        bg_color = '#fff3cd'  # Yellow for moderately cited
    else:
        bg_color = '#f0f8ff' if i % 2 == 0 else '#ffffff'

    ax.text(0.05, y_pos, article_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=bg_color,
                     edgecolor='#cccccc', linewidth=1))

    # Adjust spacing based on title length
    lines = wrapped_title.count('\n') + 1
    y_pos -= 0.038 + (lines - 1) * 0.012

# Legend
legend_y = y_pos - 0.01
legend_text = "Green = Highly cited (100+)  |  Yellow = Moderately cited (20+)  |  White/Blue = Other"
ax.text(0.05, legend_y, legend_text,
        transform=ax.transAxes,
        fontsize=9, style='italic',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lavender', alpha=0.4))

# Footer
footer = f"Source: Google Scholar | Search: 'polymer solvent' (2005-2025) | {datetime.now().strftime('%Y-%m-%d')}"
ax.text(0.5, 0.005, footer, ha='center', fontsize=8, style='italic',
        transform=ax.transAxes, color='gray')

plt.tight_layout()

# Save
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"./plots/polymer_solvent_full_titles_{timestamp}.png"
plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')

print("=" * 70)
print("✅ FULL TITLES VISUALIZATION CREATED")
print("=" * 70)
print(f"\n📊 File: {filename}")
print("\n✨ Changes:")
print("   • Full article titles (no truncation)")
print("   • Text wrapping for long titles")
print("   • Adjusted spacing for readability")
print("\n🔍 All article names shown completely!")
