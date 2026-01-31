"""
Create a simple, clean single visualization from saved search results
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# Read the saved JSON data
with open('./data/polymer_dissolution_search_20260118_153343.json', 'r') as f:
    data = json.load(f)

# Extract key information
analysis = data['analysis']
articles = data.get('articles', [])

# Create a single, clean figure
fig, ax = plt.subplots(figsize=(14, 8))

# Remove the main axes
ax.axis('off')

# Title
fig.suptitle('Polymer Dissolution Research: Recent Literature (2020-2025)',
             fontsize=18, fontweight='bold', y=0.96)

# Left side - Key Statistics Box
stats_text = f"""
KEY STATISTICS
{'='*40}

📊 Total Articles: {analysis['total_articles']}

📅 Publication Years: 2025-2026

📈 Citations:
   • Total: {analysis['citation_stats']['total_citations']}
   • Average: {analysis['citation_stats']['avg_citations']:.1f}
   • Range: {analysis['citation_stats']['min_citations']}-{analysis['citation_stats']['max_citations']}

📄 PDF Access: {analysis['pdf_available']} papers ({analysis['pdf_available']/analysis['total_articles']*100:.0f}%)

👥 Unique Authors: {len(analysis.get('top_authors', []))}+ researchers

🔬 Research Focus Areas:
   • Hansen Solubility Parameters
   • Polymer Recycling Methods
   • Solvent Selection Optimization
   • Bio-based Materials
   • Computational Modeling
"""

# Add statistics box
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
        fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.3, edgecolor='navy', linewidth=2),
        family='monospace')

# Right side - Top Cited Articles
top_cited = sorted([a for a in articles if a.get('cited_by_count', 0) > 0],
                   key=lambda x: x.get('cited_by_count', 0), reverse=True)[:5]

if not top_cited:
    top_cited = articles[:5]  # Show first 5 if no citations

cited_text = "TOP RESEARCH ARTICLES\n" + "="*50 + "\n\n"

for i, article in enumerate(top_cited[:5], 1):
    title = article.get('title', 'N/A')
    if len(title) > 50:
        title = title[:50] + "..."

    year = article.get('year', 'N/A')
    citations = article.get('cited_by_count', 0)
    authors = article.get('authors', [])
    author_str = authors[0] if authors else 'Unknown'

    if len(authors) > 1:
        author_str += ' et al.'

    cited_text += f"{i}. {title}\n"
    cited_text += f"   {author_str} ({year})\n"
    cited_text += f"   📊 Citations: {citations}\n"
    cited_text += f"   {'✅ PDF Available' if article.get('pdf_link') else '⚠️ No PDF'}\n\n"

# Add top articles box
ax.text(0.52, 0.98, cited_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', alpha=0.3, edgecolor='darkgoldenrod', linewidth=2),
        family='monospace')

# Bottom - Publications Timeline
timeline_ax = plt.subplot(4, 1, 4)
years = list(analysis['year_distribution'].keys())
counts = list(analysis['year_distribution'].values())

bars = timeline_ax.bar(years, counts, color='#3498db', edgecolor='black', linewidth=2, width=0.6)
timeline_ax.set_xlabel('Publication Year', fontweight='bold', fontsize=12)
timeline_ax.set_ylabel('Number of Articles', fontweight='bold', fontsize=12)
timeline_ax.set_title('Publication Timeline', fontweight='bold', fontsize=13, pad=10)
timeline_ax.grid(axis='y', alpha=0.3, linestyle='--')
timeline_ax.set_ylim(0, max(counts) * 1.2)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    timeline_ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)} articles',
            ha='center', va='bottom', fontweight='bold', fontsize=11)

# Add a footer note
footer_text = f"""
📌 Source: Google Scholar via SerpAPI  |  Search Query: "polymer dissolution"  |  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
💡 Note: Low citation counts indicate very recent publications (2025-2026). These represent cutting-edge research in the field.
"""

fig.text(0.5, 0.01, footer_text, ha='center', fontsize=9, style='italic',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.2))

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save figure
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"./plots/polymer_dissolution_simple_{timestamp}.png"
plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')

print("=" * 70)
print("✅ SIMPLE VISUALIZATION CREATED")
print("=" * 70)
print(f"\n📊 File saved: {filename}")
print(f"\n💡 This single-panel view provides:")
print("   • Key statistics at a glance")
print("   • Top 5 most relevant articles")
print("   • Publication timeline")
print("   • Easy-to-read format")
print("\n🔍 View the visualization in the plots directory")
