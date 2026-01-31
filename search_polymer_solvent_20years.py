"""
Search for general polymer and solvent research over the last 20 years
"""

import os
import json
import re
from datetime import datetime
from dotenv import load_dotenv
from serpapi_scholar_client import GoogleScholarClient
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

def search_polymer_solvent():
    """Search for polymer and solvent articles from 2005-2025"""
    print("=" * 70)
    print("Polymer and Solvent Research: Last 20 Years (2005-2025)")
    print("=" * 70)

    client = GoogleScholarClient()

    print("\n🔍 Searching Google Scholar...")
    print("   Query: 'polymer solvent'")
    print("   Year range: 2005-2025")
    print("   Max results: 20\n")

    results = client.search(
        query='polymer solvent',
        num_results=20,
        year_low=2005,
        year_high=2025,
        sort_by='date'  # Get recent articles
    )

    return results, client

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

def visualize_articles(results, client):
    """Create clean visualization of articles"""
    import textwrap

    organic_results = results.get('organic_results', [])

    if not organic_results:
        print("❌ No results found!")
        return

    # Parse articles
    articles = []
    for result in organic_results:
        article = client._parse_article(result)
        articles.append(article)

    print(f"✅ Found {len(articles)} articles\n")

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 14))
    ax.axis('off')

    # Title
    plt.title('Polymer and Solvent Research: Last 20 Years (2005-2025)',
              fontsize=16, fontweight='bold', pad=20)

    # Starting position
    y_pos = 0.97

    # Add each article
    for i, article in enumerate(articles, 1):
        title = article.get('title', 'N/A')
        # Wrap title instead of truncating
        wrapped_title = textwrap.fill(title, width=120)

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
                         edgecolor='#cccccc', linewidth=1),
                wrap=True)

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
    filename = f"./plots/polymer_solvent_20years_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')

    print(f"✅ Visualization saved: {filename}\n")

    # Save JSON
    json_filename = f"./data/polymer_solvent_20years_{timestamp}.json"
    with open(json_filename, 'w') as f:
        json.dump({
            'search_metadata': results.get('search_metadata', {}),
            'articles': articles
        }, f, indent=2)

    print(f"✅ Data saved: {json_filename}\n")

    return filename, articles

def print_summary(articles):
    """Print summary statistics"""
    print("=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)

    total = len(articles)
    years = [int(a['year']) for a in articles if a['year'] != 'N/A' and a['year'].isdigit()]
    citations = [a['cited_by_count'] for a in articles]
    with_pdf = sum(1 for a in articles if a['pdf_link'])

    print(f"\nTotal Articles: {total}")
    print(f"Year Range: {min(years) if years else 'N/A'} - {max(years) if years else 'N/A'}")
    print(f"Total Citations: {sum(citations):,}")
    print(f"Average Citations: {sum(citations)/len(citations):.1f}")
    print(f"Most Cited: {max(citations)}")
    print(f"PDF Available: {with_pdf} ({with_pdf/total*100:.0f}%)")

    print("\n🏆 TOP 3 MOST CITED:")
    top = sorted(articles, key=lambda x: x['cited_by_count'], reverse=True)[:3]
    for i, article in enumerate(top, 1):
        print(f"\n{i}. [{article['cited_by_count']} citations] {article['title'][:70]}...")
        print(f"   {article['publication_info'][:80]}")

def main():
    """Main execution"""
    results, client = search_polymer_solvent()
    filename, articles = visualize_articles(results, client)
    print_summary(articles)

    print("\n" + "=" * 70)
    print("✅ SEARCH COMPLETE")
    print("=" * 70)
    print(f"\n📊 Visualization: {filename}")
    print("\n💡 This search covers broader polymer-solvent research")
    print("   across 20 years, showing highly influential papers.")

if __name__ == "__main__":
    main()
