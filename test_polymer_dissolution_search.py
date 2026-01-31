"""
Test search and visualization for polymer dissolution articles (2020-2025)
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from serpapi_scholar_client import GoogleScholarClient
from collections import Counter

# Configure matplotlib for non-interactive backend (for headless environments)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load environment variables
load_dotenv()

def search_polymer_dissolution():
    """Search for polymer dissolution articles from the last 5 years"""
    print("=" * 70)
    print("Polymer Dissolution Literature Search (2020-2025)")
    print("=" * 70)

    client = GoogleScholarClient()

    # Search for polymer dissolution articles
    print("\n🔍 Searching Google Scholar for polymer dissolution articles...")
    print("   Query: 'polymer dissolution'")
    print("   Year range: 2020-2025")
    print("   Max results: 20\n")

    articles = client.search(
        query='"polymer dissolution"',
        num_results=20,
        year_low=2020,
        year_high=2025,
        sort_by='date'
    )

    return articles


def parse_and_analyze(results):
    """Parse results and extract useful information"""
    print("📊 Analyzing results...\n")

    organic_results = results.get('organic_results', [])

    if not organic_results:
        print("❌ No results found!")
        return None

    # Parse all articles
    client = GoogleScholarClient()
    parsed_articles = []

    for result in organic_results:
        article = client._parse_article(result)
        parsed_articles.append(article)

    # Analysis
    analysis = {
        'total_articles': len(parsed_articles),
        'articles': parsed_articles,
        'year_distribution': {},
        'citation_stats': {
            'total_citations': 0,
            'avg_citations': 0,
            'max_citations': 0,
            'min_citations': float('inf')
        },
        'top_cited': [],
        'authors_frequency': {},
        'pdf_available': 0
    }

    # Year distribution
    year_counts = Counter()
    citation_totals = []

    for article in parsed_articles:
        year = article['year']
        if year != 'N/A' and year.isdigit():
            year_counts[int(year)] += 1

        citations = article['cited_by_count']
        citation_totals.append(citations)

        if article['pdf_link']:
            analysis['pdf_available'] += 1

        # Count authors
        for author in article['authors']:
            if author:
                analysis['authors_frequency'][author] = analysis['authors_frequency'].get(author, 0) + 1

    analysis['year_distribution'] = dict(sorted(year_counts.items()))

    # Citation statistics
    if citation_totals:
        analysis['citation_stats']['total_citations'] = sum(citation_totals)
        analysis['citation_stats']['avg_citations'] = sum(citation_totals) / len(citation_totals)
        analysis['citation_stats']['max_citations'] = max(citation_totals)
        analysis['citation_stats']['min_citations'] = min(citation_totals) if citation_totals else 0

    # Top cited articles
    analysis['top_cited'] = sorted(parsed_articles, key=lambda x: x['cited_by_count'], reverse=True)[:5]

    # Top authors
    analysis['top_authors'] = sorted(analysis['authors_frequency'].items(), key=lambda x: x[1], reverse=True)[:10]

    return analysis


def visualize_results(analysis):
    """Create visualizations of the search results"""
    print("📈 Creating visualizations...\n")

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Polymer Dissolution Literature Analysis (2020-2025)',
                 fontsize=16, fontweight='bold', y=0.98)

    # 1. Publications by Year
    ax1 = plt.subplot(2, 3, 1)
    years = list(analysis['year_distribution'].keys())
    counts = list(analysis['year_distribution'].values())

    colors = plt.cm.viridis(range(len(years)))
    bars = ax1.bar(years, counts, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Year', fontweight='bold')
    ax1.set_ylabel('Number of Articles', fontweight='bold')
    ax1.set_title('Publications per Year', fontweight='bold', pad=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')

    # 2. Citation Distribution
    ax2 = plt.subplot(2, 3, 2)
    citations = [a['cited_by_count'] for a in analysis['articles']]

    ax2.hist(citations, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
    ax2.axvline(analysis['citation_stats']['avg_citations'],
                color='red', linestyle='--', linewidth=2,
                label=f"Avg: {analysis['citation_stats']['avg_citations']:.1f}")
    ax2.set_xlabel('Citation Count', fontweight='bold')
    ax2.set_ylabel('Number of Articles', fontweight='bold')
    ax2.set_title('Citation Distribution', fontweight='bold', pad=10)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # 3. Top 5 Most Cited Articles
    ax3 = plt.subplot(2, 3, 3)
    top_titles = [a['title'][:40] + '...' if len(a['title']) > 40 else a['title']
                  for a in analysis['top_cited'][:5]]
    top_citations = [a['cited_by_count'] for a in analysis['top_cited'][:5]]

    y_pos = range(len(top_titles))
    colors_top = plt.cm.Reds([(c/max(top_citations) if max(top_citations) > 0 else 0)
                               for c in top_citations])

    ax3.barh(y_pos, top_citations, color=colors_top, edgecolor='black')
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(top_titles, fontsize=8)
    ax3.invert_yaxis()
    ax3.set_xlabel('Citations', fontweight='bold')
    ax3.set_title('Top 5 Most Cited Articles', fontweight='bold', pad=10)
    ax3.grid(axis='x', alpha=0.3, linestyle='--')

    # Add citation numbers
    for i, v in enumerate(top_citations):
        ax3.text(v + 1, i, str(v), va='center', fontweight='bold')

    # 4. PDF Availability
    ax4 = plt.subplot(2, 3, 4)
    pdf_data = [analysis['pdf_available'],
                analysis['total_articles'] - analysis['pdf_available']]
    labels = [f'PDF Available\n({analysis["pdf_available"]})',
              f'No PDF\n({analysis["total_articles"] - analysis["pdf_available"]})']
    colors_pie = ['#2ecc71', '#e74c3c']

    wedges, texts, autotexts = ax4.pie(pdf_data, labels=labels, colors=colors_pie,
                                        autopct='%1.1f%%', startangle=90,
                                        textprops={'fontweight': 'bold'})
    ax4.set_title('PDF Availability', fontweight='bold', pad=10)

    # 5. Top Authors
    ax5 = plt.subplot(2, 3, 5)
    if analysis['top_authors']:
        top_author_names = [name[:25] + '...' if len(name) > 25 else name
                           for name, _ in analysis['top_authors'][:8]]
        top_author_counts = [count for _, count in analysis['top_authors'][:8]]

        y_pos = range(len(top_author_names))
        colors_authors = plt.cm.Blues([(c/max(top_author_counts)) for c in top_author_counts])

        ax5.barh(y_pos, top_author_counts, color=colors_authors, edgecolor='black')
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels(top_author_names, fontsize=8)
        ax5.invert_yaxis()
        ax5.set_xlabel('Number of Papers', fontweight='bold')
        ax5.set_title('Top Contributing Authors', fontweight='bold', pad=10)
        ax5.grid(axis='x', alpha=0.3, linestyle='--')

        for i, v in enumerate(top_author_counts):
            ax5.text(v + 0.05, i, str(v), va='center', fontweight='bold')

    # 6. Summary Statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    summary_text = f"""
    📊 SUMMARY STATISTICS

    Total Articles Found: {analysis['total_articles']}
    Year Range: {min(analysis['year_distribution'].keys())} - {max(analysis['year_distribution'].keys())}

    📈 Citations:
    • Total: {analysis['citation_stats']['total_citations']:,}
    • Average: {analysis['citation_stats']['avg_citations']:.1f}
    • Maximum: {analysis['citation_stats']['max_citations']}
    • Minimum: {analysis['citation_stats']['min_citations']}

    📄 PDF Availability:
    • Available: {analysis['pdf_available']} ({analysis['pdf_available']/analysis['total_articles']*100:.1f}%)
    • Not Available: {analysis['total_articles'] - analysis['pdf_available']}

    👥 Unique Authors: {len(analysis['authors_frequency'])}

    🏆 Most Cited Article:
    {analysis['top_cited'][0]['title'][:60]}...
    ({analysis['top_cited'][0]['cited_by_count']} citations)
    """

    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
             family='monospace')

    plt.tight_layout()

    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"./plots/polymer_dissolution_analysis_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved: {filename}\n")

    return filename


def print_detailed_results(analysis):
    """Print detailed article information"""
    print("=" * 70)
    print("📚 DETAILED ARTICLE LISTING")
    print("=" * 70)
    print()

    for i, article in enumerate(analysis['articles'], 1):
        print(f"{i}. {article['title']}")
        print(f"   Authors: {', '.join(article['authors'][:4])}")
        if len(article['authors']) > 4:
            print(f"            ... and {len(article['authors']) - 4} more")
        print(f"   Year: {article['year']}")
        print(f"   Citations: {article['cited_by_count']}")
        if article['pdf_link']:
            print(f"   📄 PDF: {article['pdf_link']}")
        print(f"   🔗 Link: {article['link']}")
        print(f"   Snippet: {article['snippet'][:150]}...")
        print()

    print("=" * 70)
    print("🏆 TOP 5 MOST CITED ARTICLES")
    print("=" * 70)
    print()

    for i, article in enumerate(analysis['top_cited'][:5], 1):
        print(f"{i}. [{article['cited_by_count']} citations] {article['title']}")
        print(f"   {article['publication_info']}")
        print(f"   🔗 {article['link']}")
        print()


def save_results_json(analysis, results):
    """Save results to JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"./data/polymer_dissolution_search_{timestamp}.json"

    # Prepare data for JSON
    output = {
        'search_metadata': results.get('search_metadata', {}),
        'analysis': {
            'total_articles': analysis['total_articles'],
            'year_distribution': analysis['year_distribution'],
            'citation_stats': analysis['citation_stats'],
            'pdf_available': analysis['pdf_available'],
            'top_authors': analysis['top_authors'][:10]
        },
        'articles': analysis['articles']
    }

    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"✅ Results saved to: {filename}\n")
    return filename


def main():
    """Main execution"""
    # Search
    results = search_polymer_dissolution()

    # Analyze
    analysis = parse_and_analyze(results)

    if not analysis:
        return

    print(f"✅ Found {analysis['total_articles']} articles\n")

    # Print summary
    print("=" * 70)
    print("📊 QUICK SUMMARY")
    print("=" * 70)
    print(f"Total Articles: {analysis['total_articles']}")
    print(f"Year Range: {min(analysis['year_distribution'].keys())} - {max(analysis['year_distribution'].keys())}")
    print(f"Total Citations: {analysis['citation_stats']['total_citations']:,}")
    print(f"Average Citations: {analysis['citation_stats']['avg_citations']:.1f}")
    print(f"PDF Available: {analysis['pdf_available']} ({analysis['pdf_available']/analysis['total_articles']*100:.1f}%)")
    print(f"Unique Authors: {len(analysis['authors_frequency'])}")
    print()

    # Visualize
    viz_file = visualize_results(analysis)

    # Print detailed results
    print_detailed_results(analysis)

    # Save JSON
    json_file = save_results_json(analysis, results)

    print("=" * 70)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\n📊 Visualization: {viz_file}")
    print(f"💾 Data export: {json_file}")
    print(f"\n🔍 View visualization in plots directory")


if __name__ == "__main__":
    main()
