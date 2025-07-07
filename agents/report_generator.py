"""
Report Generator Agent for Co-Researcher.AI
===========================================

This agent creates comprehensive, structured reports from analyzed research papers.
It generates both HTML and JSON formats with visualizations and structured sections.

Features:
- Professional HTML reports with styling
- JSON data export for programmatic access
- Key insights and trends analysis
- Paper comparison and ranking
- Interactive elements and citations

Author: Co-Researcher.AI Team
"""

import json
import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import plotly.graph_objs as go
import plotly.io as pio

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates comprehensive research reports from analyzed papers
    
    This agent creates structured reports that summarize the research findings,
    provide insights, and present the information in a readable format.
    """
    
    def __init__(self, llm_client):
        """
        Initialize the report generator
        
        Args:
            llm_client: Instance of LLMClient for generating insights
        """
        self.llm_client = llm_client
        logger.info("📄 Report Generator initialized")
    
    def generate_report(
        self, 
        query: str, 
        papers: List[Dict[str, Any]], 
        summaries: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Generate a comprehensive research report
        
        Args:
            query (str): Original research query
            papers (List[Dict]): List of analyzed papers
            summaries (List[Dict]): List of paper summaries
            
        Returns:
            Dict[str, str]: Dictionary with 'html' and 'json' report paths
        """
        logger.info(f"📊 Generating research report for query: {query[:50]}...")
        
        try:
            # Generate report data structure
            report_data = self._create_report_data(query, papers, summaries)
            
            # Generate insights using LLM
            insights = self._generate_insights(query, papers, summaries)
            report_data['insights'] = insights
            
            # Create output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = "./outputs/reports"
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate JSON report
            json_path = os.path.join(output_dir, f"research_report_{timestamp}.json")
            self._save_json_report(report_data, json_path)
            
            # Generate HTML report
            html_path = os.path.join(output_dir, f"research_report_{timestamp}.html")
            self._generate_html_report(report_data, html_path)
            
            logger.info(f"✅ Report generated successfully")
            logger.info(f"   📄 HTML: {html_path}")
            logger.info(f"   📊 JSON: {json_path}")
            
            return {
                'html': html_path,
                'json': json_path,
                'timestamp': timestamp
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating report: {e}")
            raise
    
    def _create_report_data(
        self, 
        query: str, 
        papers: List[Dict[str, Any]], 
        summaries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create the structured report data
        
        Args:
            query (str): Research query
            papers (List[Dict]): Papers list
            summaries (List[Dict]): Summaries list
            
        Returns:
            Dict: Structured report data
        """
        # Basic report metadata
        report_data = {
            'metadata': {
                'query': query,
                'generated_at': datetime.now().isoformat(),
                'total_papers': len(papers),
                'research_scope': self._determine_research_scope(query),
                'time_range': self._get_time_range(papers)
            },
            'papers': [],
            'summary_statistics': self._calculate_summary_stats(papers, summaries),
            'key_themes': self._extract_key_themes(papers, summaries),
            'source_distribution': self._calculate_source_distribution(papers)
        }
        
        # Process each paper with its summary
        for i, paper in enumerate(papers):
            summary = summaries[i] if i < len(summaries) else {}
            
            paper_data = {
                'title': paper.get('title', 'Unknown'),
                'authors': paper.get('authors', []),
                'published_date': paper.get('published_date', ''),
                'source': paper.get('source', 'Unknown'),
                'url': paper.get('url', ''),
                'relevance_score': paper.get('relevance_score', 0),
                'summary': {
                    'objective': summary.get('objective', ''),
                    'methodology': summary.get('methodology', ''),
                    'key_findings': summary.get('key_findings', ''),
                    'significance': summary.get('significance', ''),
                    'limitations': summary.get('limitations', ''),
                    'future_directions': summary.get('future_directions', '')
                }
            }
            
            report_data['papers'].append(paper_data)
        
        return report_data
    
    def _generate_insights(
        self, 
        query: str, 
        papers: List[Dict[str, Any]], 
        summaries: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Generate key insights using LLM analysis
        
        Args:
            query (str): Research query
            papers (List[Dict]): Papers list
            summaries (List[Dict]): Summaries list
            
        Returns:
            Dict[str, str]: Generated insights
        """
        logger.info("🧠 Generating research insights...")
        
        # Prepare content for LLM analysis
        paper_summaries = []
        for i, paper in enumerate(papers[:5]):  # Limit to top 5 papers
            summary = summaries[i] if i < len(summaries) else {}
            
            paper_text = f"""
Title: {paper.get('title', 'Unknown')}
Authors: {', '.join(paper.get('authors', [])[:3])}
Key Findings: {summary.get('key_findings', 'Not specified')}
Methodology: {summary.get('methodology', 'Not specified')}
Significance: {summary.get('significance', 'Not specified')}
"""
            paper_summaries.append(paper_text)
        
        combined_content = "\n---\n".join(paper_summaries)
        
        # Improved prompt with clearer structure
        insights_prompt = f"""Analyze these research papers and provide key insights for the query: "{query}"

Papers Summary:
{combined_content}

Please provide insights in the following format. Use EXACTLY these headers:

## RESEARCH TRENDS
[2-3 sentences about main trends and directions]

## KEY FINDINGS
[2-3 sentences about most significant findings]

## METHODOLOGICAL APPROACHES
[2-3 sentences about methods and approaches used]

## RESEARCH GAPS
[2-3 sentences about gaps or limitations identified]

## FUTURE DIRECTIONS
[2-3 sentences about promising future research directions]

## PRACTICAL IMPLICATIONS
[2-3 sentences about real-world applications and implications]

Base your analysis ONLY on the provided papers."""
        
        try:
            insights_response = self.llm_client.generate_response(
                insights_prompt,
                temperature=0.3,
                max_tokens=1000
            )
            
            # Parse insights into categories
            insights = self._parse_insights_response(insights_response)
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return {
                'research_trends': 'Unable to generate trend analysis.',
                'key_findings': 'Unable to analyze key findings.',
                'methodological_approaches': 'Unable to analyze methodologies.',
                'research_gaps': 'Unable to identify research gaps.',
                'future_directions': 'Unable to identify future directions.',
                'practical_implications': 'Unable to assess practical implications.'
            }
    
    def _parse_insights_response(self, response: str) -> Dict[str, str]:
        """Parse the LLM insights response into structured categories"""
        insights = {}
        
        # Category mapping with flexible matching
        categories = {
            'research_trends': ['research trends', 'trends', 'research directions'],
            'key_findings': ['key findings', 'findings', 'main findings', 'results'],
            'methodological_approaches': ['methodological approaches', 'methodologies', 'methods', 'approaches'],
            'research_gaps': ['research gaps', 'gaps', 'limitations', 'challenges'],
            'future_directions': ['future directions', 'future work', 'future research'],
            'practical_implications': ['practical implications', 'implications', 'applications', 'impact']
        }
        
        # Split by markdown headers or double newlines
        import re
        
        # Try to split by markdown headers first (## HEADER)
        sections = re.split(r'\n##\s+', response)
        if len(sections) < 2:
            # Fallback: split by double newlines
            sections = response.split('\n\n')
        
        for section in sections:
            if not section.strip():
                continue
                
            lines = section.strip().split('\n')
            if not lines:
                continue
            
            # Get header and content
            header_line = lines[0].strip().lower()
            # Remove any markdown header symbols
            header_line = header_line.replace('#', '').strip()
            
            # Get content (everything after the first line)
            content_lines = lines[1:] if len(lines) > 1 else []
            content = ' '.join(content_lines).strip()
            
            # If no content after header, check if content is in the same line
            if not content and ':' in lines[0]:
                # Handle format like "## RESEARCH TRENDS: content here"
                parts = lines[0].split(':', 1)
                if len(parts) == 2:
                    header_line = parts[0].replace('#', '').strip().lower()
                    content = parts[1].strip()
            
            # Match categories flexibly
            for category, keywords in categories.items():
                if any(keyword in header_line for keyword in keywords):
                    if content:
                        insights[category] = content
                    break
        
        # Ensure all categories exist with meaningful fallbacks
        for category in categories.keys():
            if category not in insights or not insights[category]:
                category_name = category.replace('_', ' ').title()
                insights[category] = f"{category_name} analysis will be available in future reports."
        
        return insights
    
    def _determine_research_scope(self, query: str) -> str:
        """Determine the research scope from the query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['machine learning', 'deep learning', 'ai', 'neural']):
            return 'Artificial Intelligence & Machine Learning'
        elif any(word in query_lower for word in ['medical', 'clinical', 'disease', 'treatment']):
            return 'Biomedical & Health Sciences'
        elif any(word in query_lower for word in ['computer', 'algorithm', 'software', 'system']):
            return 'Computer Science & Engineering'
        else:
            return 'Multidisciplinary Research'
    
    def _get_time_range(self, papers: List[Dict[str, Any]]) -> Dict[str, str]:
        """Get the time range of the analyzed papers"""
        dates = []
        for paper in papers:
            date_str = paper.get('published_date', '')
            if date_str:
                try:
                    year = date_str[:4]
                    if year.isdigit():
                        dates.append(int(year))
                except:
                    continue
        
        if dates:
            return {
                'earliest': str(min(dates)),
                'latest': str(max(dates)),
                'span_years': str(max(dates) - min(dates) + 1)
            }
        
        return {'earliest': 'Unknown', 'latest': 'Unknown', 'span_years': 'Unknown'}
    
    def _calculate_summary_stats(
        self, 
        papers: List[Dict[str, Any]], 
        summaries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate summary statistics for the report"""
        return {
            'total_papers': len(papers),
            'unique_authors': len(set([
                author for paper in papers 
                for author in paper.get('authors', [])
            ])),
            'source_breakdown': self._calculate_source_distribution(papers),
            'avg_relevance_score': sum(
                paper.get('relevance_score', 0) for paper in papers
            ) / len(papers) if papers else 0
        }
    
    def _extract_key_themes(
        self, 
        papers: List[Dict[str, Any]], 
        summaries: List[Dict[str, Any]]
    ) -> List[str]:
        """Extract key themes from papers and summaries"""
        # Simple keyword extraction from titles and summaries
        all_text = []
        
        for paper in papers:
            all_text.append(paper.get('title', ''))
        
        for summary in summaries:
            all_text.append(summary.get('key_findings', ''))
        
        # Count common research terms
        research_terms = [
            'learning', 'algorithm', 'model', 'method', 'approach', 'system',
            'analysis', 'optimization', 'performance', 'accuracy', 'network',
            'framework', 'technique', 'evaluation', 'experimental'
        ]
        
        term_counts = {}
        combined_text = ' '.join(all_text).lower()
        
        for term in research_terms:
            count = combined_text.count(term)
            if count > 0:
                term_counts[term] = count
        
        # Return top themes
        sorted_themes = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)
        return [theme[0] for theme in sorted_themes[:5]]
    
    def _calculate_source_distribution(self, papers: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate distribution of papers by source"""
        sources = {}
        for paper in papers:
            source = paper.get('source', 'Unknown')
            sources[source] = sources.get(source, 0) + 1
        return sources
    
    def _save_json_report(self, report_data: Dict[str, Any], json_path: str):
        """Save the report data as JSON"""
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 JSON report saved: {json_path}")
        except Exception as e:
            logger.error(f"Error saving JSON report: {e}")
            raise
    
    def _generate_html_report(self, report_data: Dict[str, Any], html_path: str):
        """Generate an HTML report from the report data"""
        try:
            html_content = self._create_html_template(report_data)
            
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"📄 HTML report generated: {html_path}")
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
            raise
    
    def _create_html_template(self, data: Dict[str, Any]) -> str:
        """Create the HTML template for the report"""
        metadata = data['metadata']
        papers = data['papers']
        insights = data.get('insights', {})
        stats = data.get('summary_statistics', {})
        
        # Generate papers HTML
        papers_html = ""
        for i, paper in enumerate(papers, 1):
            papers_html += f"""
            <div class="paper-card">
                <h3>📄 {i}. {paper['title']}</h3>
                <div class="paper-meta">
                    <p><strong>Authors:</strong> {', '.join(paper['authors'][:3])}{'...' if len(paper['authors']) > 3 else ''}</p>
                    <p><strong>Published:</strong> {paper['published_date'][:10]} | <strong>Source:</strong> {paper['source']}</p>
                    <p><strong>Relevance Score:</strong> {paper.get('relevance_score', 'N/A')}</p>
                </div>
                
                <div class="summary-section">
                    <h4>🎯 Research Objective</h4>
                    <p style="text-align: justify; line-height: 1.6;">{paper['summary']['objective']}</p>
                    
                    <h4>🔬 Methodology</h4>
                    <p style="text-align: justify; line-height: 1.6;">{paper['summary']['methodology']}</p>
                    
                    <h4>💡 Key Findings</h4>
                    <p style="text-align: justify; line-height: 1.6;">{paper['summary']['key_findings']}</p>
                    
                    <h4>🌟 Significance & Impact</h4>
                    <p style="text-align: justify; line-height: 1.6;">{paper['summary']['significance']}</p>
                    
                    <h4>⚠️ Limitations</h4>
                    <p style="text-align: justify; line-height: 1.6;">{paper['summary']['limitations']}</p>
                    
                    <h4>🚀 Future Directions</h4>
                    <p style="text-align: justify; line-height: 1.6;">{paper['summary']['future_directions']}</p>
                </div>
                
                <div class="paper-links">
                    <a href="{paper['url']}" target="_blank">📖 View Paper</a>
                </div>
            </div>
            """
        
        # Create the complete HTML
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Research Report - Co-Researcher.AI</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f7fa;
                    color: #333;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                    border-radius: 10px;
                    overflow: hidden;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    text-align: center;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 2.5em;
                    font-weight: 300;
                }}
                .meta-info {{
                    padding: 20px 30px;
                    background: #f8f9fa;
                    border-bottom: 1px solid #dee2e6;
                }}
                .meta-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin-top: 15px;
                }}
                .meta-item {{
                    text-align: center;
                    padding: 15px;
                    background: white;
                    border-radius: 8px;
                    border: 1px solid #e9ecef;
                }}
                .insights-section {{
                    padding: 30px;
                    background: #fff;
                }}
                .insight-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin-top: 20px;
                }}
                .insight-card {{
                    padding: 20px;
                    background: #f8f9fa;
                    border-left: 4px solid #667eea;
                    border-radius: 5px;
                }}
                .papers-section {{
                    padding: 30px;
                    background: #fff;
                }}
                .paper-card {{
                    margin-bottom: 30px;
                    padding: 25px;
                    border: 1px solid #dee2e6;
                    border-radius: 10px;
                    background: #fafbfc;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .paper-card h3 {{
                    color: #495057;
                    margin-top: 0;
                    border-bottom: 2px solid #e9ecef;
                    padding-bottom: 10px;
                    font-size: 1.3em;
                }}
                .paper-meta {{
                    background: white;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 15px 0;
                    border-left: 4px solid #667eea;
                }}
                .summary-section {{
                    margin: 20px 0;
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    border-left: 3px solid #28a745;
                }}
                .summary-section h4 {{
                    color: #667eea;
                    margin-top: 20px;
                    margin-bottom: 15px;
                    font-size: 1.1em;
                    border-bottom: 1px solid #e9ecef;
                    padding-bottom: 5px;
                }}
                .summary-section h4:first-child {{
                    margin-top: 0;
                }}
                .summary-section p {{
                    margin-bottom: 20px;
                    color: #333;
                    font-size: 0.95em;
                }}
                .paper-links {{
                    text-align: right;
                    margin-top: 15px;
                }}
                .paper-links a {{
                    background: #667eea;
                    color: white;
                    padding: 8px 15px;
                    text-decoration: none;
                    border-radius: 5px;
                    font-size: 0.9em;
                }}
                .paper-links a:hover {{
                    background: #5a6fd8;
                }}
                .footer {{
                    text-align: center;
                    padding: 20px;
                    background: #f8f9fa;
                    color: #6c757d;
                    font-size: 0.9em;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <header class="header">
                    <h1>🧠 Co-Researcher.AI</h1>
                    <h2>Research Report</h2>
                    <p style="font-size: 1.2em; margin: 10px 0 0 0;">Query: "{metadata['query']}"</p>
                </header>
                
                <div class="meta-info">
                    <h3>📊 Report Overview</h3>
                    <div class="meta-grid">
                        <div class="meta-item">
                            <h4>{metadata['total_papers']}</h4>
                            <p>Papers Analyzed</p>
                        </div>
                        <div class="meta-item">
                            <h4>{metadata['research_scope']}</h4>
                            <p>Research Scope</p>
                        </div>
                        <div class="meta-item">
                            <h4>{metadata['time_range']['earliest']} - {metadata['time_range']['latest']}</h4>
                            <p>Time Range</p>
                        </div>
                        <div class="meta-item">
                            <h4>{metadata['generated_at'][:16]}</h4>
                            <p>Generated At</p>
                        </div>
                    </div>
                </div>
                
                <section class="insights-section">
                    <h2>🔍 Key Insights</h2>
                    <div class="insight-grid">
                        <div class="insight-card">
                            <h4>📈 Research Trends</h4>
                            <p>{insights.get('research_trends', 'Not available')}</p>
                        </div>
                        <div class="insight-card">
                            <h4>💡 Key Findings</h4>
                            <p>{insights.get('key_findings', 'Not available')}</p>
                        </div>
                        <div class="insight-card">
                            <h4>🔬 Methodologies</h4>
                            <p>{insights.get('methodological_approaches', 'Not available')}</p>
                        </div>
                        <div class="insight-card">
                            <h4>🚀 Future Directions</h4>
                            <p>{insights.get('future_directions', 'Not available')}</p>
                        </div>
                    </div>
                </section>
                
                <section class="papers-section">
                    <h2>📚 Analyzed Papers</h2>
                    {papers_html}
                </section>
                
                <footer class="footer">
                    <p>Generated by Co-Researcher.AI | {metadata['generated_at'][:10]}</p>
                </footer>
            </div>
        </body>
        </html>
        """
        
        return html_template 