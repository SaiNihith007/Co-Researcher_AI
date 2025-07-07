#!/usr/bin/env python3
"""
Co-Researcher.AI - Intelligent Research Assistant
Main Streamlit Application
"""

import streamlit as st
import os
import logging
from datetime import datetime
import json
import uuid

# Import our modules
from config import load_config
from router import QueryRouter
from modules.llm_client import LLMClient
from modules.paper_reranker import PaperReranker
from modules.vector_store import VectorStore
from agents.data_collection_agent import DataCollectionAgent
from agents.summarization_agent import SummarizationAgent
from agents.report_generator import ReportGenerator
from agents.rag_chat_agent import RAGChatAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Co-Researcher.AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'research_history' not in st.session_state:
    st.session_state.research_history = []
if 'current_chat_context' not in st.session_state:
    st.session_state.current_chat_context = None
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'current_session_id' not in st.session_state:
    st.session_state.current_session_id = None
if 'validated_api_key' not in st.session_state:
    st.session_state.validated_api_key = None
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "gpt-3.5-turbo"
if 'api_key_last_validated' not in st.session_state:
    st.session_state.api_key_last_validated = None

def validate_api_key(api_key: str) -> bool:
    """
    Simple API key validation
    
    Args:
        api_key (str): API key to validate
        
    Returns:
        bool: True if valid format
    """
    if not api_key or not isinstance(api_key, str):
        return False
    
    # OpenAI API keys start with 'sk-'
    if not api_key.startswith('sk-'):
        return False
    
    # Check reasonable length
    if len(api_key) < 20:
        return False
    
    return True

def main():
    """Main application interface"""
    
    # Run startup cleanup on first load
    if 'startup_cleanup_done' not in st.session_state:
        with st.spinner("🧹 Cleaning up old research sessions..."):
            try:
                from modules.vector_store import VectorStore
                VectorStore.cleanup_old_sessions()
                st.session_state.startup_cleanup_done = True
                logger.info("✅ Startup cleanup completed")
            except Exception as e:
                logger.error(f"❌ Startup cleanup failed: {e}")
                st.session_state.startup_cleanup_done = True  # Don't retry every time
    
    # Sidebar configuration
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "OpenAI API Key", 
            value=st.session_state.validated_api_key if st.session_state.validated_api_key else "",
            type="password",
            help="Get your API key from https://platform.openai.com/api-keys"
        )
        
        # Validate and store API key
        if api_key:
            if validate_api_key(api_key):
                st.success("✅ API Key format looks valid")
                
                # Store validated API key in session state only if it's different or first time
                if st.session_state.validated_api_key != api_key:
                    st.session_state.validated_api_key = api_key
                    st.session_state.api_key_last_validated = datetime.now().isoformat()
                    
                    # Clear chat context when API key changes to ensure new LLM client
                    if st.session_state.current_chat_context:
                        st.session_state.current_chat_context = None
                        st.info("🔄 Chat context refreshed with new API key")
            else:
                st.error("❌ Invalid API key format")
                st.warning("⚠️ API key should start with 'sk-' and be around 51 characters")
                
                # Clear corrupted API key from session state
                if st.session_state.validated_api_key:
                    st.session_state.validated_api_key = None
                    st.session_state.api_key_last_validated = None
                    st.warning("🚨 Cleared potentially corrupted API key from session")
                
                api_key = None  # Don't use invalid API key
        
        # Model selection
        model = st.selectbox(
            "Model",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
            index=0,
            help="Choose the AI model to use"
        )
        
        # Store selected model in session state
        st.session_state.selected_model = model
        
        # Research parameters
        max_papers = st.slider("Max Papers", 3, 20, 5, help="Maximum number of papers to analyze")
        years_back = st.slider("Years Back", 1, 10, 3, help="How many years back to search")
        
        # Display session info
        if st.session_state.validated_api_key:
            st.info(f"🔑 API Key: Valid (last validated: {st.session_state.api_key_last_validated[:19] if st.session_state.api_key_last_validated else 'Unknown'})")
        else:
            st.warning("🔑 No valid API key stored")
    
    # Main content area
    st.title("🔬 Co-Researcher.AI")
    st.markdown("*Your intelligent research assistant powered by multi-agent AI*")
    st.markdown("<div style='text-align: center; color: #666; font-size: 0.8em; margin-top: -10px; margin-bottom: 20px;'>Project by Sai Nihith Immaneni</div>", unsafe_allow_html=True)
    
    # Create tabs for different interfaces
    tab1, tab2, tab3 = st.tabs(["🔍 Research", "💬 Chat", "📚 History"])
    
    with tab1:
        research_interface(
            api_key=st.session_state.validated_api_key,  # Use validated API key from session
            model=st.session_state.selected_model,
            max_papers=max_papers,
            years_back=years_back
        )
    
    with tab2:
        chat_interface(
            api_key=st.session_state.validated_api_key,  # Use validated API key from session
            model=st.session_state.selected_model
        )
    
    with tab3:
        history_interface()

def research_interface(api_key, model, max_papers, years_back):
    """Research query interface"""
    
    st.header("Ask a Research Question")
    
    # Check for valid API key
    if not api_key or not validate_api_key(api_key):
        st.warning("⚠️ Please enter a valid OpenAI API key in the sidebar to continue")
        st.info("💡 You can get an API key from [OpenAI's platform](https://platform.openai.com/api-keys)")
        return
    
    # Check if there are existing results to display
    if 'current_results' in st.session_state:
        results = st.session_state.current_results
        st.info(f"🔄 Showing results for: \"{results['query']}\" (Generated: {results['timestamp'][:19]})")
        
        # Option to start new research
        if st.button("🆕 Start New Research"):
            del st.session_state.current_results
            st.rerun()
        
        # Display existing results
        display_research_results(
            results['query'], 
            results['papers'], 
            results['summaries'], 
            results['report_paths']
        )
        return
    
    # Research query input
    query = st.text_area(
        "Enter your research question:",
        height=100,
        placeholder="Example: Latest advances in cancer treatment using machine learning",
        help="Ask about recent developments, compare approaches, or explore specific research areas"
    )
    
    # Start research button
    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("🚀 Start Research", type="primary", disabled=not query.strip()):
            if query.strip():
                # Validate API key one more time before starting
                if not validate_api_key(api_key):
                    st.error("🚨 API key validation failed! Please check your API key.")
                    st.session_state.validated_api_key = None  # Clear corrupted key
                    st.rerun()
                    return
                
                # Run the research pipeline
                run_research_pipeline(query.strip(), api_key, model, max_papers, years_back)
    
    with col2:
        if st.button("💡 Example Query"):
            example_queries = [
                "Latest advances in cancer treatment using machine learning",
                "Recent developments in quantum computing algorithms", 
                "Current research on renewable energy storage solutions",
                "New approaches to drug discovery using AI",
                "Recent progress in autonomous vehicle safety"
            ]
            import random
            st.session_state.example_query = random.choice(example_queries)
            st.rerun()
    
    # Show example query if generated
    if hasattr(st.session_state, 'example_query'):
        st.info(f"💡 Example: {st.session_state.example_query}")
        if st.button("Use This Example"):
            query = st.session_state.example_query
            del st.session_state.example_query
            # Run research with example query
            if validate_api_key(api_key):
                run_research_pipeline(query, api_key, model, max_papers, years_back)
            else:
                st.error("🚨 API key validation failed!")
    
    # Research tips
    with st.expander("💡 Research Tips"):
        st.markdown("""
        **For best results:**
        - Be specific about your research area
        - Include keywords like "latest", "recent", "advances"
        - Mention specific techniques or applications
        - Ask comparative questions (e.g., "compare X vs Y approaches")
        
        **Example queries:**
        - "Latest transformer architectures for natural language processing"
        - "Recent advances in federated learning privacy preservation"
        - "Current methods for few-shot learning in computer vision"
        """)
    
    # Show research history if available
    if st.session_state.research_history:
        st.subheader("🕐 Recent Research")
        for i, entry in enumerate(st.session_state.research_history[-3:]):  # Show last 3
            with st.expander(f"{entry['timestamp'][:19]} - {entry['query'][:50]}..."):
                st.write(f"**Query:** {entry['query']}")
                st.write(f"**Papers found:** {entry['papers_count']}")
                st.write(f"**Model used:** {entry['model']}")
                if st.button(f"View Report {i}", key=f"view_report_{i}"):
                    # Load and display this report
                    st.info("Report viewing coming soon!")

def run_research_pipeline(query, api_key, model, max_papers, years_back):
    """Execute the complete research pipeline"""
    
    # Final API key validation before starting pipeline
    if not api_key or not validate_api_key(api_key):
        st.error("🚨 Invalid API key! Cannot start research pipeline.")
        st.session_state.validated_api_key = None  # Clear corrupted key
        st.rerun()
        return
    
    # Create a clean copy of the API key to prevent corruption
    # This ensures the original API key string is preserved
    clean_api_key = str(api_key).strip()
    
    # Validate the clean copy
    if not validate_api_key(clean_api_key):
        st.error("🚨 API key corruption detected during pipeline initialization!")
        st.session_state.validated_api_key = None
        st.rerun()
        return
    
    # Clear existing results and chat context when starting new research
    logger.info("🧹 Clearing previous session data for new research")
    if 'current_results' in st.session_state:
        del st.session_state.current_results
    st.session_state.current_chat_context = None
    st.session_state.vector_store = None
    st.session_state.current_session_id = None
    st.session_state.chat_messages = []  # Clear chat history for new session
    
    # Progress tracking
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    
    try:
        status_text.text("🧭 Initializing AI agents...")
        progress_bar.progress(0.05)
        
        # Initialize components with validated API key
        try:
            llm_client = LLMClient(api_key, model)
            router = QueryRouter(llm_client)
        except Exception as e:
            st.error(f"❌ Failed to initialize LLM client: {e}")
            if "api_key" in str(e).lower():
                st.error("🚨 This appears to be an API key issue.")
                st.session_state.validated_api_key = None
            return
        
        status_text.text("🤔 Analyzing query type...")
        progress_bar.progress(0.10)
        
        # Classify the query
        try:
            query_type = router.classify_query(query)
        except Exception as e:
            st.error(f"❌ Query classification failed: {e}")
            if "api_key" in str(e).lower() or "unauthorized" in str(e).lower():
                st.error("🚨 API key validation failed during query classification!")
                st.session_state.validated_api_key = None
            return
        
        if query_type == "general":
            # Handle general questions directly
            st.info(f"🤖 This looks like a general question. Getting direct answer...")
            
            try:
                answer = router.generate_direct_answer(query)
                
                # Display the answer prominently
                st.markdown("### 🤖 AI Assistant Response")
                st.write(answer)
                
                # Show helpful tips
                with st.expander("💡 Tips for Research Questions"):
                    st.markdown("""
                    For research-based analysis, try questions like:
                    - "Latest developments in [your topic]"
                    - "Recent advances in [specific field]"
                    - "Compare current approaches to [problem]"
                    - "What are the new methods for [application]"
                    """)
                
                # Option to convert to research question
                st.markdown("---")
                if st.button("🔬 Convert to Research Question"):
                    research_query = f"Latest research on {query}"
                    st.info(f"🔄 Converting to: '{research_query}'")
                    run_research_pipeline(research_query, api_key, model, max_papers, years_back)
                    return
                
            except Exception as e:
                st.error(f"❌ Error generating answer: {e}")
                if "api_key" in str(e).lower():
                    st.error("🚨 API key issue detected!")
                    st.session_state.validated_api_key = None
            
            progress_bar.progress(1.0)
            status_text.text("✅ General question answered!")
            return
        
        # Research pipeline for research questions
        status_text.text("🔍 Collecting research papers...")
        progress_bar.progress(0.20)
        
        # Initialize research agents with validated API key
        try:
            data_agent = DataCollectionAgent(years_back=years_back, router=router, enable_full_papers=True)
            reranker = PaperReranker(llm_client)
            summarization_agent = SummarizationAgent(llm_client)
            report_generator = ReportGenerator(llm_client)
        except Exception as e:
            st.error(f"❌ Failed to initialize research agents: {e}")
            if "api_key" in str(e).lower():
                st.error("🚨 API key validation failed during agent initialization!")
                st.session_state.validated_api_key = None
            return
        
        # Step 1: Collect papers (abstract-only for fast ranking)
        papers = data_agent.collect_papers(query, max_papers * 3, process_full_content=False)  # Faster collection for ranking
        if not papers:
            st.warning("⚠️ No papers found for your query. Try rephrasing or using different keywords.")
            progress_bar.progress(1.0)
            status_text.text("❌ No papers found")
            return
        
        status_text.text(f"📊 Ranking {len(papers)} papers by relevance...")
        progress_bar.progress(0.40)
        
        # Step 2: Rank papers (using abstracts)
        try:
            ranked_papers = reranker.rank_papers(query, papers, top_k=max_papers)
        except Exception as e:
            st.error(f"❌ Paper ranking failed: {e}")
            if "api_key" in str(e).lower():
                st.error("🚨 API key issue during paper ranking!")
                st.session_state.validated_api_key = None
            return
        
        # Step 2.5: Now process full content for ONLY the top-ranked papers
        status_text.text("📄 Processing full content for selected papers...")
        progress_bar.progress(0.50)
        
        try:
            # Process full content for the selected papers only
            enhanced_papers = data_agent.process_full_papers(ranked_papers)
            ranked_papers = enhanced_papers  # Replace with enhanced versions
            logger.info(f"✅ Full content processed for {len(ranked_papers)} selected papers")
        except Exception as e:
            logger.warning(f"⚠️ Full content processing failed: {e}")
            # Continue with abstract-only papers
            pass
        
        status_text.text("📝 Generating AI summaries...")
        progress_bar.progress(0.65)
        
        # Step 3: Generate summaries
        summaries = []
        for i, paper in enumerate(ranked_papers):
            try:
                # Validate API key before each summary to catch corruption early
                if not validate_api_key(llm_client.api_key):
                    st.error(f"🚨 API key corruption detected during summarization of paper {i+1}!")
                    st.session_state.validated_api_key = None
                    return
                
                summary = summarization_agent.summarize_paper(paper)
                summaries.append(summary)
                
                # Update progress
                summary_progress = 0.65 + (0.20 * (i + 1) / len(ranked_papers))
                progress_bar.progress(summary_progress)
                status_text.text(f"📝 Summarizing paper {i+1}/{len(ranked_papers)}...")
                
            except Exception as e:
                st.error(f"❌ Failed to summarize paper {i+1}: {e}")
                if "api_key" in str(e).lower() or "unauthorized" in str(e).lower():
                    st.error(f"🚨 API key issue during summarization of paper {i+1}!")
                    st.session_state.validated_api_key = None
                    return
                
                # Create fallback summary for failed papers
                fallback_summary = {
                    'objective': f"Research focus: {paper.get('title', 'Unknown')}",
                    'methodology': "Methodology details not available due to processing error",
                    'key_findings': "Key findings not available due to processing error", 
                    'significance': "Significance not available due to processing error",
                    'limitations': "Limitations not available due to processing error",
                    'future_directions': "Future directions not available due to processing error",
                    'error': str(e)
                }
                summaries.append(fallback_summary)
        
        status_text.text("📄 Generating comprehensive report...")
        progress_bar.progress(0.88)
        
        # Step 4: Generate report
        try:
            # Final API key validation before report generation
            if not validate_api_key(llm_client.api_key):
                st.error("🚨 API key corruption detected before report generation!")
                st.session_state.validated_api_key = None
                return
            
            report_paths = report_generator.generate_report(query, ranked_papers, summaries)
        except Exception as e:
            st.error(f"❌ Report generation failed: {e}")
            if "api_key" in str(e).lower():
                st.error("🚨 API key issue during report generation!")
                st.session_state.validated_api_key = None
            return
        
        # Step 5: Set up RAG for chat
        status_text.text("🧠 Setting up chat functionality...")
        
        try:
            # Generate unique session ID for this research session
            session_id = str(uuid.uuid4())[:8]  # Short unique ID
            st.session_state.current_session_id = session_id
            
            # Create a NEW session-based vector store (isolated from previous sessions)
            from modules.vector_store import VectorStore
            st.session_state.vector_store = VectorStore(session_id=session_id)
            
            logger.info(f"🔒 Created isolated vector store for session: {session_id}")
            
            # Add papers to the NEW session vector store for RAG (enhanced with full content)
            st.session_state.vector_store.add_papers(ranked_papers, summaries)
            
            # Initialize RAG chat agent with validated API key
            if validate_api_key(api_key):
                validated_llm_client = LLMClient(api_key, model)  # Create fresh client with validated key
                st.session_state.current_chat_context = RAGChatAgent(
                    llm_client=validated_llm_client,
                    vector_store=st.session_state.vector_store
                )
                
                logger.info(f"✅ RAG chat initialized for session {session_id} with {len(ranked_papers)} papers")
            else:
                st.error("🚨 API key validation failed during RAG setup!")
                st.session_state.validated_api_key = None
                
        except Exception as e:
            st.warning(f"⚠️ Chat setup failed: {e}")
            # Continue without chat functionality
        
        # Success! Store results and display
        progress_bar.progress(1.0)
        status_text.text("✅ Research complete!")
        
        # Store results in session state
        st.session_state.current_results = {
            'query': query,
            'papers': ranked_papers,
            'summaries': summaries,
            'report_paths': report_paths,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add to research history
        st.session_state.research_history.append({
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'papers_count': len(ranked_papers),
            'model': model
        })
        
        st.success("🎉 Research analysis complete!")
        
        # Display results
        display_research_results(query, ranked_papers, summaries, report_paths)
        
    except Exception as e:
        st.error(f"❌ Pipeline failed: {e}")
        if "api_key" in str(e).lower() or "unauthorized" in str(e).lower():
            st.error("🚨 This appears to be an API key related error!")
            st.session_state.validated_api_key = None
            st.rerun()
        
        # Clean up on failure
        progress_bar.progress(0.0)
        status_text.text("❌ Research failed")

def display_research_results(query, papers, summaries, report_paths):
    """Display the research results"""
    
    st.success("🎉 Research Complete!")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Papers Analyzed", len(papers))
    with col2:
        sources = list(set(p.get('source', 'Unknown') for p in papers))
        st.metric("Sources", len(sources))
    with col3:
        years = [p.get('published_date', '')[:4] for p in papers if p.get('published_date')]
        year_range = f"{min(years)}-{max(years)}" if years else "Various"
        st.metric("Year Range", year_range)
    with col4:
        # Calculate average relevance with proper fallback
        relevance_scores = []
        for i, p in enumerate(papers):
            score = p.get('relevance_score', len(papers) - i)
            relevance_scores.append(score)
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        st.metric("Avg Relevance", f"{avg_relevance:.2f}")
    
    # Report download
    st.subheader("📄 Research Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # HTML report
        try:
            with open(report_paths['html'], 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            st.download_button(
                label="📄 Download HTML Report",
                data=html_content,
                file_name=f"research_report_{report_paths['timestamp']}.html",
                mime="text/html",
                key=f"download_html_{report_paths['timestamp']}"
            )
        except Exception as e:
            st.error(f"Error loading HTML report: {e}")
    
    with col2:
        # JSON data
        try:
            with open(report_paths['json'], 'r', encoding='utf-8') as f:
                json_content = f.read()
            
            st.download_button(
                label="📊 Download JSON Data",
                data=json_content,
                file_name=f"research_data_{report_paths['timestamp']}.json",
                mime="application/json",
                key=f"download_json_{report_paths['timestamp']}"
            )
        except Exception as e:
            st.error(f"Error loading JSON data: {e}")
    
    # Paper summaries
    st.subheader("📚 Paper Summaries")
    
    for i, (paper, summary) in enumerate(zip(papers, summaries), 1):
        with st.expander(f"📄 {i}. {paper['title'][:80]}..."):
            
            # Paper metadata
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Authors:** {', '.join(paper['authors'][:3])}{'...' if len(paper['authors']) > 3 else ''}")
                st.write(f"**Published:** {paper['published_date'][:10]}")
                st.write(f"**Source:** {paper['source']}")
            
            with col2:
                # Get relevance score with better fallback
                relevance_score = paper.get('relevance_score', len(papers) - i)
                st.metric("Relevance Score", f"{relevance_score:.2f}")
                
                # Show embedding score if available
                if paper.get('embedding_score', 0) > 0:
                    st.metric("Embedding Score", f"{paper.get('embedding_score', 0):.3f}")
                
                if paper.get('url'):
                    st.link_button("📖 View Paper", paper['url'])
            
            # Summary sections
            st.write("**🎯 Objective:**", summary.get('objective', 'Not specified'))
            st.write("**🔬 Methodology:**", summary.get('methodology', 'Not specified'))
            st.write("**💡 Key Findings:**", summary.get('key_findings', 'Not specified'))
            st.write("**🌟 Significance:**", summary.get('significance', 'Not specified'))

def chat_interface(api_key, model):
    """Chat interface for follow-up questions"""
    
    st.header("💬 Chat with Research Papers")
    
    # Check for valid API key
    if not api_key or not validate_api_key(api_key):
        st.warning("⚠️ Please enter a valid OpenAI API key in the sidebar to enable chat")
        st.info("💡 You can get an API key from [OpenAI's platform](https://platform.openai.com/api-keys)")
        return
    
    # Check if we need to refresh chat context due to API key changes
    if (st.session_state.current_chat_context and 
        hasattr(st.session_state.current_chat_context, 'llm_client') and
        hasattr(st.session_state.current_chat_context.llm_client, 'api_key') and
        st.session_state.current_chat_context.llm_client.api_key != api_key):
        
        st.info("🔄 API key changed. Updating chat context...")
        
        # Recreate the chat context with new API key if vector store exists
        if st.session_state.vector_store:
            try:
                from modules.llm_client import LLMClient
                from agents.rag_chat_agent import RAGChatAgent
                
                # Validate API key before creating new client
                if validate_api_key(api_key):
                    llm_client = LLMClient(api_key, model)
                    st.session_state.current_chat_context = RAGChatAgent(
                        llm_client=llm_client,
                        vector_store=st.session_state.vector_store
                    )
                    st.success("✅ Chat context updated successfully")
                else:
                    st.error("🚨 API key validation failed during context refresh")
                    st.session_state.current_chat_context = None
                    return
                    
            except Exception as e:
                st.error(f"❌ Error refreshing chat context: {e}")
                st.session_state.current_chat_context = None
                return
    
    # Check if there are papers available for chat
    if not st.session_state.current_chat_context:
        st.info("📝 No research papers loaded for chat. Please run a research query first in the Research tab.")
        
        # Show quick start option
        with st.expander("🚀 Quick Start - Run Research First"):
            st.markdown("""
            To chat with research papers, you need to:
            1. Go to the **Research** tab
            2. Enter a research question
            3. Wait for the analysis to complete
            4. Come back to this chat tab
            
            The system will then load all analyzed papers into a searchable knowledge base 
            that you can ask questions about.
            
            **🔒 Session Isolation**: Each research session creates its own isolated chat context. 
            You can only ask questions about papers from your current research session.
            """)
        return
    
    # Display session info with enhanced details
    if st.session_state.current_session_id and st.session_state.vector_store:
        vector_stats = st.session_state.vector_store.get_stats()
        cleanup_info = vector_stats.get('cleanup_info', {})
        
        # Main session info
        st.info(f"""
        🔒 **Session-Based Chat Active**
        - Session ID: `{st.session_state.current_session_id}`
        - Papers available: {vector_stats.get('unique_papers', 0)}
        - Document chunks: {vector_stats.get('total_documents', 0)}
        - Status: {vector_stats.get('session_info', 'Unknown')}
        """)
        
        # Cleanup info in expandable section
        with st.expander("🧹 Session Cleanup Information"):
            if cleanup_info.get('cleanup_enabled'):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Sessions", cleanup_info.get('total_sessions', 0))
                with col2:
                    st.metric("Old Sessions", cleanup_info.get('old_sessions_ready_for_cleanup', 0))
                with col3:
                    st.metric("Storage Used", f"{cleanup_info.get('total_storage_size_mb', 0)} MB")
                
                st.markdown(f"""
                **Cleanup Policy**: Sessions older than {cleanup_info.get('cleanup_age_threshold_hours', 6)} hours are automatically deleted.
                
                **How it works**:
                - Each research query creates a new isolated session folder
                - Vector embeddings are stored separately per session 
                - Old sessions are cleaned up automatically to save disk space
                - Your current session is safe until it expires
                """)
                
                if cleanup_info.get('old_sessions_ready_for_cleanup', 0) > 0:
                    st.warning(f"⚠️ {cleanup_info.get('old_sessions_ready_for_cleanup')} old sessions ready for cleanup on next app restart")
                else:
                    st.success("✅ No old sessions to clean up")
            else:
                st.info("Session cleanup not applicable for shared vector store")
        
        st.info("💡 **Note**: This chat only knows about papers from your current research session. Start new research to analyze different papers.")
    else:
        st.warning("⚠️ Session information not available")
    
    st.info(f"🔑 Using API Key: {api_key[:20]}... (Validated: {st.session_state.api_key_last_validated[:19] if st.session_state.api_key_last_validated else 'Unknown'})")
    
    # Chat interface
    st.subheader("Ask Questions About Your Research")
    
    # Show suggested questions
    with st.expander("💡 Suggested Questions"):
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("What are the main findings across all papers?"):
                user_question = "What are the main findings across all papers?"
                st.session_state.chat_question = user_question
                st.rerun()
        
        with col2:
            if st.button("Compare the methodologies used"):
                user_question = "Compare the methodologies used in the different papers"
                st.session_state.chat_question = user_question
                st.rerun()
        
        with col1:
            if st.button("What are the limitations mentioned?"):
                user_question = "What are the limitations mentioned in the research?"
                st.session_state.chat_question = user_question
                st.rerun()
        
        with col2:
            if st.button("What future research is suggested?"):
                user_question = "What future research directions are suggested?"
                st.session_state.chat_question = user_question
                st.rerun()
    
    # Question input
    question = st.text_input(
        "Ask a question about the research papers:",
        value=getattr(st.session_state, 'chat_question', ''),
        placeholder="e.g., What are the key differences between the approaches?"
    )
    
    # Clear the stored question after displaying it
    if hasattr(st.session_state, 'chat_question'):
        del st.session_state.chat_question
    
    # Answer question
    if st.button("💬 Get Answer", type="primary") and question:
        
        # Validate API key before processing
        if not validate_api_key(api_key):
            st.error("🚨 API key validation failed! Please check your API key in the sidebar.")
            st.session_state.validated_api_key = None  # Clear corrupted key
            st.rerun()
            return
        
        with st.spinner("🔍 Searching through research papers..."):
            try:
                response = st.session_state.current_chat_context.answer_question(question)
                
                # Display answer
                st.markdown("### 🤖 Answer")
                st.write(response['answer'])
                
                # Display sources
                if response['sources']:
                    st.markdown("### 📚 Sources")
                    for i, source in enumerate(response['sources'], 1):
                        with st.expander(f"📄 {i}. {source['title']}"):
                            st.write(f"**Authors:** {', '.join(source['authors'][:3])}")
                            st.write(f"**Published:** {source['published_date'][:10]}")
                            st.write(f"**URL:** {source['url']}")
                
                # Add to chat history
                st.session_state.chat_messages.append({
                    'question': question,
                    'answer': response['answer'],
                    'timestamp': response['timestamp'],
                    'sources_count': len(response['sources'])
                })
                
            except Exception as e:
                st.error(f"❌ Error processing question: {e}")
                # Check if it's an API key related error
                if "api_key" in str(e).lower() or "unauthorized" in str(e).lower():
                    st.error("🚨 This appears to be an API key issue. Please check your API key.")
                    st.session_state.validated_api_key = None
    
    # Show chat history
    if st.session_state.chat_messages:
        st.divider()
        st.subheader("📜 Chat History")
        
        for i, msg in enumerate(reversed(st.session_state.chat_messages[-5:])):  # Show last 5
            with st.expander(f"Q{len(st.session_state.chat_messages)-i}: {msg['question'][:50]}..."):
                st.write(f"**Question:** {msg['question']}")
                st.write(f"**Answer:** {msg['answer']}")
                st.caption(f"Asked: {msg['timestamp'][:19]} | Sources: {msg['sources_count']}")
        
        # Clear history button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_messages = []
            if st.session_state.current_chat_context:
                st.session_state.current_chat_context.clear_conversation()
            st.rerun()

def history_interface():
    """Display research history"""
    
    st.header("📚 Research History")
    
    if not st.session_state.research_history:
        st.info("🔍 No research queries yet. Run your first research in the Research tab!")
        return
    
    for i, entry in enumerate(reversed(st.session_state.research_history)):
        with st.expander(f"🔍 {entry['query'][:60]}... ({entry['timestamp'][:10]})"):
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**Query:** {entry['query']}")
                st.write(f"**Date:** {entry['timestamp'][:19]}")
                st.write(f"**Papers:** {entry['papers_count']}")
            
            with col2:
                if st.button("🔄 Rerun", key=f"rerun_{i}"):
                    st.session_state.rerun_query = entry['query']
                    st.rerun()
    
    # Clear history button
    if st.button("🗑️ Clear History"):
        st.session_state.research_history = []
        st.rerun()

if __name__ == "__main__":
    main() 