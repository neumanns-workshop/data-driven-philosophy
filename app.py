#!/usr/bin/env python3
"""
Streamlit app for exploring philosophical questions dataset.
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime
import numpy as np
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
import plotly.express as px
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import os

@st.cache_data
def load_data(file_path):
    """Load questions from JSONL file into a pandas DataFrame."""
    # Read JSONL file
    questions = []
    
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            questions.append(entry)
    
    # Convert to DataFrame
    df = pd.DataFrame(questions)
    
    # Convert timestamp to datetime - fix the warning by explicitly converting to numeric first
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df['date'] = df['datetime'].dt.date
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    
    return df

@st.cache_resource
def load_embeddings(embeddings_path):
    """Load pre-computed embeddings for semantic search."""
    try:
        # Check if we have a single embeddings file
        if os.path.exists(embeddings_path):
            with open(embeddings_path, 'rb') as f:
                return pickle.load(f)
        
        # Check if we have chunked embeddings
        chunks_dir = os.path.join(os.path.dirname(embeddings_path), "chunks")
        if os.path.exists(chunks_dir):
            # Find all chunk files
            chunk_files = [f for f in os.listdir(chunks_dir) if f.startswith("embeddings_chunk_") and f.endswith(".pkl")]
            if not chunk_files:
                return None
            
            # Sort to ensure correct order
            chunk_files.sort()
            
            # Load and combine chunks
            combined_embeddings = []
            combined_ids = []
            
            for chunk_file in chunk_files:
                with open(os.path.join(chunks_dir, chunk_file), 'rb') as f:
                    chunk_data = pickle.load(f)
                
                combined_embeddings.append(chunk_data['embeddings'])
                combined_ids.extend(chunk_data['ids'])
            
            # Concatenate embeddings
            combined_embeddings = np.vstack(combined_embeddings)
            
            # Create combined data
            return {
                'embeddings': combined_embeddings,
                'ids': combined_ids
            }
        
        return None
    except FileNotFoundError:
        return None

@st.cache_resource
def load_sentence_transformer(model_name="all-mpnet-base-v2"):
    """Load the sentence transformer model for generating query embeddings."""
    return SentenceTransformer(model_name)

def semantic_search(query, embeddings_data, model, top_k=10, threshold=None):
    """Perform semantic search using pre-computed embeddings."""
    if embeddings_data is None or model is None:
        return []
    
    # Generate embedding for the query
    query_embedding = model.encode(query)
    
    # Calculate cosine similarity
    embeddings = embeddings_data['embeddings']
    similarities = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    
    # Get top-k results, optionally filtering by threshold
    if threshold is not None:
        # Get indices where similarity is above threshold
        indices = np.where(similarities >= threshold)[0]
        # Sort by similarity
        sorted_indices = indices[np.argsort(similarities[indices])[::-1]]
        # Limit to top_k
        top_indices = sorted_indices[:top_k]
    else:
        # Just get top_k by similarity
        top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    results = []
    
    for idx in top_indices:
        results.append({
            'index': embeddings_data['ids'][idx],
            'similarity': float(similarities[idx])
        })
    
    return results

@st.cache_data
def generate_embedding_visualization(embeddings_data, df, sample_size=2000, perplexity=30):
    """Generate t-SNE visualization of embeddings colored by category."""
    if embeddings_data is None:
        return None
    
    # Get embeddings and corresponding data
    embeddings = embeddings_data['embeddings']
    ids = embeddings_data['ids']
    
    # Sample if too many points
    if len(embeddings) > sample_size:
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        sampled_embeddings = embeddings[indices]
        sampled_ids = [ids[i] for i in indices]
    else:
        sampled_embeddings = embeddings
        sampled_ids = ids
    
    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings_2d = tsne.fit_transform(sampled_embeddings)
    
    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'id': sampled_ids
    })
    
    # Add metadata from the original DataFrame
    metadata = df.loc[plot_df['id']].reset_index()
    plot_df['category'] = metadata['philosophical_category']
    plot_df['subreddit'] = metadata['subreddit']
    plot_df['question'] = metadata['questions'].apply(lambda q: q[0] if isinstance(q, list) and len(q) > 0 else "")
    
    return plot_df

def main():
    st.set_page_config(page_title="Philosophical Questions Explorer", layout="wide")
    
    # Custom CSS to make the visualization container fill more space
    st.markdown("""
    <style>
    .stApp {
        max-width: 100%;
    }
    iframe {
        width: 100%;
        min-height: 900px;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("Philosophical Questions Explorer")
    st.markdown("Explore high-quality philosophical questions from Reddit (score ≥ 5)")
    
    # Load data
    with st.spinner('Loading philosophical questions...'):
        try:
            df = load_data("data/phil_questions_enhanced.jsonl")
            st.success(f"Loaded {len(df):,} questions")
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return
    
    # Check if embeddings exist and load them
    embeddings_path = "data/question_embeddings_enhanced.pkl"
    embeddings_data = load_embeddings(embeddings_path)
    
    # Load model for semantic search if embeddings exist
    model = None
    if embeddings_data is not None:
        model = load_sentence_transformer()
        st.sidebar.success("✓ Semantic search enabled")
    else:
        st.sidebar.warning("⚠️ Semantic search not available - embeddings not found")
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Subreddit filter
    subreddits = sorted(df['subreddit'].unique())
    
    # Set default subreddits that we know exist
    default_subreddits = ["askphilosophy"]
    # Try to add other subreddits if they exist
    for sub in ["philosophy", "academicphilosophy", "StonerPhilosophy", "badphilosophy"]:
        if sub in subreddits:
            default_subreddits.append(sub)
    
    selected_subreddits = st.sidebar.multiselect(
        "Select Subreddits",
        subreddits,
        default=default_subreddits
    )
    
    # Category filter
    categories = sorted(df['philosophical_category'].unique())
    
    # Set default categories that we know exist
    default_categories = ["epistemological", "metaphysical", "ethical"]
    # Try to add other categories if they exist
    for cat in ["social", "existential"]:
        if cat in categories:
            default_categories.append(cat)
    
    selected_categories = st.sidebar.multiselect(
        "Select Categories",
        categories,
        default=default_categories
    )
    
    # Date range filter
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(df['datetime'].min().date(), df['datetime'].max().date())
    )
    
    # Apply filters
    mask = (
        df['subreddit'].isin(selected_subreddits) &
        df['philosophical_category'].isin(selected_categories) &
        (df['date'] >= date_range[0]) &
        (df['date'] <= date_range[1])
    )
    filtered_df = df[mask]
    
    # Debug info about filtered data
    st.sidebar.write(f"Filtered records: {len(filtered_df):,} / {len(df):,} ({len(filtered_df)/len(df):.1%})")
    
    # Move dataset info to the bottom of the sidebar
    st.sidebar.markdown("---")
    st.sidebar.header("Dataset Information")
    st.sidebar.write(f"Available subreddits: {len(subreddits)}")
    st.sidebar.write(f"Available categories: {len(categories)}")
    st.sidebar.write(f"From: {df['datetime'].min().date()}")
    st.sidebar.write(f"To: {df['datetime'].max().date()}")

    # Add author information
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About the Author")
    st.sidebar.markdown("""
    **Jared Neumann**  
    AI Consultant specializing in generative AI and natural language processing
    
    Neumann's Workshop provides tailored AI solutions, research support, and educational resources for individuals and organizations.
    
    [Email](mailto:jared@neumannsworkshop.com) | [GitHub](https://github.com/neumanns-workshop) | [Website](https://neumannsworkshop.com/)
    """)
    
    # Add data compliance notice
    st.sidebar.markdown("---")
    with st.sidebar.expander("Data Compliance Notice"):
        st.markdown("""
        **Reddit Data Usage**
        
        This application uses publicly available data from Reddit:
        
        - All data was collected in accordance with Reddit's API Terms of Service
        - Personal identifying information has been removed
        - Only publicly available posts with a score of 5 or higher are included
        - The data is used for educational and research purposes only
        - This application is not affiliated with or endorsed by Reddit
        
        The Reddit content displayed belongs to their respective authors and is subject to Reddit's content policy and licensing terms.
        
        If you are a content creator and wish to have your content removed, please contact the repository owner.
        """)
    
    # Add models information
    with st.sidebar.expander("Models Used"):
        st.markdown("""
        **Categorization Model**  
        [ruggsea/Llama3-stanford-encyclopedia-philosophy-QA](https://huggingface.co/ruggsea/Llama3-stanford-encyclopedia-philosophy-QA)
        
        **Semantic Search Model**  
        [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
        
        **Dimensionality Reduction**  
        Principal Component Analysis (PCA)
        """)
    
    # Basic stats
    st.header("Basic Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Subreddit Distribution")
        if len(filtered_df) > 0:
            # Simple bar chart using Streamlit
            subreddit_counts = filtered_df['subreddit'].value_counts().reset_index()
            subreddit_counts.columns = ['Subreddit', 'Count']
            st.bar_chart(subreddit_counts.set_index('Subreddit'))
        else:
            st.info("No data available with current filters.")
    
    with col2:
        st.subheader("Category Distribution")
        if len(filtered_df) > 0:
            # Simple bar chart using Streamlit
            category_counts = filtered_df['philosophical_category'].value_counts().reset_index()
            category_counts.columns = ['Category', 'Count']
            st.bar_chart(category_counts.set_index('Category'))
        else:
            st.info("No data available with current filters.")
    
    # Embedding Visualization
    st.header("Semantic Space Visualization (3D)")
    
    # Check if visualization files exist
    filtered_viz_dir = "visualizations/filtered"
    
    if os.path.exists(filtered_viz_dir):
        # Get available categories and subreddits
        categories = []
        subreddits = []
        
        for file in os.listdir(filtered_viz_dir):
            if file.startswith("category_") and file.endswith(".html"):
                category = file[len("category_"):-len(".html")]
                categories.append(category)
            elif file.startswith("subreddit_") and file.endswith(".html"):
                subreddit = file[len("subreddit_"):-len(".html")]
                subreddits.append(subreddit)
        
        categories.sort()
        subreddits.sort()
        
        # Add visualization options
        col1, col2 = st.columns(2)
        
        with col1:
            filter_type = st.selectbox(
                "Filter Type",
                ["All Data", "By Category", "By Subreddit"],
                index=0
            )
        
        with col2:
            if filter_type == "By Category":
                selected_filter = st.selectbox("Select Category", categories)
                viz_path = os.path.join(filtered_viz_dir, f"category_{selected_filter}.html")
            elif filter_type == "By Subreddit":
                selected_filter = st.selectbox("Select Subreddit", subreddits)
                viz_path = os.path.join(filtered_viz_dir, f"subreddit_{selected_filter}.html")
            else:
                color_by = st.selectbox("Color By", ["Category", "Subreddit"], index=0)
                viz_path = os.path.join(filtered_viz_dir, f"all_by_{color_by.lower()}.html")
        
        # Display the selected visualization
        if os.path.exists(viz_path):
            with open(viz_path, 'r') as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=900, width=None, scrolling=False)
            
            # Display appropriate info message
            if filter_type == "All Data":
                st.info(f"This 3D visualization shows all philosophical questions colored by {color_by.lower()}. " +
                       "Similar questions are positioned closer together in the 3D space. " +
                       "You can rotate, zoom, and pan the visualization to explore different perspectives. " +
                       "Hover over points to see the actual questions.")
            elif filter_type == "By Category":
                st.info(f"This 3D visualization shows philosophical questions in the '{selected_filter}' category, colored by subreddit. " +
                       "Similar questions are positioned closer together in the 3D space. " +
                       "You can rotate, zoom, and pan the visualization to explore different perspectives. " +
                       "Hover over points to see the actual questions.")
            else:  # By Subreddit
                st.info(f"This 3D visualization shows philosophical questions from the '{selected_filter}' subreddit, colored by category. " +
                       "Similar questions are positioned closer together in the 3D space. " +
                       "You can rotate, zoom, and pan the visualization to explore different perspectives. " +
                       "Hover over points to see the actual questions.")
        else:
            st.error(f"Visualization file not found: {viz_path}")
    else:
        st.warning("Filtered 3D visualizations are not available. Please run the visualization script first:\n\n" +
                  "```\npython scripts/generate_filtered_visualizations.py\n```")
    
    # Question explorer
    st.header("Question Explorer")
    
    # Search tabs
    search_tabs = st.tabs(["Keyword Search", "Semantic Search"])
    
    with search_tabs[0]:
        # Keyword search
        col1, col2 = st.columns([1, 3])
        
        with col1:
            sort_by = st.selectbox(
                "Sort by",
                ["datetime", "score"],
                index=0,
                key="keyword_sort"
            )
        
        with col2:
            # Search box
            keyword_query = st.text_input(
                "Search in questions or content", 
                placeholder="Enter keywords to search...",
                key="keyword_search"
            )
        
        # Apply keyword search filter if query exists
        search_results_df = filtered_df
        if keyword_query:
            search_mask = (
                df['questions'].astype(str).str.contains(keyword_query, case=False) |
                df['content'].astype(str).str.contains(keyword_query, case=False)
            )
            search_results_df = filtered_df[search_mask]
            st.info(f"Found {len(search_results_df):,} questions matching '{keyword_query}'")
        
        # Sort and display questions
        sorted_df = search_results_df.sort_values(sort_by, ascending=False)
    
    with search_tabs[1]:
        # Semantic search
        semantic_query = st.text_input(
            "Semantic search (finds conceptually similar questions)",
            placeholder="Describe what you're looking for...",
            key="semantic_search"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            top_k = st.slider("Number of results", min_value=5, max_value=50, value=20)
        
        with col2:
            use_threshold = st.checkbox("Use similarity threshold", value=False)
            threshold = None
            if use_threshold:
                threshold = st.slider("Minimum similarity", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
        
        if semantic_query and embeddings_data is not None and model is not None:
            with st.spinner("Performing semantic search..."):
                search_results = semantic_search(semantic_query, embeddings_data, model, top_k=top_k, threshold=threshold)
                
                if search_results:
                    result_indices = [result['index'] for result in search_results]
                    similarities = [result['similarity'] for result in search_results]
                    
                    # Get the questions from the dataframe
                    semantic_results_df = df.loc[result_indices].copy()
                    semantic_results_df['similarity'] = similarities
                    
                    # Apply filters to semantic results
                    semantic_mask = (
                        semantic_results_df['subreddit'].isin(selected_subreddits) &
                        semantic_results_df['philosophical_category'].isin(selected_categories) &
                        (semantic_results_df['date'] >= date_range[0]) &
                        (semantic_results_df['date'] <= date_range[1])
                    )
                    semantic_results_df = semantic_results_df[semantic_mask]
                    
                    st.info(f"Found {len(semantic_results_df):,} semantically similar questions")
                    
                    # Show example related terms
                    st.write("**Try searching for related terms:**")
                    if "child" in semantic_query.lower() or "children" in semantic_query.lower():
                        st.write("Related to children: baby, infant, kid, parent, family, birth")
                    elif "knowledge" in semantic_query.lower():
                        st.write("Related to knowledge: epistemology, belief, truth, justified, understanding")
                    elif "god" in semantic_query.lower() or "religion" in semantic_query.lower():
                        st.write("Related to religion: faith, belief, divine, worship, spiritual, theology")
                    elif "mind" in semantic_query.lower() or "consciousness" in semantic_query.lower():
                        st.write("Related to mind: awareness, thought, perception, cognition, brain")
                    
                    # Sort by similarity
                    sorted_df = semantic_results_df.sort_values('similarity', ascending=False)
        elif semantic_query:
            st.error("Semantic search is not available. Embeddings or model not loaded.")
            sorted_df = pd.DataFrame()
        else:
            # Default to showing filtered results sorted by date
            sorted_df = filtered_df.sort_values('datetime', ascending=False)
    
    # Pagination
    if len(sorted_df) > 0:
        items_per_page = 10
        total_pages = max(1, len(sorted_df) // items_per_page + (1 if len(sorted_df) % items_per_page > 0 else 0))
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
        
        start_idx = (page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, len(sorted_df))
        
        for _, row in sorted_df.iloc[start_idx:end_idx].iterrows():
            question_text = row['questions'][0] if isinstance(row['questions'], list) and len(row['questions']) > 0 else "No question text"
            
            # Add similarity score to title if available
            title = f"[{row['philosophical_category']}] {question_text}"
            if 'similarity' in row:
                title = f"{title} (Similarity: {row['similarity']:.2f})"
                
            with st.expander(title):
                st.write(f"**Subreddit:** r/{row['subreddit']}")
                st.write(f"**Posted:** {row['datetime'].strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"**Score:** {row['score']}")
                if 'content' in row and row['content']:
                    st.write("**Content:**")
                    st.write(row['content'])
                if 'url' in row:
                    st.write(f"**URL:** [{row['url']}]({row['url']})")
        
        st.write(f"Showing page {page} of {total_pages} ({len(sorted_df)} questions total)")
    else:
        st.info("No questions match your filters. Try adjusting the filter criteria.")

if __name__ == "__main__":
    main() 