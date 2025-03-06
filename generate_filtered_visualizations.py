#!/usr/bin/env python3
"""
Generate filtered 3D visualizations of the embedding space for the philosophical questions dataset.
This script creates multiple visualizations filtered by category or subreddit.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import os
import pickle
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import argparse

def generate_filtered_visualizations(embeddings_path, data_path, output_dir):
    """
    Generate multiple filtered 3D visualizations of the embedding space.
    
    Args:
        embeddings_path: Path to the embeddings file
        data_path: Path to the data file
        output_dir: Directory to save the visualizations
    """
    print(f"Generating filtered 3D visualizations from {embeddings_path}")
    print(f"Using data from {data_path}")
    print(f"Saving visualizations to {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load embeddings
    with open(embeddings_path, 'rb') as f:
        embeddings_data = pickle.load(f)
    
    # Load data
    questions = []
    with open(data_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            questions.append(entry)
    df = pd.DataFrame(questions)
    
    # Get embeddings and corresponding data
    embeddings = embeddings_data['embeddings']
    ids = embeddings_data['ids']
    
    # Apply PCA for dimensionality reduction (3D)
    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(embeddings)
    
    # Scale the embeddings to spread them out more
    scaler = MinMaxScaler(feature_range=(-1, 1))
    embeddings_3d = scaler.fit_transform(embeddings_3d)
    
    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'x': embeddings_3d[:, 0],
        'y': embeddings_3d[:, 1],
        'z': embeddings_3d[:, 2],
        'id': ids
    })
    
    # Add metadata from the original DataFrame
    metadata = df.loc[plot_df['id']].reset_index()
    plot_df['category'] = metadata['philosophical_category']
    plot_df['subreddit'] = metadata['subreddit']
    plot_df['question'] = metadata['questions'].apply(lambda q: q[0] if isinstance(q, list) and len(q) > 0 else "")
    
    # Get unique categories and subreddits
    categories = sorted(plot_df['category'].unique())
    subreddits = sorted(plot_df['subreddit'].unique())
    
    # Generate visualization for all data colored by category
    generate_visualization(plot_df, os.path.join(output_dir, "all_by_category.html"), color_by="category")
    
    # Generate visualization for all data colored by subreddit
    generate_visualization(plot_df, os.path.join(output_dir, "all_by_subreddit.html"), color_by="subreddit")
    
    # Generate visualizations filtered by category
    for category in categories:
        filtered_df = plot_df[plot_df['category'] == category]
        output_path = os.path.join(output_dir, f"category_{category}.html")
        generate_visualization(filtered_df, output_path, color_by="subreddit", title=f"3D Semantic Space - Category: {category}")
    
    # Generate visualizations filtered by subreddit
    for subreddit in subreddits:
        filtered_df = plot_df[plot_df['subreddit'] == subreddit]
        output_path = os.path.join(output_dir, f"subreddit_{subreddit}.html")
        generate_visualization(filtered_df, output_path, color_by="category", title=f"3D Semantic Space - Subreddit: {subreddit}")
    
    # Create an index file with links to all visualizations
    create_index_file(output_dir, categories, subreddits)
    
    print(f"Generated {2 + len(categories) + len(subreddits)} visualizations")

def generate_visualization(plot_df, output_path, color_by="category", title=None):
    """
    Generate a 3D visualization and save it as an HTML file.
    
    Args:
        plot_df: DataFrame with the data to plot
        output_path: Path to save the visualization
        color_by: Field to color points by (category or subreddit)
        title: Title for the visualization
    """
    # Reduce marker size for better visibility
    marker_size = 3
    
    if title is None:
        title = "3D Semantic Space of Philosophical Questions (PCA)"
    
    if color_by == "category":
        fig = px.scatter_3d(
            plot_df, x='x', y='y', z='z', color='category',
            hover_data=['question', 'subreddit'],
            title=title,
            labels={'category': 'Category', 'x': 'Component 1', 'y': 'Component 2', 'z': 'Component 3'},
            color_discrete_sequence=px.colors.qualitative.Bold,
            opacity=0.8,
            size_max=marker_size
        )
    else:
        fig = px.scatter_3d(
            plot_df, x='x', y='y', z='z', color='subreddit',
            hover_data=['question', 'category'],
            title=title,
            labels={'subreddit': 'Subreddit', 'x': 'Component 1', 'y': 'Component 2', 'z': 'Component 3'},
            color_discrete_sequence=px.colors.qualitative.Vivid,
            opacity=0.8,
            size_max=marker_size
        )
    
    # Update layout for better visualization
    fig.update_layout(
        scene=dict(
            aspectmode='cube',
            xaxis=dict(showticklabels=False, title='', showgrid=False, showbackground=False, zeroline=False),
            yaxis=dict(showticklabels=False, title='', showgrid=False, showbackground=False, zeroline=False),
            zaxis=dict(showticklabels=False, title='', showgrid=False, showbackground=False, zeroline=False),
            bgcolor='rgba(0,0,0,0)',
            camera=dict(
                eye=dict(x=1.25, y=1.25, z=1.25)
            )
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.5)"
        ),
        width=1200,
        height=900,
        autosize=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Update traces for better appearance
    fig.update_traces(
        marker=dict(
            size=5,
            opacity=0.8,
            line=dict(width=0.5, color='rgba(255,255,255,0.5)')
        )
    )
    
    # Save the figure to an HTML file
    fig.write_html(
        output_path,
        include_plotlyjs='cdn',
        full_html=True,
        config={
            'responsive': True,
            'displayModeBar': True,
            'displaylogo': False,
            'scrollZoom': True
        }
    )
    print(f"Visualization saved to {output_path}")

def create_index_file(output_dir, categories, subreddits):
    """
    Create an index.html file with links to all visualizations.
    
    Args:
        output_dir: Directory with the visualizations
        categories: List of categories
        subreddits: List of subreddits
    """
    index_path = os.path.join(output_dir, "index.html")
    
    with open(index_path, 'w') as f:
        f.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Philosophical Questions Visualizations</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2 { color: #333; }
                ul { list-style-type: none; padding: 0; }
                li { margin: 10px 0; }
                a { color: #0066cc; text-decoration: none; }
                a:hover { text-decoration: underline; }
                .section { margin-bottom: 30px; }
            </style>
        </head>
        <body>
            <h1>Philosophical Questions Visualizations</h1>
            
            <div class="section">
                <h2>All Data</h2>
                <ul>
                    <li><a href="all_by_category.html" target="_blank">All Data (Colored by Category)</a></li>
                    <li><a href="all_by_subreddit.html" target="_blank">All Data (Colored by Subreddit)</a></li>
                </ul>
            </div>
            
            <div class="section">
                <h2>Filtered by Category</h2>
                <ul>
        """)
        
        for category in categories:
            f.write(f'            <li><a href="category_{category}.html" target="_blank">{category}</a></li>\n')
        
        f.write("""
                </ul>
            </div>
            
            <div class="section">
                <h2>Filtered by Subreddit</h2>
                <ul>
        """)
        
        for subreddit in subreddits:
            f.write(f'            <li><a href="subreddit_{subreddit}.html" target="_blank">{subreddit}</a></li>\n')
        
        f.write("""
                </ul>
            </div>
        </body>
        </html>
        """)
    
    print(f"Index file created at {index_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate filtered 3D visualizations for the Streamlit app")
    parser.add_argument("--embeddings", default="data/question_embeddings_enhanced.pkl", help="Path to embeddings file")
    parser.add_argument("--data", default="data/phil_questions_enhanced.jsonl", help="Path to data file")
    parser.add_argument("--output", default="visualizations/filtered", help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    generate_filtered_visualizations(args.embeddings, args.data, args.output)

if __name__ == "__main__":
    main() 