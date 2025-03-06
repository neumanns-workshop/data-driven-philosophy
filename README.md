# Philosophical Questions Explorer

A Streamlit application for exploring and visualizing philosophical questions from Reddit.

## Features

- **3D Visualization**: Explore philosophical questions in a 3D semantic space, with filtering options by category and subreddit.
- **Question Explorer**: Search for philosophical questions using semantic search.

## Setup and Installation

1. Install the required dependencies:

```bash
pip install streamlit pandas numpy plotly scikit-learn sentence-transformers
```

2. If you're using the chunked embeddings (for GitHub compatibility):

```bash
# Combine the chunked embeddings
python split_embeddings.py combine
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

## Directory Structure

- `app.py` - Main Streamlit application
- `generate_filtered_visualizations.py` - Script to generate filtered 3D visualizations
- `data/` - Contains the philosophical questions data and embeddings
- `visualizations/` - Contains the 3D visualizations
- `requirements.txt` - Required Python packages

## Requirements

```
streamlit>=1.22.0
pandas>=1.5.0
numpy>=1.23.0
plotly>=5.13.0
scikit-learn>=1.2.0
sentence-transformers>=2.2.2
```

Install the requirements with:

```bash
pip install -r requirements.txt
```

## License

This project is for educational and research purposes only. The Reddit content displayed in this application belongs to their respective authors and is subject to Reddit's content policy and licensing terms.

## Author

**Jared Neumann**  
AI Consultant specializing in generative AI and natural language processing

Neumann's Workshop provides tailored AI solutions, research support, and educational resources for individuals and organizations.

- Email: [jared@neumannsworkshop.com](mailto:jared@neumannsworkshop.com)
- GitHub: [github.com/neumanns-workshop](https://github.com/neumanns-workshop)
- Website: [neumannsworkshop.com](https://neumannsworkshop.com/)

## Chunked Embeddings

The embeddings file is split into smaller chunks for GitHub compatibility (which has a 50MB file size limit). The app can automatically load and combine these chunks at runtime, or you can combine them manually:

```bash
# Split the embeddings into chunks (if needed)
python split_embeddings.py split

# Combine the chunks back into a single file
python split_embeddings.py combine
```

## Usage

To run the app, simply use the Streamlit command:

```bash
streamlit run app.py
```

### Data Acquisition and Processing

The philosophical questions dataset was created through the following process:

1. **Data Collection**: Reddit posts were collected from philosophy-related subreddits using the Pushshift API and Reddit API.

2. **Filtering and Cleaning**:
   - Only self-posts (text posts) with a score of 5 or higher were included
   - Posts were filtered to focus on question-asking content
   - Duplicates and non-English content were removed

3. **Enhancement**:
   - Each question was categorized into philosophical categories using a fine-tuned language model (ruggsea/Llama3-stanford-encyclopedia-philosophy-QA)
   - Question embeddings were generated using the SentenceTransformer model (all-mpnet-base-v2)
   - Metadata such as subreddit, timestamp, and score were preserved

4. **Visualization Preparation**:
   - 3D visualizations were created using PCA dimensionality reduction
   - Points were colored by philosophical category or subreddit
   - Interactive features were added for exploration

The dataset contains approximately 30,000 philosophical questions spanning from 2009 to 2024, covering a wide range of philosophical topics and discussions.

### Models Used

This application leverages several AI models:

1. **Categorization Model**: [ruggsea/Llama3-stanford-encyclopedia-philosophy-QA](https://huggingface.co/ruggsea/Llama3-stanford-encyclopedia-philosophy-QA)
   - Used to categorize questions into philosophical categories
   - Fine-tuned on the Stanford Encyclopedia of Philosophy
   - Provides accurate classification of philosophical topics

2. **Semantic Search Model**: [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
   - Used for generating embeddings and semantic search functionality
   - Provides state-of-the-art sentence embeddings
   - Enables finding semantically similar questions

3. **Dimensionality Reduction**: Principal Component Analysis (PCA)
   - Used to reduce the high-dimensional embeddings to 3D for visualization
   - Preserves the semantic relationships between questions
   - Enables interactive exploration of the semantic space

### Reddit Data Usage Note

This application uses data from Reddit, which is subject to Reddit's content policy and licensing terms. By using this application, you agree to comply with these terms and to use the data responsibly. 