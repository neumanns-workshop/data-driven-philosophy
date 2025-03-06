# Philosophical Questions Explorer

A Streamlit application for exploring and visualizing philosophical questions from Reddit.

## Features

- **3D Visualization**: Explore philosophical questions in a 3D semantic space, with filtering options by category and subreddit.
- **Question Explorer**: Search for philosophical questions using semantic search.

## Running the App

To run the app, simply use the Streamlit command:

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