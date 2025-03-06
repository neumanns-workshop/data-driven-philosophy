#!/usr/bin/env python3
"""
Script to split large embeddings file into smaller chunks for GitHub upload.
"""

import pickle
import numpy as np
import os

def split_embeddings(input_file, output_dir, num_chunks=2):
    """
    Split a large embeddings file into smaller chunks.
    
    Args:
        input_file: Path to the input embeddings file
        output_dir: Directory to save the chunks
        num_chunks: Number of chunks to split the file into
    """
    print(f"Loading embeddings from {input_file}...")
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    
    # Get the embeddings and ids
    embeddings = data['embeddings']
    ids = data['ids']
    
    # Calculate chunk size
    chunk_size = len(embeddings) // num_chunks
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Split and save chunks
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < num_chunks - 1 else len(embeddings)
        
        chunk_data = {
            'embeddings': embeddings[start_idx:end_idx],
            'ids': ids[start_idx:end_idx],
            'chunk_info': {
                'chunk_number': i + 1,
                'total_chunks': num_chunks,
                'start_idx': start_idx,
                'end_idx': end_idx
            }
        }
        
        output_file = os.path.join(output_dir, f"embeddings_chunk_{i+1}of{num_chunks}.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump(chunk_data, f)
        
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"Saved chunk {i+1}/{num_chunks} to {output_file} ({file_size_mb:.2f} MB)")

def combine_embeddings(input_dir, output_file):
    """
    Combine chunked embeddings back into a single file.
    
    Args:
        input_dir: Directory containing the chunked embeddings
        output_file: Path to save the combined embeddings
    """
    # Find all chunk files
    chunk_files = [f for f in os.listdir(input_dir) if f.startswith("embeddings_chunk_") and f.endswith(".pkl")]
    chunk_files.sort()  # Ensure correct order
    
    if not chunk_files:
        print(f"No chunk files found in {input_dir}")
        return
    
    print(f"Found {len(chunk_files)} chunk files")
    
    # Load the first chunk to get the structure
    with open(os.path.join(input_dir, chunk_files[0]), 'rb') as f:
        first_chunk = pickle.load(f)
    
    # Initialize combined data
    combined_embeddings = []
    combined_ids = []
    
    # Process each chunk
    for chunk_file in chunk_files:
        print(f"Processing {chunk_file}...")
        with open(os.path.join(input_dir, chunk_file), 'rb') as f:
            chunk_data = pickle.load(f)
        
        combined_embeddings.append(chunk_data['embeddings'])
        combined_ids.extend(chunk_data['ids'])
    
    # Concatenate embeddings
    combined_embeddings = np.vstack(combined_embeddings)
    
    # Create combined data
    combined_data = {
        'embeddings': combined_embeddings,
        'ids': combined_ids
    }
    
    # Save combined data
    with open(output_file, 'wb') as f:
        pickle.dump(combined_data, f)
    
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"Saved combined embeddings to {output_file} ({file_size_mb:.2f} MB)")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Split or combine embeddings files")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Split command
    split_parser = subparsers.add_parser("split", help="Split embeddings file into chunks")
    split_parser.add_argument("--input", default="data/question_embeddings_enhanced.pkl", help="Input embeddings file")
    split_parser.add_argument("--output-dir", default="data/chunks", help="Output directory for chunks")
    split_parser.add_argument("--num-chunks", type=int, default=2, help="Number of chunks")
    
    # Combine command
    combine_parser = subparsers.add_parser("combine", help="Combine chunked embeddings")
    combine_parser.add_argument("--input-dir", default="data/chunks", help="Input directory with chunks")
    combine_parser.add_argument("--output", default="data/question_embeddings_enhanced.pkl", help="Output file")
    
    args = parser.parse_args()
    
    if args.command == "split":
        split_embeddings(args.input, args.output_dir, args.num_chunks)
    elif args.command == "combine":
        combine_embeddings(args.input_dir, args.output)
    else:
        parser.print_help() 