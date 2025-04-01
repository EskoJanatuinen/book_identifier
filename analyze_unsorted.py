#!/usr/bin/env python3

"""
Script for processing book images by finding matching pairs.
Automatically analyzes book covers to find pairs that belong to the same book,
and attempts to identify ISBNs using AI-based image analysis.

Usage:
    python analyze_unsorted.py
    
The script automatically processes all folders found in the import directory,
moving identified book pairs to the processed directory and storing unidentified
images in isbn_missing folders.
"""

import os
import shutil
import logging
from book_identifier import BookIdentifier
from dotenv import load_dotenv  # type: ignore

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Main program
def process_subfolder(folder_path, output_folder=None):
    """
    Process a single subfolder, finding all book pairs within it.
    
    Args:
        folder_path (str): Path to the subfolder
        output_folder (str, optional): Name of the output subfolder within processed directory
    
    Returns:
        dict: Statistics about the processing or None if skipped
    """
    # Extract folder name from path
    folder_name = os.path.basename(folder_path)
    
    # Determine paths
    root_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(root_dir, "processed")
    if output_folder:
        processed_dir = os.path.join(processed_dir, output_folder)
    
    # Check if the target directory already exists
    target_folder = os.path.join(processed_dir, folder_name)
    if os.path.exists(target_folder):
        logging.warning(f"Skipping folder '{folder_name}' - folder already exists in processed directory")
        logging.warning(f"To process this folder again, please delete or rename the existing folder: {target_folder}")
        return None
    
    # Create subfolder in processed directory if output_folder is specified
    if output_folder:
        os.makedirs(processed_dir, exist_ok=True)
    
    # Initialize Book Identifier (unidentified_dir parameter is kept for backward compatibility but not used)
    identifier = BookIdentifier(
        import_dir=folder_path,
        processed_dir=processed_dir,
        unidentified_dir=os.path.join(root_dir, "unidentified")  # This parameter is ignored in the new implementation
    )
    
    # Run the identification process
    stats = identifier.run(folder_name)
    
    return stats

def process_all_subfolders():
    """
    Process all subfolders in the import directory.
    Each subfolder is processed independently and its images are
    moved to a corresponding subfolder in the processed directory.
    After processing, each subfolder is removed from the import directory.
    If a subfolder already exists in the processed directory, it is skipped.
    
    Returns:
        dict: Combined statistics from all subfolder processing
    """
    root_dir = os.path.dirname(os.path.abspath(__file__))
    import_dir = os.path.join(root_dir, "import")
    
    # Check if import directory exists
    if not os.path.exists(import_dir):
        logging.error(f"Import directory not found: {import_dir}")
        return None
    
    # Get all subfolders in import directory
    subfolders = []
    for item in os.listdir(import_dir):
        item_path = os.path.join(import_dir, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            subfolders.append(item_path)
    
    if not subfolders:
        logging.info("No subfolders found in import directory. Processing main directory.")
        # If no subfolders, use the main import directory
        subfolders = [import_dir]
    
    # Variables to track combined statistics
    total_identified = 0
    total_unidentified = 0
    total_time = 0
    total_prompt_tokens = 0
    total_response_tokens = 0
    total_processed = 0  # Track how many folders were actually processed
    total_skipped = 0    # Track how many folders were skipped
    
    # Process each subfolder
    for i, subfolder in enumerate(subfolders):
        folder_name = os.path.basename(subfolder)
        logging.info(f"\n[{i+1}/{len(subfolders)}] Processing folder: {folder_name}")
        
        # Process the subfolder
        stats = process_subfolder(subfolder)
        
        if stats:
            # Folder was processed successfully
            total_processed += 1
            total_identified += stats['identified_pairs']
            total_unidentified += stats['unidentified_images']
            total_time += stats['processing_time']
            total_prompt_tokens += stats['prompt_tokens']
            total_response_tokens += stats['response_tokens']
            
            # Remove the subfolder after processing, but only if it's a subfolder
            # (not the main import directory itself)
            if subfolder != import_dir:
                try:
                    logging.info(f"Removing processed folder: {subfolder}")
                    shutil.rmtree(subfolder)
                    logging.info(f"Folder removed successfully")
                except Exception as e:
                    logging.error(f"Error removing folder {subfolder}: {e}")
        else:
            # Folder was skipped or processing failed
            total_skipped += 1
            
            # NOTE: When a folder is skipped, we do NOT remove it from the import directory
            # This allows the user to manually handle the situation
            if subfolder != import_dir:
                subfolder_name = os.path.basename(subfolder)
                logging.info(f"Skipped folder '{subfolder_name}' has been left in the import directory")
    
    # Calculate totals
    total_tokens = total_prompt_tokens + total_response_tokens
    
    # Print final summary
    logging.info("\n============= FINAL SUMMARY =============")
    logging.info(f"Total folders processed: {total_processed}")
    if total_skipped > 0:
        logging.info(f"Total folders skipped: {total_skipped}")
    logging.info(f"Total pairs identified: {total_identified}")
    logging.info(f"Total images remaining unidentified: {total_unidentified}")
    logging.info(f"Total processing time: {total_time:.2f} seconds")
    if total_tokens > 0:
        logging.info(f"Total AI tokens used: {total_tokens:,}")
        logging.info(f"  • Prompt tokens: {total_prompt_tokens:,}")
        logging.info(f"  • Response tokens: {total_response_tokens:,}")
    logging.info("=========================================\n")
    
    return {
        'folders_processed': total_processed,
        'folders_skipped': total_skipped,
        'identified_pairs': total_identified,
        'unidentified_images': total_unidentified,
        'processing_time': total_time,
        'total_tokens': total_tokens
    }

def main():
    """
    Main function to run the identification process.
    """
    logging.info("Starting book identifier...")
    
    # Check if there are images directly in the import directory
    root_dir = os.path.dirname(os.path.abspath(__file__))
    import_dir = os.path.join(root_dir, "import")
    
    if not os.path.exists(import_dir):
        os.makedirs(import_dir)
        logging.info(f"Created import directory: {import_dir}")
        logging.info("Please place book images in the import directory.")
        return
    
    # Process all subfolders (or main directory if no subfolders)
    process_all_subfolders()
    
    logging.info("Book identification complete!")

if __name__ == "__main__":
    main()