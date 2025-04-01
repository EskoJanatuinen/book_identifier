#!/usr/bin/env python3

"""
Script for processing book images based on their filenames.
Automatically pairs images based on numeric ordering and then
attempts to find ISBNs for the pairs.

The script expects images to be named with a sequential number at the end:
- Example: 0702-Kirjat11-ET-1.jpg, 0702-Kirjat11-ET-2.jpg, 0702-Kirjat11-ET-3.jpg, 0702-Kirjat11-ET-4.jpg
- Images are paired by taking consecutive numbers: (1-2), (3-4), (5-6), etc.
- Only the number at the end of the filename matters; the prefix can be anything
- The number must be the very last part of the filename before the extension

Usage:
    python analyze_sorted.py
    
The script automatically processes all folders found in the import directory,
one by one, and removes each folder after processing.
"""

import os
import re
import sys
import shutil
import logging
import time
from book_identifier import BookIdentifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

class BookPairProcessor:
    """
    Processor for pairing book images based on numeric filenames and finding ISBNs.
    """
    def __init__(self, folder_name=None):
        """
        Initialize the processor.
        
        Args:
            folder_name (str, optional): Name of the folder containing images to process
        """
        self.root_dir = os.path.dirname(os.path.abspath(__file__))
        self.import_dir = os.path.join(self.root_dir, "import")
        self.processed_dir = os.path.join(self.root_dir, "processed")
        
        # The specific folder within import directory to process
        self.folder_name = folder_name
        self.folder_path = os.path.join(self.import_dir, folder_name) if folder_name else None
        
        # Initialize BookIdentifier for ISBN lookup (unidentified_dir parameter kept for backward compatibility but not used)
        self.book_identifier = BookIdentifier(
            self.import_dir, 
            self.processed_dir,
            os.path.join(self.root_dir, "unidentified")  # This parameter is ignored in the new implementation
        )
        
        # Image formats to process
        self.image_formats = ['.jpg', '.jpeg', '.png']
        
        # Track processing time
        self.start_time = 0
        self.end_time = 0

    def find_all_folders_in_import(self):
        """
        Find all folders in the import directory.
        
        Returns:
            list: List of folder names found
        """
        if not os.path.exists(self.import_dir):
            logging.error(f"Import directory '{self.import_dir}' not found.")
            return []
            
        folders = []
        for item in os.listdir(self.import_dir):
            item_path = os.path.join(self.import_dir, item)
            # Skip .DS_Store and other hidden files
            if not item.startswith('.') and os.path.isdir(item_path):
                folders.append(item)
                
        return folders
    
    def extract_number_from_filename(self, filename):
        """
        Extract the numeric part from the end of the filename.
        
        Args:
            filename (str): The filename to process
            
        Returns:
            int: The extracted number, or None if no number found
        """
        # Extract the last part of the filename that's a number
        match = re.search(r'(\d+)$', os.path.splitext(filename)[0])
        if match:
            return int(match.group(1))
        return None
    
    def find_images_in_folder(self):
        """
        Find all images in the specified folder.
        
        Returns:
            list: List of image file paths
        """
        if not self.folder_path or not os.path.exists(self.folder_path):
            logging.error(f"Folder '{self.folder_path}' not found in import directory.")
            return []
        
        images = []
        for file in os.listdir(self.folder_path):
            file_path = os.path.join(self.folder_path, file)
            if os.path.isfile(file_path):
                ext = os.path.splitext(file)[1].lower()
                # Skip hidden files and check for valid image extensions
                if not file.startswith('.') and ext in self.image_formats:
                    images.append(file_path)
        
        return images
    
    def create_pairs_by_number(self, images):
        """
        Create pairs of images based on numeric ordering (1-2, 3-4, etc.)
        
        Args:
            images (list): List of image file paths
            
        Returns:
            list: List of image pairs [{'front': path1, 'back': path2}, ...]
        """
        # Dictionary to store files by their numeric part
        files_by_number = {}
        
        for image_path in images:
            filename = os.path.basename(image_path)
            number = self.extract_number_from_filename(filename)
            
            if number is not None:
                files_by_number[number] = image_path
        
        # Sort numbers
        sorted_numbers = sorted(files_by_number.keys())
        
        # Create pairs (1-2, 3-4, etc.)
        pairs = []
        for i in range(0, len(sorted_numbers) - 1, 2):
            if i+1 < len(sorted_numbers):
                front_number = sorted_numbers[i]
                back_number = sorted_numbers[i+1]
                
                # Smaller number is front cover, larger is back cover
                pairs.append({
                    'front': files_by_number[front_number],
                    'back': files_by_number[back_number]
                })
        
        return pairs
    
    def find_isbn_for_pair(self, pair):
        """
        Try to find ISBN for a pair of images.
        First check back cover, then front cover.
        
        Args:
            pair (dict): Dictionary with 'front' and 'back' image paths
            
        Returns:
            str: Found ISBN or None
        """
        # First try back cover (more likely to contain ISBN)
        isbn = self.book_identifier.find_isbn_from_image(pair['back'])
        if isbn:
            return isbn
            
        # If not found, try front cover
        isbn = self.book_identifier.find_isbn_from_image(pair['front'])
        if isbn:
            return isbn
            
        # If still not found, try title-based lookup
        # Extract book info from front cover
        book_info = self.book_identifier.extract_book_info(pair['front'])
        if book_info and book_info.get('title'):
            # Here we could implement Google Books API lookup by title
            # But that's beyond the scope of this script
            logging.info(f"Found title: {book_info.get('title')}, but no ISBN")
        
        return None
    
    def process_folder(self):
        """
        Process the folder, find pairs, and lookup ISBNs.
        
        Returns:
            dict: Statistics about processed pairs or None if processing failed or skipped
        """
        if not self.folder_name:
            logging.error("No folder specified. Use: python analyze_sorted.py")
            return None
            
        # Check if the target directory already exists
        batch_processed_dir = os.path.join(self.processed_dir, self.folder_name)
        if os.path.exists(batch_processed_dir):
            logging.warning(f"Skipping folder '{self.folder_name}' - folder already exists in processed directory")
            logging.warning(f"To process this folder again, please delete or rename the existing folder: {batch_processed_dir}")
            return None
            
        self.start_time = time.time()
        logging.info(f"Processing folder: {self.folder_name}")
        
        # Find all images
        images = self.find_images_in_folder()
        if not images:
            logging.error(f"No images found in {self.folder_path}")
            return None
            
        logging.info(f"Found {len(images)} images")
        
        # Create pairs
        pairs = self.create_pairs_by_number(images)
        logging.info(f"Created {len(pairs)} pairs")
        
        # Create processed subfolder for this batch
        os.makedirs(batch_processed_dir, exist_ok=True)
        
        # Create subfolder for unidentified images
        missing_isbn_dir = os.path.join(batch_processed_dir, "isbn_missing")
        os.makedirs(missing_isbn_dir, exist_ok=True)
        
        # Counter for unknown pairs
        unknown_counter = 1
        
        # Find ISBNs for each pair
        pairs_with_isbn = 0
        pairs_with_title = 0
        pairs_without_any_info = 0
        
        for i, pair in enumerate(pairs):
            front_name = os.path.basename(pair['front'])
            back_name = os.path.basename(pair['back'])
            
            logging.info(f"Pair {i+1}: {front_name} + {back_name}")
            
            try:
                # Try to find ISBN
                isbn = self.find_isbn_for_pair(pair)
                
                front_ext = os.path.splitext(pair['front'])[1]
                back_ext = os.path.splitext(pair['back'])[1]
                
                if isbn:
                    # Case 1: ISBN found
                    pairs_with_isbn += 1
                    logging.info(f"  ✓ Found ISBN: {isbn}")
                    
                    # Prepare paths for copying
                    front_dest = os.path.join(batch_processed_dir, f"{isbn}_1{front_ext}")
                    back_dest = os.path.join(batch_processed_dir, f"{isbn}_2{back_ext}")
                    
                    # Check if destination files already exist
                    if os.path.exists(front_dest):
                        base, ext = os.path.splitext(front_dest)
                        counter = 1
                        while os.path.exists(f"{base}_{counter}{ext}"):
                            counter += 1
                        front_dest = f"{base}_{counter}{ext}"
                    
                    if os.path.exists(back_dest):
                        base, ext = os.path.splitext(back_dest)
                        counter = 1
                        while os.path.exists(f"{base}_{counter}{ext}"):
                            counter += 1
                        back_dest = f"{base}_{counter}{ext}"
                    
                    # Copy files immediately
                    shutil.copy2(pair['front'], front_dest)
                    shutil.copy2(pair['back'], back_dest)
                    
                    logging.info(f"  ✓ Copied images to '{self.folder_name}' folder as '{os.path.basename(front_dest)}' and '{os.path.basename(back_dest)}'")
                else:
                    # No ISBN found, try to get title
                    book_info = self.book_identifier.extract_book_info(pair['front'])
                    title = book_info.get('title') if book_info else None
                    
                    if title:
                        # Case 2: Title found but no ISBN
                        pairs_with_title += 1
                        
                        # Make title safe for filename
                        safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
                        if len(safe_title) > 30:  # Limit length
                            safe_title = safe_title[:30]
                        
                        logging.info(f"  ✓ No ISBN, but found title: {title}")
                        
                        # Prepare paths for copying
                        front_dest = os.path.join(missing_isbn_dir, f"{safe_title}_1{front_ext}")
                        back_dest = os.path.join(missing_isbn_dir, f"{safe_title}_2{back_ext}")
                        
                        # Check if destination files already exist
                        if os.path.exists(front_dest):
                            base, ext = os.path.splitext(front_dest)
                            counter = 1
                            while os.path.exists(f"{base}_{counter}{ext}"):
                                counter += 1
                            front_dest = f"{base}_{counter}{ext}"
                        
                        if os.path.exists(back_dest):
                            base, ext = os.path.splitext(back_dest)
                            counter = 1
                            while os.path.exists(f"{base}_{counter}{ext}"):
                                counter += 1
                            back_dest = f"{base}_{counter}{ext}"
                        
                        # Copy files immediately
                        shutil.copy2(pair['front'], front_dest)
                        shutil.copy2(pair['back'], back_dest)
                        
                        logging.info(f"  ✓ Copied images to 'isbn_missing' folder as '{os.path.basename(front_dest)}' and '{os.path.basename(back_dest)}'")
                    else:
                        # Case 3: No ISBN, no title
                        pairs_without_any_info += 1
                        
                        logging.warning(f"  ✗ No ISBN or title found for pair")
                        
                        # Prepare paths for copying
                        front_dest = os.path.join(missing_isbn_dir, f"unknown{unknown_counter}_1{front_ext}")
                        back_dest = os.path.join(missing_isbn_dir, f"unknown{unknown_counter}_2{back_ext}")
                        
                        # Check if destination files already exist
                        if os.path.exists(front_dest):
                            base, ext = os.path.splitext(front_dest)
                            counter = 1
                            while os.path.exists(f"{base}_{counter}{ext}"):
                                counter += 1
                            front_dest = f"{base}_{counter}{ext}"
                        
                        if os.path.exists(back_dest):
                            base, ext = os.path.splitext(back_dest)
                            counter = 1
                            while os.path.exists(f"{base}_{counter}{ext}"):
                                counter += 1
                            back_dest = f"{base}_{counter}{ext}"
                        
                        # Copy files immediately
                        shutil.copy2(pair['front'], front_dest)
                        shutil.copy2(pair['back'], back_dest)
                        
                        logging.info(f"  ✓ Copied images to 'isbn_missing' folder as '{os.path.basename(front_dest)}' and '{os.path.basename(back_dest)}'")
                        
                        # Increment counter for next unknown pair
                        unknown_counter += 1
            except Exception as e:
                logging.error(f"Error processing pair {i+1}: {e}")
        
        self.end_time = time.time()
        processing_time = self.end_time - self.start_time
        
        # Print summary
        logging.info("\n" + "="*60)
        logging.info("PROCESSING SUMMARY")
        logging.info("="*60)
        logging.info(f"Total pairs processed: {len(pairs)}")
        logging.info(f"Pairs with ISBN: {pairs_with_isbn}")
        logging.info(f"Pairs with title only: {pairs_with_title}")
        logging.info(f"Pairs without any info: {pairs_without_any_info}")
        logging.info(f"Total processing time: {processing_time:.2f} seconds")
        
        # Report AI token usage if available
        tokens_used = (
            self.book_identifier.total_prompt_tokens + 
            self.book_identifier.total_response_tokens
        )
        if tokens_used > 0:
            logging.info("\nAI Usage:")
            logging.info(f"- Total tokens: {tokens_used:,}")
            logging.info(f"  • Prompt tokens: {self.book_identifier.total_prompt_tokens:,}")
            logging.info(f"  • Response tokens: {self.book_identifier.total_response_tokens:,}")
        
        logging.info("="*60)
        
        return {
            'total': len(pairs),
            'with_isbn': pairs_with_isbn,
            'with_title': pairs_with_title,
            'without_info': pairs_without_any_info
        }

def main():
    """
    Main function to process book images from all folders in import directory.
    Each folder is processed one by one and removed after processing.
    """
    # Initialize processor without folder name
    processor = BookPairProcessor()
    
    # Find all folders in import directory
    folders = processor.find_all_folders_in_import()
    
    if not folders:
        logging.error("No folders found in import directory.")
        logging.info("Please place folders with images in the import directory.")
        sys.exit(1)
    
    total_start_time = time.time()
    total_pairs = 0
    total_pairs_with_isbn = 0
    total_pairs_with_title = 0
    total_pairs_without_info = 0
    total_processed = 0  # Track how many folders were actually processed
    total_skipped = 0    # Track how many folders were skipped
    
    # Process each folder one by one
    logging.info(f"Found {len(folders)} folders to process in import directory")
    
    for i, folder_name in enumerate(folders):
        logging.info(f"\n{'-'*20} Processing folder {i+1}/{len(folders)}: {folder_name} {'-'*20}\n")
        
        # Update processor with current folder
        processor.folder_name = folder_name
        processor.folder_path = os.path.join(processor.import_dir, folder_name)
        
        # Process the current folder
        folder_stats = processor.process_folder()
        
        if folder_stats:
            # Update total statistics
            total_processed += 1
            total_pairs += folder_stats['total']
            total_pairs_with_isbn += folder_stats['with_isbn']
            total_pairs_with_title += folder_stats['with_title']
            total_pairs_without_info += folder_stats['without_info']
            
            # Remove the processed folder
            folder_path = os.path.join(processor.import_dir, folder_name)
            logging.info(f"Removing folder: {folder_path}")
            shutil.rmtree(folder_path)
            logging.info(f"Folder removed successfully")
        else:
            # Folder was skipped or processing failed
            total_skipped += 1
            
            # NOTE: When a folder is skipped, we do NOT remove it from the import directory
            # This allows the user to manually handle the situation
            logging.info(f"Skipped folder '{folder_name}' has been left in the import directory")
    
    # Print overall summary
    total_end_time = time.time()
    total_processing_time = total_end_time - total_start_time
    
    logging.info("\n" + "="*60)
    logging.info("OVERALL PROCESSING SUMMARY")
    logging.info("="*60)
    logging.info(f"Total folders processed: {total_processed}")
    if total_skipped > 0:
        logging.info(f"Total folders skipped: {total_skipped}")
    logging.info(f"Total pairs processed: {total_pairs}")
    logging.info(f"Total pairs with ISBN: {total_pairs_with_isbn}")
    logging.info(f"Total pairs with title only: {total_pairs_with_title}")
    logging.info(f"Total pairs without any info: {total_pairs_without_info}")
    logging.info(f"Total processing time: {total_processing_time:.2f} seconds")
    
    # Report AI token usage if available
    total_tokens = (
        processor.book_identifier.total_prompt_tokens + 
        processor.book_identifier.total_response_tokens
    )
    if total_tokens > 0:
        logging.info("\nTotal AI Usage:")
        logging.info(f"- Total tokens: {total_tokens:,}")
        logging.info(f"  • Prompt tokens: {processor.book_identifier.total_prompt_tokens:,}")
        logging.info(f"  • Response tokens: {processor.book_identifier.total_response_tokens:,}")
    
    logging.info("="*60)

if __name__ == "__main__":
    main()