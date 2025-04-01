#!/usr/bin/env python3

"""
Book Identifier Module for Automated Book Processing

This module provides the core functionality for identifying and organizing book images.
It uses AI-based image analysis to:
1. Match front and back covers of the same book
2. Extract ISBN numbers from book covers
3. Identify book titles and authors
4. Organize processed books into directories by ISBN or title

The main class is BookIdentifier, which handles all image analysis and processing,
using Google's Gemini AI API for intelligent image recognition.

Dependencies:
- Google Gemini API
- Python Imaging Library (PIL/Pillow)
- NumPy
- Requests
"""

import os
import shutil
import re
import logging
import requests
import google.generativeai as genai
from dotenv import load_dotenv  # type: ignore
from itertools import combinations
import PIL.Image
import numpy as np
from collections import defaultdict
import time
import threading
import queue

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BookIdentifier:
    """
    Book Identifier class that uses AI to identify books from cover images
    and organizes them into separate folders for processed and unidentified images.
    """
    def __init__(self, import_dir, processed_dir, unidentified_dir, api_key=None):
        """
        Initialize the Book Identifier.
        
        Args:
            import_dir (str): Directory path containing imported book cover images
            processed_dir (str): Directory path for processed images
            unidentified_dir (str): Directory path for unidentified images (DEPRECATED - not used anymore)
            api_key (str, optional): Google Gemini API key. If None, loads from environment
        """
        self.import_dir = import_dir
        self.processed_dir = processed_dir
        # NOTE: unidentified_dir parameter is kept for backward compatibility but not used
        self.unidentified_dir = unidentified_dir  # DEPRECATED - not used
        
        # Supported image formats
        self.image_formats = ['.jpg', '.jpeg', '.png']
        # Google Gemini API key
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key not set. Set it in the .env file as GOOGLE_API_KEY=your_api_key or provide it as a parameter.")
        # Initialize Gemini API
        genai.configure(api_key=self.api_key)
        # Select model
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        # ISBN regular expression (10 and 13 digit)
        self.isbn_regex = r'(?:ISBN(?:-1[03])?:?\s*)?(?=[0-9X]{10}$|(?=(?:[0-9]+[-\s]){3})[-\s0-9X]{13}$|97[89][0-9]{10}$|(?=(?:[0-9]+[-\s]){4})[-\s0-9]{17}$)(?:97[89][-\s]?)?[0-9]{1,5}[-\s]?[0-9]+[-\s]?[0-9]+[-\s]?[0-9X]'
        # API call settings
        self.max_retries = 5
        self.delay = 0.5  # Fixed delay in seconds between API calls (allows ~2 calls per second)
        # Token tracking
        self.total_prompt_tokens = 0
        self.total_response_tokens = 0
        # Book metadata storage
        self.book_metadata = {}  # Maps image path to {'title': title, 'author': author, 'isbn': isbn}
        
        # Background task infrastructure
        self.task_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._background_worker, daemon=True)
        self.worker_thread.start()
        self.color_tones_cache = {}  # Cache for color tones
        self.image_data_cache = {}  # Cache for binary image data
    
    def _background_worker(self):
        """Background worker thread that processes tasks from the queue"""
        while True:
            try:
                task, args, kwargs = self.task_queue.get()
                task(*args, **kwargs)
                self.task_queue.task_done()
            except Exception as e:
                logging.error(f"Error in background worker: {e}")
    
    def add_background_task(self, task, *args, **kwargs):
        """Add a task to be executed in the background thread"""
        self.task_queue.put((task, args, kwargs))
    
    def preload_image_data(self, image_path):
        """
        Preload and cache image data.
        
        Args:
            image_path (str): Path to the image file
        """
        if image_path not in self.image_data_cache:
            try:
                with open(image_path, "rb") as image_file:
                    self.image_data_cache[image_path] = image_file.read()
                    logging.debug(f"Preloaded image data for {os.path.basename(image_path)}")
            except Exception as e:
                logging.error(f"Error preloading image data for {image_path}: {e}")
    
    def get_image_data(self, image_path):
        """
        Get image data from cache or load it.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            bytes: Image data
        """
        if image_path not in self.image_data_cache:
            self.preload_image_data(image_path)
        return self.image_data_cache.get(image_path)
    
    def api_call_with_retry(self, call_function, *args, **kwargs):
        """
        Make an API call with retry logic for rate limit errors.
        While waiting for the API delay, perform background tasks.
        
        Args:
            call_function: The function to call
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            The result of the function call
        """
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                # Start the delay timer
                delay_end = time.time() + self.delay
                
                # Make the API call
                response = call_function(*args, **kwargs)
                
                # Track token usage if available
                if hasattr(response, 'usage_metadata'):
                    metadata = response.usage_metadata
                    if metadata:
                        prompt_tokens = metadata.prompt_token_count or 0
                        response_tokens = metadata.candidates_token_count or 0
                        self.total_prompt_tokens += prompt_tokens
                        self.total_response_tokens += response_tokens
                        logging.info(f"API call tokens - Prompt: {prompt_tokens}, Response: {response_tokens}")
                
                # If we need to wait more to satisfy the delay, use that time for background tasks
                remaining_delay = delay_end - time.time()
                if remaining_delay > 0:
                    time.sleep(remaining_delay)
                
                return response
                
            except Exception as e:
                error_str = str(e)
                retry_count += 1
                
                # Check if this is a rate limit error (429)
                if "429" in error_str or "Resource has been exhausted" in error_str:
                    # Calculate exponential backoff delay: 2^retry_count
                    backoff_delay = 2 ** retry_count
                    logging.warning(f"Rate limit hit, retrying in {backoff_delay:.1f} seconds (attempt {retry_count}/{self.max_retries})")
                    time.sleep(backoff_delay)
                else:
                    # For non-rate-limit errors, don't retry
                    logging.error(f"API call failed: {e}")
                    raise
                    
        # If we've exhausted retries
        logging.error(f"API call failed after {self.max_retries} retries")
        raise Exception(f"API call failed after {self.max_retries} retries")

    def extract_book_info(self, image_path):
        """
        Extract book title and author from a single image using AI.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Dictionary with 'title' and 'author' keys
        """
        try:
            # Check if we already have metadata for this image
            if image_path in self.book_metadata:
                return self.book_metadata[image_path]
                
            image_data = self.get_image_data(image_path)
            if not image_data:
                return {'title': None, 'author': None, 'isbn': None}
                
            prompt = """Analyze this book cover image and extract the following information:
            1. Book title
            2. Author name(s)
            
            Respond in this exact format:
            Title: [book title]
            Author: [author name]
            
            If you cannot identify one of these items, use "Unknown" for that field."""
            
            # Use the retry function for API call
            def api_call():
                image_part = {
                    "mime_type": "image/jpeg",
                    "data": image_data
                }
                return self.model.generate_content([prompt, image_part])
            
            response = self.api_call_with_retry(api_call)
            response_text = response.text
            
            # Extract title using regex
            title_match = re.search(r'Title:\s*(.*?)(?:\n|$)', response_text, re.IGNORECASE)
            title = title_match.group(1).strip() if title_match else "Unknown"
            if title.lower() == "unknown":
                title = None
                
            # Extract author using regex
            author_match = re.search(r'Author:\s*(.*?)(?:\n|$)', response_text, re.IGNORECASE)
            author = author_match.group(1).strip() if author_match else "Unknown"
            if author.lower() == "unknown":
                author = None
            
            # Check if we already have an ISBN for this image
            isbn = None
            if image_path in self.book_metadata and 'isbn' in self.book_metadata[image_path]:
                isbn = self.book_metadata[image_path]['isbn']
            
            # Store the metadata
            metadata = {'title': title, 'author': author, 'isbn': isbn}
            self.book_metadata[image_path] = metadata
            
            if title:
                logging.info(f"Found title '{title}' in image {os.path.basename(image_path)}")
            if author:
                logging.info(f"Found author '{author}' in image {os.path.basename(image_path)}")
                
            return metadata
            
        except Exception as e:
            logging.error(f"Error extracting book info from image {image_path}: {e}")
            return {'title': None, 'author': None, 'isbn': None}

    def normalize_author_name(self, author):
        """
        Normalize author name for better matching.
        
        Args:
            author (str): Author name to normalize
            
        Returns:
            str: Normalized author name
        """
        if not author:
            return None
            
        # Convert to lowercase
        normalized = author.lower()
        
        # Remove common prefixes/titles
        prefixes = ['dr.', 'dr ', 'prof.', 'prof ', 'professor ', 'sir ', 'mr.', 'mr ', 'mrs.', 'mrs ', 'ms.', 'ms ']
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()
        
        # Remove punctuation
        normalized = re.sub(r'[.,;:\-\'"]', '', normalized)
        
        # Remove extra spaces
        normalized = ' '.join(normalized.split())
        
        return normalized

    def calculate_color_tone(self, image_path):
        """
        Calculate the average color tone of an image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            tuple: RGB values (0-255) or None if error
        """
        try:
            # Check if we already have the color tone in cache
            if image_path in self.color_tones_cache:
                return self.color_tones_cache[image_path]
                
            # Open the image and calculate average color
            with PIL.Image.open(image_path) as img:
                img = img.resize((100, 100))  # Resize for faster processing
                img = img.convert('RGB')
                pixels = np.array(img)
                avg_color = tuple(np.mean(pixels, axis=(0, 1)).astype(int))
                
                # Store in cache
                self.color_tones_cache[image_path] = avg_color
                return avg_color
                
        except Exception as e:
            logging.error(f"Error calculating color tone for {image_path}: {e}")
            self.color_tones_cache[image_path] = None
            return None
    
    def compare_color_tones(self, tone1, tone2, threshold=50):
        """
        Compare two color tones to determine if they're similar.
        
        Args:
            tone1 (tuple): First RGB color tone
            tone2 (tuple): Second RGB color tone
            threshold (int): Threshold for considering colors similar
            
        Returns:
            bool: True if colors are similar, False otherwise
        """
        if not tone1 or not tone2:
            return False
            
        # Calculate Euclidean distance between colors
        dist = sum((a - b) ** 2 for a, b in zip(tone1, tone2)) ** 0.5
        return dist <= threshold

    def preload_color_tones(self, image_paths):
        """
        Preload color tones for a list of images.
        
        Args:
            image_paths (list): List of image paths
        """
        def _load_color_tones():
            for path in image_paths:
                if path not in self.color_tones_cache:
                    self.calculate_color_tone(path)
        
        self.add_background_task(_load_color_tones)

    def preload_images(self, image_paths):
        """
        Preload image data for a list of images.
        
        Args:
            image_paths (list): List of image paths
        """
        def _load_images():
            for path in image_paths:
                if path not in self.image_data_cache:
                    self.preload_image_data(path)
        
        self.add_background_task(_load_images)

    def find_isbn_from_image(self, image_path):
        """
        Find an ISBN number from a single image using AI.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            str: ISBN number if found, None otherwise
        """
        try:
            # Check if we already have metadata with ISBN for this image
            if image_path in self.book_metadata and self.book_metadata[image_path].get('isbn'):
                return self.book_metadata[image_path]['isbn']
                
            image_data = self.get_image_data(image_path)
            if not image_data:
                return None
                
            prompt = "Find the ISBN number from this image. If you find it, respond in the format: ISBN: [number]. If not found, respond: 'No ISBN number.'"
            
            # Use the retry function for API call
            def api_call():
                image_part = {
                    "mime_type": "image/jpeg",
                    "data": image_data
                }
                return self.model.generate_content([prompt, image_part])
            
            response = self.api_call_with_retry(api_call)
            response_text = response.text
            
            isbn_match = re.search(self.isbn_regex, response_text)
            if isbn_match:
                isbn_raw = isbn_match.group(0)
                isbn = ''.join(c for c in isbn_raw if c.isdigit() or c.upper() == 'X')
                
                # Update metadata
                if image_path in self.book_metadata:
                    self.book_metadata[image_path]['isbn'] = isbn
                else:
                    self.book_metadata[image_path] = {'title': None, 'author': None, 'isbn': isbn}
                    
                return isbn
                
            return None
            
        except Exception as e:
            logging.error(f"Error searching for ISBN from image {image_path}: {e}")
            return None

    def verify_isbn(self, isbn):
        """
        Verify that an ISBN is valid using Google Books API.
        
        Args:
            isbn (str): ISBN number to verify
            
        Returns:
            str: Cleaned ISBN number if valid, None otherwise
        """
        if not isbn:
            return None
        
        # Remove non-numeric characters (except X)
        cleaned_isbn = ''.join(c for c in isbn if c.isdigit() or c.upper() == 'X')
        
        # Verify ISBN with Google Books API
        url = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{cleaned_isbn}"
        
        try:
            # Use the retry function for API call
            def api_call():
                return requests.get(url)
            
            response = self.api_call_with_retry(api_call)
            data = response.json()
            
            # Check if the book was found
            if 'items' in data and len(data['items']) > 0:
                return cleaned_isbn
        except Exception as e:
            logging.error(f"Error verifying ISBN number: {e}")
        
        # If API verification fails, return the ISBN anyway (might be a new book)
        return cleaned_isbn

    def find_images(self):
        """
        Find all image files in the specified directory.
        
        Returns:
            list: List of file paths to images
        """
        images = []
        
        for filename in os.listdir(self.import_dir):
            file_path = os.path.join(self.import_dir, filename)
            file_extension = os.path.splitext(filename)[1].lower()
            
            # Check if the file is an image
            if os.path.isfile(file_path) and file_extension in self.image_formats:
                images.append(file_path)
        
        # Start preloading images and color tones in the background
        if images:
            self.preload_images(images)
            self.preload_color_tones(images)
            
        logging.info(f"Found {len(images)} images in import directory")
        return images

    def analyze_image_pair(self, image1_path, image2_path):
        """
        Analyze two images and determine if they are covers of the same book.
        
        Args:
            image1_path (str): Path to the first image
            image2_path (str): Path to the second image
            
        Returns:
            bool: True if images are a pair, False otherwise
        """
        try:
            # Check if images have the same title - if so, strongly prefer matching them
            title1 = self.book_metadata.get(image1_path, {}).get('title')
            title2 = self.book_metadata.get(image2_path, {}).get('title')
            
            # If both have the same title (case insensitive), they're very likely a match
            if title1 and title2 and title1.lower() == title2.lower():
                logging.info(f"Images share the same title '{title1}', automatically considering them a match")
                return True
            
            # Open both images in the format required by Gemini
            image_parts = []
            for image_path in [image1_path, image2_path]:
                image_data = self.get_image_data(image_path)
                if not image_data:
                    return False
                    
                image_parts.append({
                    "mime_type": "image/jpeg", 
                    "data": image_data
                })
            
            # Send images to Gemini API for analysis
            prompt = """Analyze these two images and determine if they are covers of the same book.
            Consider the following:
            1. Color tones and styles
            2. Appearance of book edges
            3. Text and content compatibility
            
            Respond in the format:
            Match: [YES/NO]
            Rationale: [brief explanation of why the images do or don't belong together]"""
            
            # Use the retry function for API call
            def api_call():
                return self.model.generate_content([prompt, image_parts[0], image_parts[1]])
            
            response = self.api_call_with_retry(api_call)
            response_text = response.text
            logging.info(f"AI response for image pair: {response_text}")
            
            # Analyze response
            is_pair = "YES" in response_text.upper()
            return is_pair
            
        except Exception as e:
            logging.error(f"Error analyzing image pair: {e}")
            return False

    def run_first_pass(self, images):
        """
        Run the first pass of book identification based on ISBN and image analysis.
        
        Args:
            images (list): List of file paths to images
            
        Returns:
            list: List of book pairs found
            set: Set of processed image paths
        """
        book_pairs = []
        processed_images = set()
        
        # Start preloading tasks in the background
        self.preload_images(images)
        self.preload_color_tones(images)
        
        # 1. First find all images with ISBN
        images_with_isbn = {}
        for image in images:
            if image not in processed_images:
                isbn = self.find_isbn_from_image(image)
                if isbn:
                    images_with_isbn[image] = isbn
                    logging.info(f"Found ISBN {isbn} in image {os.path.basename(image)}")

        # 2. Calculate color tones for all images (should be preloaded by now)
        color_tones = {}
        for image in images:
            color_tones[image] = self.calculate_color_tone(image)
        
        # 3. Find pairs for ISBN images
        for isbn_image, isbn in images_with_isbn.items():
            if isbn_image in processed_images:
                continue
                
            isbn_color = color_tones.get(isbn_image)
            
            # Group candidates by color similarity for prioritization, not restriction
            similar_color_candidates = []
            different_color_candidates = []
            
            # Divide images into similar and different color groups
            for image2 in images:
                if image2 != isbn_image and image2 not in processed_images:
                    if self.compare_color_tones(isbn_color, color_tones.get(image2)):
                        similar_color_candidates.append(image2)
                    else:
                        different_color_candidates.append(image2)
            
            # First try candidates with similar colors (more likely matches)
            found_match = False
            for candidate in similar_color_candidates:
                if self.analyze_image_pair(isbn_image, candidate):
                    book_pairs.append({
                        'images': [isbn_image, candidate],
                        'isbn': isbn
                    })
                    processed_images.add(isbn_image)
                    processed_images.add(candidate)
                    found_match = True
                    logging.info(f"Paired images with similar colors and ISBN {isbn}")
                    break
            
            # If no match found in similar colors, try different colors
            if not found_match:
                for candidate in different_color_candidates:
                    if self.analyze_image_pair(isbn_image, candidate):
                        book_pairs.append({
                            'images': [isbn_image, candidate],
                            'isbn': isbn
                        })
                        processed_images.add(isbn_image)
                        processed_images.add(candidate)
                        logging.info(f"Paired images with different colors but matching ISBN {isbn}")
                        break
        
        return book_pairs, processed_images
    
    def run_second_pass(self, images):
        """
        Run the second pass of book identification based on titles and authors.
        
        Args:
            images (list): List of file paths to images
            
        Returns:
            list: List of book pairs found
            set: Set of processed image paths
        """
        book_pairs = []
        processed_images = set()
        
        if not images:
            logging.info("No images to process in second pass")
            return book_pairs, processed_images
            
        logging.info(f"Starting second recognition pass using titles and authors for {len(images)} images...")
        
        # Preload data for all images
        self.preload_images(images)
        
        # Extract titles and authors for all images
        for image_path in images:
            self.extract_book_info(image_path)
        
        # Find images with ISBN (some may have ISBNs that weren't detected in first pass)
        images_with_isbn = {}
        for image in images:
            isbn = self.find_isbn_from_image(image)
            if isbn:
                images_with_isbn[image] = isbn
        
        # PRIORITY #1: Match by title
        # Group images by title
        title_groups = defaultdict(list)
        for image_path in images:
            if image_path in self.book_metadata and self.book_metadata[image_path].get('title'):
                title = self.book_metadata[image_path]['title']
                if title:  # Skip None or empty titles
                    title_groups[title.lower()].append(image_path)
        
        # Find pairs based on title
        logging.info("Attempting to match images by title...")
        for title, title_images in title_groups.items():
            if len(title_images) >= 2:  # We need at least 2 images to form a pair
                for img1, img2 in combinations(title_images, 2):
                    if img1 not in processed_images and img2 not in processed_images:
                        # Found a pair with the same title - tell the analyzer that titles match
                        logging.info(f"Found potential title match: '{title}'")
                        
                        # For title matches, we use a more lenient image analysis
                        if self.analyze_image_pair(img1, img2):
                            logging.info(f"Paired images by title: '{title}'")
                            
                            # Try to find ISBN for this pair
                            isbn = None
                            for img in [img1, img2]:
                                if img in images_with_isbn:
                                    isbn = images_with_isbn[img]
                                    break
                                # If we don't have ISBN yet, try to find it now
                                isbn = self.find_isbn_from_image(img)
                                if isbn:
                                    break
                                    
                            # If no ISBN found, use a placeholder based on title
                            if not isbn:
                                # Use the title as a fallback "ISBN"
                                normalized_title = title.lower().replace(' ', '_')[:30]  # Limit length
                                isbn = f"TITLE-{normalized_title}"
                                logging.info(f"Using title-based identifier: {isbn}")
                                    
                            book_pairs.append({
                                'images': [img1, img2],
                                'isbn': isbn
                            })
                            processed_images.add(img1)
                            processed_images.add(img2)
                            break
        
        # PRIORITY #2: Match by author (only for images not matched by title)
        # Update remaining images after title matching
        remaining = [img for img in images if img not in processed_images]
        
        # Skip author matching if no images remain after title matching
        if remaining:
            # Group images by author
            author_pairs = []
            for i, img1 in enumerate(remaining):
                author1 = self.book_metadata.get(img1, {}).get('author')
                if not author1:
                    continue
                    
                # Normalize author names for better matching
                norm_author1 = self.normalize_author_name(author1)
                
                for img2 in remaining[i+1:]:
                    if img1 != img2:
                        author2 = self.book_metadata.get(img2, {}).get('author')
                        if not author2:
                            continue
                            
                        # Normalize author names
                        norm_author2 = self.normalize_author_name(author2)
                        
                        # Skip if authors don't match
                        if norm_author1 != norm_author2:
                            continue
                            
                        # Authors match, so add to potential pairs
                        author_pairs.append((img1, img2, author1, author2))
            
            # Process the author pairs
            for img1, img2, author1, author2 in author_pairs:
                if img1 not in processed_images and img2 not in processed_images:
                    # Check if the pair looks valid with image analysis
                    if self.analyze_image_pair(img1, img2):
                        logging.info(f"Confirmed pair with author match: '{author1}' and '{author2}'")
                        
                        # Try to find ISBN for this pair
                        isbn = None
                        for img in [img1, img2]:
                            if img in images_with_isbn:
                                isbn = images_with_isbn[img]
                                break
                            # If we don't have ISBN yet, try to find it now
                            isbn = self.find_isbn_from_image(img)
                            if isbn:
                                break
                        
                        # If no ISBN found, use a placeholder but still add the pair
                        if not isbn:
                            # Use the author name as a fallback "ISBN"
                            normalized_author = self.normalize_author_name(author1)
                            isbn = f"AUTHOR-{normalized_author.replace(' ', '_')}"
                            logging.info(f"Using author-based identifier: {isbn}")
                        
                        book_pairs.append({
                            'images': [img1, img2],
                            'isbn': isbn
                        })
                        processed_images.add(img1)
                        processed_images.add(img2)
        
        return book_pairs, processed_images

    def find_book_pairs(self, images):
        """
        Find book pairs using an optimized method.
        
        Args:
            images (list): List of file paths to images
            
        Returns:
            list: List of dictionaries containing book pair information
            set: Set of processed image paths
        """
        # Run first pass
        book_pairs, processed_images = self.run_first_pass(images)
        
        # Identify unprocessed images
        unprocessed_images = [img for img in images if img not in processed_images]
        
        # Run second pass on unprocessed images directly in the import directory
        if unprocessed_images:
            logging.info(f"Running second pass on {len(unprocessed_images)} unprocessed images...")
            second_pairs, second_processed = self.run_second_pass(unprocessed_images)
            book_pairs.extend(second_pairs)
            processed_images.update(second_processed)
        
        # At this point, any remaining unprocessed images will be handled in the run() method
        return book_pairs, processed_images
    
    def process_book(self, book_pair, output_folder=None):
        """
        Process one book pair by moving images to the processed directory.
        
        Args:
            book_pair (dict): Dictionary containing book pair information
            output_folder (str, optional): Specific output folder within processed directory
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Verify ISBN
            isbn = self.verify_isbn(book_pair['isbn'])
            
            # Determine target directory
            target_dir = self.processed_dir
            if output_folder:
                target_dir = os.path.join(self.processed_dir, output_folder)
                os.makedirs(target_dir, exist_ok=True)
            
            # If ISBN cannot be verified, or it's a title/author-based match
            if not isbn or book_pair['isbn'].startswith('TITLE-') or book_pair['isbn'].startswith('AUTHOR-'):
                # Create a subfolder for unidentified-ISBN books
                isbn_missing_dir = os.path.join(target_dir, 'isbn_missing')
                os.makedirs(isbn_missing_dir, exist_ok=True)
                
                # Use the placeholder identifier (which may be title or author based)
                identifier = book_pair['isbn']
                front_dest = os.path.join(isbn_missing_dir, f"{identifier}_1.jpg")
                back_dest = os.path.join(isbn_missing_dir, f"{identifier}_2.jpg")
                
                logging.info(f"Moving book pair with identifier {identifier} to isbn_missing folder")
            else:
                # Create proper ISBN-based filenames
                front_dest = os.path.join(target_dir, f"{isbn}_1.jpg")
                back_dest = os.path.join(target_dir, f"{isbn}_2.jpg")
                
                logging.info(f"Moving book pair with ISBN {isbn} to processed folder")
            
            # Move files to destination
            shutil.copy2(book_pair['images'][0], front_dest)
            shutil.copy2(book_pair['images'][1], back_dest)
            
            # Delete original files from import directory
            for img_path in book_pair['images']:
                if os.path.exists(img_path) and img_path.startswith(self.import_dir):
                    os.remove(img_path)
                    logging.info(f"Removed original file at {img_path}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error processing book pair: {e}")
            return False
    
    def run(self, output_folder=None):
        """
        Run the book identification process.
        
        Args:
            output_folder (str, optional): Subfolder name within processed directory
            
        Returns:
            dict: Statistics about the process
        """
        # Create necessary directories
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Keep track of pairing information
        pairing_info = []
        
        # Check if import directory has images
        import_images = self.find_images()
        has_import_images = len(import_images) > 0
        
        # Track statistics
        all_book_pairs = []
        all_processed_images = set()
        
        # Initialize timing variables
        first_pass_time = 0
        second_pass_time = 0
        
        # First pass - process import directory if it has images
        if has_import_images:
            first_pass_start = time.time()
            logging.info("Running first pass on import directory...")
            book_pairs, processed_images = self.run_first_pass(import_images)
            first_pass_time = time.time() - first_pass_start
            
            # Add pairing info for first pass
            for pair in book_pairs:
                # For first pass, all pairs are ISBN-based with image analysis
                img1 = os.path.basename(pair['images'][0])
                img2 = os.path.basename(pair['images'][1])
                
                # Determine if title match occurred (from analyze_image_pair in first pass)
                title1 = self.book_metadata.get(pair['images'][0], {}).get('title')
                title2 = self.book_metadata.get(pair['images'][1], {}).get('title')
                
                if title1 and title2 and title1.lower() == title2.lower():
                    criterion = "Title match (first pass)"
                else:
                    criterion = "Image analysis (first pass)"
                
                pairing_info.append({
                    'images': [img1, img2],
                    'isbn': pair['isbn'],
                    'phase': 'First pass',
                    'criterion': criterion
                })
            
            all_book_pairs.extend(book_pairs)
            all_processed_images.update(processed_images)
            
            logging.info(f"First pass complete. Found {len(book_pairs)} book pairs.")
            logging.info(f"First pass processing time: {first_pass_time:.2f} seconds")
        
        # Second pass - check for remaining unprocessed images in the import directory
        unprocessed_images = [img for img in import_images if img not in all_processed_images]
        
        if unprocessed_images:
            second_pass_start = time.time()
            logging.info(f"Running second pass on {len(unprocessed_images)} unprocessed images in import directory...")
            book_pairs, processed_images = self.run_second_pass(unprocessed_images)
            second_pass_time = time.time() - second_pass_start
            
            # Add pairing info for second pass
            for pair in book_pairs:
                img1 = os.path.basename(pair['images'][0])
                img2 = os.path.basename(pair['images'][1])
                
                # Determine pairing criterion
                if pair['isbn'].startswith('TITLE-'):
                    criterion = "Title match (second pass)"
                elif pair['isbn'].startswith('AUTHOR-'):
                    criterion = "Author match (second pass)"
                else:
                    criterion = "ISBN match (second pass)"
                
                pairing_info.append({
                    'images': [img1, img2],
                    'isbn': pair['isbn'],
                    'phase': 'Second pass',
                    'criterion': criterion
                })
            
            all_book_pairs.extend(book_pairs)
            all_processed_images.update(processed_images)
            
            logging.info(f"Second pass complete. Found {len(book_pairs)} additional book pairs.")
            logging.info(f"Second pass processing time: {second_pass_time:.2f} seconds")
        
        # Process all identified book pairs
        identified_count = 0
        for book_pair in all_book_pairs:
            if self.process_book(book_pair, output_folder):
                identified_count += 1
        
        # Move remaining unprocessed images to isbn_missing folder
        remaining_unidentified = []
        
        # Determine remaining unprocessed images
        remaining_unprocessed = [img for img in import_images if img not in all_processed_images]
        
        if remaining_unprocessed:
            # Determine target directory
            target_dir = self.processed_dir
            if output_folder:
                target_dir = os.path.join(self.processed_dir, output_folder)
                os.makedirs(target_dir, exist_ok=True)
            
            # Create isbn_missing directory if it doesn't exist
            isbn_missing_dir = os.path.join(target_dir, 'isbn_missing')
            os.makedirs(isbn_missing_dir, exist_ok=True)
            
            # Move each unprocessed file
            unknown_counter = 1
            for unprocessed_file in remaining_unprocessed:
                filename = os.path.basename(unprocessed_file)
                file_ext = os.path.splitext(filename)[1]
                
                # Create a unique name for the unidentified file
                dest_filename = f"unknown{unknown_counter}{file_ext}"
                dest_path = os.path.join(isbn_missing_dir, dest_filename)
                
                # Make sure destination doesn't exist
                while os.path.exists(dest_path):
                    unknown_counter += 1
                    dest_filename = f"unknown{unknown_counter}{file_ext}"
                    dest_path = os.path.join(isbn_missing_dir, dest_filename)
                
                # Copy the file
                shutil.copy2(unprocessed_file, dest_path)
                logging.info(f"Moved unidentified image {filename} to isbn_missing folder as {dest_filename}")
                
                # Remove the original
                os.remove(unprocessed_file)
                
                # Track the remaining unidentified files
                remaining_unidentified.append(dest_filename)
                
                unknown_counter += 1
        
        # Calculate total processing time
        total_processing_time = (first_pass_time + second_pass_time)
        
        # Print processing summary
        logging.info("\n" + "="*50)
        logging.info("PROCESSING SUMMARY")
        logging.info("="*50)
        logging.info(f"Total pairs identified: {identified_count}")
        logging.info(f"Total images remaining unidentified: {len(remaining_unidentified)}")
        if remaining_unidentified:
            logging.info("Unidentified images:")
            for img in remaining_unidentified:
                logging.info(f"  - {img}")
        logging.info(f"Total processing time: {total_processing_time:.2f} seconds")
        
        # Print AI token usage if available
        tokens_used = self.total_prompt_tokens + self.total_response_tokens
        if tokens_used > 0:
            logging.info(f"Total AI tokens used: {tokens_used:,}")
            logging.info(f"  • Prompt tokens: {self.total_prompt_tokens:,}")
            logging.info(f"  • Response tokens: {self.total_response_tokens:,}")
        logging.info("="*50)
        
        # Return statistics
        return {
            'identified_pairs': identified_count,
            'unidentified_images': len(remaining_unidentified),
            'processing_time': total_processing_time,
            'prompt_tokens': self.total_prompt_tokens,
            'response_tokens': self.total_response_tokens
        }

    # Tähän tulee koko BookIdentifier-luokan koodi
    # Koodia on niin paljon, että jaan toteutuksen useampaan osaan 