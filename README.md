# Book Identifier

Book Identifier is an AI-powered application that identifies book information (ISBN, title, author) from cover images and organizes them accordingly.

## Benefits

- **Automatic Image Pairing**: Matches front and back covers using ISBN numbers
- **E-commerce Integration**: ISBN-based file naming enables direct shop integration
- **Automated Descriptions**: Product details can be fetched automatically using ISBN data
- **Efficient Workflow**: Drastically reduces manual work in book listing process

By extracting ISBN numbers from book covers, the application creates a bridge between your physical book inventory and digital catalog systems, enabling automation of downstream processes like product description generation.

## Features

The project consists of two main scripts for different use cases:

### 1. `analyze_unsorted.py` - Process unsorted books

Processes unsorted book images that may be in any order:

- Requires exactly two images per book (front cover and back cover)
- Uses AI to identify which images are front and back covers of the same book
- Analyzes color tones to prioritize potential matches, optimizing the pairing process
- Extracts book information (ISBN, title, author) using AI
- Organizes identified books into folders by ISBN or title
- Renames image files using ISBN (or title when ISBN not found) for easier identification and future use

Usage:

```
python analyze_unsorted.py
```

The script automatically processes all images in the import directory. If subfolders are found in the import directory, each subfolder will be processed separately, and the results will be placed in a corresponding subfolder in the processed directory.

### 2. `analyze_sorted.py` - Process pre-numbered pairs

Processes images that are already numbered in sequence:

- Requires exactly two images per book (front cover and back cover)
- Images must be named so that they end with a sequential number
  - Example: `0702-Kirjat11-ET-1.jpg`, `0702-Kirjat11-ET-2.jpg`, `0702-Kirjat11-ET-3.jpg`, `0702-Kirjat11-ET-4.jpg`
  - The script pairs images by taking two consecutive numbers (1-2, 3-4, 5-6, etc.)
  - Only the number at the end of the filename matters; the rest of the name can be anything
  - The number must be the very last part of the filename before the extension
- Finds ISBN and book title from each pair
- Copies images to folders based on ISBN or title

Usage:

```
python analyze_sorted.py
```

## Folder Structure

- `import/`: Place your book images here for processing
  - For `analyze_unsorted.py`: Images can be placed directly here or in subfolders
  - For `analyze_sorted.py`: Create subfolders and place sequentially numbered images in them
- `processed/`: Processed book images will be saved here
  - Identified books are organized by ISBN (e.g., `9789510328620_1.jpg` and `9789510328620_2.jpg`)
  - Books without ISBN are saved in `isbn_missing` subfolder with title or sequential numbering

## Installation

1. Clone the project
2. Set up a virtual environment:

   ```
   # Create a virtual environment
   python -m venv book_identifier

   # Activate the virtual environment
   # On Windows:
   book_identifier\Scripts\activate
   # On macOS/Linux:
   source book_identifier/bin/activate
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```
4. Create a `.env` file and add your Google Gemini API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

## Requirements

- Python 3.9 or newer
- Google Gemini API key
- The following Python packages (installed via requirements.txt):
  - google-generativeai>=0.3.1
  - python-dotenv>=1.0.0
  - requests>=2.31.0
  - Pillow>=10.0.0
  - numpy>=1.24.0
- Image directory with book cover images

# Import Directory

Place your book images in this directory for processing.

- For `analyze_unsorted.py`: Images can be placed directly here or in subfolders
- For `analyze_sorted.py`: Create subfolders and place images in them

# Processed Directory

Processed book images will be saved here, organized by ISBN or in 'isbn_missing' subfolder.
