# backend/services/pdf_processor.py
import anthropic
import PyPDF2
from PyPDF2 import PdfReader
import logging
import regex as re
from typing import List, Dict, Any, Union
import pdfplumber
import time
import requests
from pathlib import Path
from flask import Blueprint
from dotenv import load_dotenv
import os
from sklearn.metrics.pairwise import cosine_similarity



class PDFProcessor:
    def __init__(self):
        # Initialize Anthropic client
        self.client = anthropic.Anthropic()
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def process_with_instructions(self, pdf_path: str, instructions: List[str]) -> Dict[str, Any]:
        try:
            # Extract text from PDF
            text = self._extract_text(pdf_path)
            
            # Process each instruction with Claude
            results = []
            for instruction in instructions:
                analysis = self._ask_claude(text, instruction)
                results.append({
                    'instruction': instruction,
                    'response': analysis
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing PDF with instructions: {str(e)}")
            raise

    def search_in_pdf(self, pdf_path: str, search_term: str) -> List[Dict[str, Any]]:
      
        try:
            # Extract text page by page
            reader = PdfReader(pdf_path)
            results = []
            
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                
                # Find all occurrences in this page
                start_idx = 0
                while True:
                    idx = text.lower().find(search_term.lower(), start_idx)
                    if idx == -1:
                        break
                        
                    # Get context around the found term
                    context_start = max(0, idx - 50)
                    context_end = min(len(text), idx + len(search_term) + 50)
                    
                    results.append({
                        'page': page_num,
                        'context': text[context_start:context_end],
                        'position': idx
                    })
                    
                    start_idx = idx + 1
                    
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching PDF: {str(e)}")
            raise
            # PDF is in binary format, this function converts to text
    def _extract_text_with_title(self, filepath):
     print("Extracting formatted text...")
     with pdfplumber.open(filepath) as pdf:
        first_page = pdf.pages[0]
        
        # Extract text with font information
        words = first_page.extract_words(
            x_tolerance=3,
            y_tolerance=3,
            keep_blank_chars=True,
            extra_attrs=['fontname', 'size']
        )
        
        # Find the most common large font size (often the body text)
        font_sizes = [float(word['size']) for word in words if word.get('size')]
        body_text_size = max(set(font_sizes), key=font_sizes.count)
        
        # Filter criteria
        MIN_FONT_MULTIPLIER = 1.2  # Title should be at least 20% larger than body text
        MIN_CHARS = 20  # Minimum title length
        MAX_CHARS = 300  # Maximum title length
        Y_THRESHOLD = first_page.height * 0.3  # Only look in top 30% of page
        
        # Apply filters
        filtered_words = [
            word for word in words
            if (float(word.get('size', 0)) >= body_text_size * MIN_FONT_MULTIPLIER  # Larger than body text
                and word['top'] <= Y_THRESHOLD  # In top portion of page
                and not any(x in word['text'].lower() for x in [
                    'abstract', 'introduction', 'copyright', 'license',
                    'http', 'www', '.com', '@', 'doi'
                ])  # Exclude common non-title text
            )
        ]
        
        # Sort by vertical position and font size
        filtered_words.sort(key=lambda w: (w['top'], -float(w.get('size', 0))))
        
        # Combine words into lines
        text = ' '.join(word['text'] for word in filtered_words)

        
        return text

    def ask_claude(self, text: str, instruction: str, api_key) -> str:
    
     try:
        # Create our message structure - combining the instruction and text
        message = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": f"{instruction}\n\nText:\n{text}"
                }
            ]
        }
        
        # Send our request to Claude
        client = anthropic.Anthropic(
            api_key= api_key # Replace with your actual API key
        )
        
        # Get the response
        response = client.messages.create(**message)
        
        return response
        
     except Exception as e:
        # If something goes wrong, log it and raise the error
        self.logger.error(f"Error communicating with Claude: {str(e)}")
        raise
    
      
    
    def search_word_context(pdf_path, search_word, context_length=150):
    # Open the PDF file in read-binary mode
     with open(pdf_path, 'rb') as file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(file)
        
        # Initialize empty string to store all text
        full_text = ""
        
        # Extract text from each page
        for page in pdf_reader.pages:
            full_text += page.extract_text()
        
        # Search for the word using regex (case-insensitive)
        pattern = f"({re.escape(search_word)}.{{0,{context_length}}})"
        matches = re.finditer(pattern, full_text, re.IGNORECASE)
        
        # Store and return all matches
        results = []
        for match in matches:
            results.append(match.group(0))
        
        return results
    
    
    
    def _extract_text(self, pdf_path: str) -> str:
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n\n"
            return text
        except Exception as e:
            self.logger.error(f"Error extracting text from PDF: {str(e)}")
            raise
        
 
    def parse_paper_info(self, info_string: str) -> Dict[str, Any]:
     try:
        result = {
            'title': '',
            'core_methodologies': [],
            'conceptual_angles': [],
            'random': []
        }

        sections = info_string.split('\n')
        
        for line in sections:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('TITLE:'):
                result['title'] = line[len('TITLE:'):].strip('[]').strip()
            elif line.startswith('CORE_METHODOLOGIES:'):  # Has colon
                content = line[len('CORE_METHODOLOGIES:'):].strip('[]').strip()
                if content:
                    result['core_methodologies'] = [
                        item.strip() for item in content.split(',') 
                        if item.strip()
                    ]
            elif line.startswith('CONCEPTUAL_ANGLES:'):  # Added colon here
                content = line[len('CONCEPTUAL_ANGLES:'):].strip('[]').strip()
                if content:
                    result['conceptual_angles'] = [
                        item.strip() for item in content.split(',') 
                        if item.strip()
                    ]
            elif line.startswith('RANDOM:'):  # Added colon here
                content = line[len('RANDOM:'):].strip('[]').strip()
                if content:
                    result['random'] = [
                        item.strip() for item in content.split(',') 
                        if item.strip()
                    ]

        self.logger.debug(f"Parsed paper info: {result}")
        return result
            
     except Exception as e:
        self.logger.error(f"Error parsing paper info: {str(e)}")
        return None

    def parse_haiku_output(output: str) -> Dict[str, Union[str, Dict]]:
     """
    Parse the output from Claude/Haiku into the same structure as Semantic Scholar output.
    
    Args:
        output (str): Raw output from Claude/Haiku
        
    Returns:
        dict: Structured paper information matching Semantic Scholar format
    """
    # Split into metadata and analysis sections
     sections = output.strip().split('\n\n')
     metadata_section = sections[0]
     analysis_section = sections[1]
     
    # Parse metadata section
     metadata = {}
     for line in metadata_section.split('\n')[1:]:  # Skip the "METADATA:" header
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.lower().strip()
            value = value.replace(';', '').strip()
            metadata[key] = value
    
     # Parse analysis section
     analysis = {}
     for line in analysis_section.split('\n')[1:]:  # Skip the "ANALYSIS:" header
        if ':' in line:
            key, value = line.split(':', 1)
            value = value.replace(';', '').strip()
            if value:  # If not empty
                analysis[key] = [item.strip() for item in value.split(',')]
            else:
                analysis[key] = []
    
     # Format in the same structure as semantic scholar output
     result = {
        'title': metadata.get('title', ''),
        'semantic_scholar_info': {
            'authors': [author.strip() for author in metadata.get('authors', '').split(',') if author.strip()],
            'abstract': metadata.get('abstract', ''),
            'year': int(metadata.get('year', 0)),
            'citation_count': int(metadata.get('citation_count', 0)),
            'reference_count': int(metadata.get('reference_count', 0)),
            'citations': [],  # Can't get actual citations without Semantic Scholar
            'references': []  # Can't get actual references without Semantic Scholar
        },
        'analysis': analysis
    }
    
     return result

    
    
