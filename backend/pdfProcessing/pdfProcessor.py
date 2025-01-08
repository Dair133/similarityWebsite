# backend/services/pdf_processor.py
import anthropic
import PyPDF2
from PyPDF2 import PdfReader
import logging
import regex as re
from typing import List, Dict, Any
import pdfplumber
import time
import requests
from pathlib import Path
from flask import Blueprint
from dotenv import load_dotenv
import os



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
        
        # Debug information
        # print(f"Body text font size: {body_text_size}")
        #  print(f"Filtered text: {text[:200]}")
        
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
            # Initialize result structure
            result = {
                'title': '',
                'core_concepts': [],
                'core_methodologies': [],
                'related_methodologies': []
            }

            # Split main sections by semicolon
            sections = info_string.split(';')

            for section in sections:
                section = section.strip()
                
                # Parse title
                if section.startswith('TITLE:'):
                    # Remove TITLE: and any quotes
                    result['title'] = section[6:].strip("'")
                    
                # Parse core concepts    
                elif section.startswith('CORE_CONCEPTS:'):
                    # Remove section header and split by comma
                    concepts = section[13:].strip()
                    # Remove square brackets and split
                    concepts = [c.strip('[]') for c in concepts.split(',')]
                    result['core_concepts'] = [c for c in concepts if c]  # Remove empty strings
                    
                # Parse core methodologies    
                elif section.startswith('CORE_METHODOLOGIES:'):
                    methods = section[19:].strip()
                    methods = [m.strip('[]') for m in methods.split(',')]
                    result['core_methodologies'] = [m for m in methods if m]
                    
                # Parse related methodologies    
                elif section.startswith('RELATED_METHODOLOGIES:'):
                    rel_methods = section[22:].strip()
                    rel_methods = [rm.strip('[]') for rm in rel_methods.split(',')]
                    result['related_methodologies'] = [rm for rm in rel_methods if rm]

            return result
            
        except Exception as e:
            self.logger.error(f"Error parsing info string: {str(e)}")
            return None

