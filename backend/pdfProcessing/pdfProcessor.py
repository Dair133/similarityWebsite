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
from google import genai
from google.genai import types
from openai import OpenAI


class PDFProcessor:
    def __init__(self):
        # Initialize Anthropic client
        # self.client = anthropic.Anthropic()
        
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

    def ask_claude(self, pdfText: str, systemInstructions: str, api_key: str) -> str:
     try:
        message = {
            "model": "claude-3-5-haiku-20241022",
            "max_tokens": 1024,
            "system": systemInstructions,
            "messages": [
                {
                    "role": "user",
                    "content": f"{pdfText}"
                }
            ]
        }
        
        client = anthropic.Anthropic(api_key=api_key)
        
        # Get the response
        response = client.messages.create(**message)
        
        return response
        
     except Exception as e:
        # If something goes wrong, log it and raise the error
        self.logger.error(f"Error communicating with Claude: {str(e)}")
        raise
    
    def ask_gemini(self,api_key):
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=["Explain how AI works"],
        config=types.GenerateContentConfig(
        max_output_tokens=500,
        temperature=0.1
    )
)
        print(response.text)
        
        
        
    def ask_deepseek(self,api_key:str):
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

        response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello, please explain ai to me"},
    ],
        stream=False
)

        print(response.choices[0].message.content)
    
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

    def parse_haiku_output(self, output) -> Dict[str, Any]:
     """
    Parse the output from Claude/Haiku into a structure compatible with Semantic Scholar results.
    
    Args:
        output: Raw string response from Claude/Haiku
        
    Returns:
        dict: Structured paper information matching Semantic Scholar format
    """
     try:
        # Ensure output is a string
        if not isinstance(output, str):
            # Handle if it's a response object with content
            if hasattr(output, 'content') and output.content:
                if isinstance(output.content, list) and output.content and hasattr(output.content[0], 'text'):
                    output = output.content[0].text
                else:
                    output = str(output.content)
            else:
                # Try to convert to string
                output = str(output)
                
        # Initialize result structure matching Semantic Scholar's return format
        result = {
            'title': '',
            'authors': '',
            'abstract': '',
            'year': 0,
            'citation_count': 0,
            'reference_count': 0,
            'citations': '',
            'references': ''
        }
        
        # Check if the output contains the expected section header
        if 'SEMANTIC_SCHOLAR_INFO:' not in output:
            self.logger.error("Missing 'SEMANTIC_SCHOLAR_INFO:' section in Haiku output")
            return result
            
        # Extract the SEMANTIC_SCHOLAR_INFO section
        sections = output.split('SEMANTIC_SCHOLAR_INFO:')
        if len(sections) < 2:
            self.logger.error("Could not split on SEMANTIC_SCHOLAR_INFO section")
            return result
            
        scholar_section = sections[1].strip()
        
        # Extract individual fields
        # Use a more robust approach - look for field markers
        fields = {
            'TITLE:': 'title',
            'AUTHORS:': 'authors',
            'YEAR:': 'year',
            'ABSTRACT:': 'abstract',
            'CITATIONS:': 'citations',
            'REFERENCES:': 'references'
        }
        
        # Find the positions of each field in the text
        field_positions = {}
        for field in fields.keys():
            pos = scholar_section.find(field)
            if pos >= 0:
                field_positions[field] = pos
        
        # Sort fields by their position in the text
        sorted_fields = sorted(field_positions.items(), key=lambda x: x[1])
        
        # Extract each field's content
        for i, (field, pos) in enumerate(sorted_fields):
            # Find the end of this field (start of next field or end of text)
            if i < len(sorted_fields) - 1:
                end_pos = sorted_fields[i + 1][1]
            else:
                end_pos = len(scholar_section)
                
            # Extract the field content
            content = scholar_section[pos + len(field):end_pos].strip()
            
            # Remove trailing semicolons and clean up
            if content.endswith(';'):
                content = content[:-1].strip()
                
            field_name = fields[field]
            
            # Special handling for different field types
            if field_name == 'year':
                try:
                    result[field_name] = int(content) if content else 0
                except ValueError:
                    result[field_name] = 0
            elif field_name in ['authors', 'citations', 'references']:
                # These fields might be lists
                if field_name == 'authors':
                    items = [item.strip() for item in content.split(',') if item.strip()]
                    result[field_name] = ', '.join(items)  # Join with comma for authors
                else:
                    items = [item.strip() for item in content.split(';') if item.strip()]
                    result[field_name] = '; '.join(items)  # Join with semicolon for citations/references
                
                # Set counts
                if field_name == 'citations':
                    result['citation_count'] = len(items)
                elif field_name == 'references':
                    result['reference_count'] = len(items)
            else:
                result[field_name] = content
                
        self.logger.info(f"Successfully parsed Haiku output: title={result['title']}, " +
                         f"abstract length={len(result['abstract'])} chars")
        return result
        
     except Exception as e:
        self.logger.error(f"Error parsing Haiku output: {str(e)}")
        # Return empty structure on error, matching Semantic Scholar format
        return {
            'title': '',
            'authors': '',
            'abstract': '',
            'year': 0,
            'citation_count': 0,
            'reference_count': 0,
            'citations': '',
            'references': ''
        }
        

    def remove_duplicates(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen_titles = set()
        unique_papers = []
        
        for paper in papers:
            title = paper.get('title', '').strip().lower()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_papers.append(paper)
        
        return unique_papers
 
 
 
 
  
    
    
