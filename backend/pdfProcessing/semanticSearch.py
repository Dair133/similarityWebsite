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
import requests
import time
from typing import List, Dict, Any
from torch.nn.functional import cosine_similarity

class SemanticScholar:

    def get_paper_details(self, paper_id: str, api_key: str) -> Dict[str, Any]:
        url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
        headers = {"x-api-key": api_key}

        fields = "title,abstract,year,citationCount,authors.name,citations.title,references.title"
        url = f"{url}?fields={fields}"

        max_retries = 5
        retry_count = 0
        retry_delay = 2  # Start with 2 second delay

        while retry_count < max_retries:
            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Rate limit error
                    retry_count += 1
                    if retry_count == max_retries:
                        print(f"Failed after {max_retries} retries due to rate limiting")
                        return None

                    # Exponential backoff: increase delay each retry
                    wait_time = retry_delay * (2 ** (retry_count - 1))
                    print(f"Rate limit hit, waiting {wait_time} seconds before retry {retry_count}")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"Error getting paper details: {e}")
                    return None
            except Exception as e:
                print(f"Error getting paper details: {e}")
                return None

   
   
    def return_info_by_title(self, title: str, api_key: str) -> Dict[str, Any]:
        try:
            # Search for the paper
            search_url = "https://api.semanticscholar.org/graph/v1/paper/search"
            headers = {"x-api-key": api_key}

            params = {
       "query": title,
              "limit": 1,
                              "fields": "paperId,title,abstract,year,citationCount,authors.name,citations.title,references.title"
}

            max_retries = 5
            retry_count = 0
            retry_delay = 1  # Start with 2 second delay

            while retry_count < max_retries:
                try:
                    response = requests.get(search_url, headers=headers, params=params)
                    response.raise_for_status()
                    search_data = response.json()
                    break  # If successful, exit the retry loop
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 429:  # Rate limit error
                        retry_count += 1
                        if retry_count == max_retries:
                            print(f"Failed after {max_retries} retries due to rate limiting")
                            return None

                        # Exponential backoff: increase delay each retry
                        wait_time = retry_delay * (2 ** (retry_count - 1))
                        print(f"Rate limit hit, waiting {wait_time} seconds before retry {retry_count}")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"Error searching for paper: {e}")
                        return None
                except Exception as e:
                    print(f"Error searching for paper: {e}")
                    return None

            if not search_data.get('data') or len(search_data['data']) == 0:
                # print(f"No results found for title: {title}")
                return None

            paper_id = search_data['data'][0].get('paperId')
            if not paper_id:
                return None

            paper_data = self.get_paper_details(paper_id, api_key)
            if not paper_data:
                return None

            return self.format_paper_data(paper_data)

        except Exception as e:
            print(f"Error searching for paper: {e}")
            return None
        
        
        
    def format_paper_data(self, paper_data: Dict) -> Dict[str, Any]:
     if not paper_data or not paper_data.get('title'):
         return None

     # Format authors
     authors = [author.get('name', '') for author in paper_data.get('authors', [])]
     authors_str = ', '.join(filter(None, authors))

     # Format citations
     citations = []
     for citation in paper_data.get('citations', []):
        if citation and citation.get('title'):
            citations.append(citation['title'])
     citations_str = '; '.join(citations) if citations else 'No citations'

    # Format references
     references = []
     for reference in paper_data.get('references', []):
         if reference and reference.get('title'):
             references.append(reference['title'])
     references_str = '; '.join(references) if references else 'No references'

     return {
        'title': paper_data.get('title'),
        'authors': authors_str,
        'abstract': paper_data.get('abstract'),
        'year': paper_data.get('year'),
        'citation_count': paper_data.get('citationCount', 0),
        'reference_count': len(references),
        'citations': citations_str,
        'references': references_str,
    }
     
     
     # Searches semantic scholar based on string entered.
    def search_papers_on_string(self, query: str, num_results: int, api_key: str) -> List[Dict[str, Any]]:
  
     url = "https://api.semanticscholar.org/graph/v1/paper/search"
     headers = {"x-api-key": api_key}
     fields = "title,abstract,year,citationCount,authors.name,citations.title,references.title"
    
     # Parameters for the search API
     params = {
        "query": query,
        "limit": num_results,
        "fields": fields
     }
    
     max_retries = 5
     retry_count = 0
     retry_delay = 2  # Start with 2-second delay

     while retry_count < max_retries:
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx, 5xx)
            
            # Return the list of papers
            data = response.json()
            papers = data.get("data", [])  # "data" contains the list of papers
            
            # Format papers into a simplified array
            formatted_papers = []
            for paper in papers:
                formatted_papers.append({
    "title": paper.get("title", "N/A"),
    "abstract": paper.get("abstract", "N/A"),
    "year": paper.get("year", "N/A"),
    "citation_count": paper.get("citationCount", 0),
    "reference_count": paper.get("referenceCount", 0),
    "authors": [author.get("name", "N/A") for author in paper.get("authors", [])],
    "citations": [citation.get("title", "N/A") for citation in paper.get("citations", [])],
    "references": [reference.get("title", "N/A") for reference in paper.get("references", [])]
})
            
            return formatted_papers
        
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Rate limit error
                retry_count += 1
                if retry_count == max_retries:
                    print(f"Failed after {max_retries} retries due to rate limiting.")
                    return []
                
                # Exponential backoff: increase delay each retry
                wait_time = retry_delay * (2 ** (retry_count - 1))
                print(f"Rate limit hit, waiting {wait_time} seconds before retry {retry_count}.")
                time.sleep(wait_time)
                continue
            else:
                print(f"HTTP Error: {e}")
                return []
        except Exception as e:
            print(f"Error: {e}")
            return []
        
        
        
        