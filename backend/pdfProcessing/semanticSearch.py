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
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import requests
import time
class SemanticScholar:
    def __init__(self):
        self.max_workers = 5
        self.session = requests.Session()
        self.base_delay = 0.2  # Base delay between requests
        
        
    def get_paper_details(self, paper_id: str, api_key: str) -> Dict[str, Any]:
        url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
        headers = {"x-api-key": api_key}

        fields = "title,abstract,year,citationCount,authors.name,citations.title,references.title"
        url = f"{url}?fields={fields}"

        max_retries = 7
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
                    print(f"Rate limit hit inside get paper details, waiting {wait_time} seconds before retry {retry_count}")
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
                        print(f"Rate limit hit in return by title, waiting {wait_time} seconds before retry {retry_count}")
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
            # If a paper does not have a valid abstract then return none
            # Abstracts are too important to the overall calcualtion of similarity to do without them
            if paper_data['abstract'] == None:
                return None
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
                print(f"Rate limit hit inside search papers on string, waiting {wait_time} seconds before retry {retry_count}.")
                time.sleep(wait_time)
                continue
            else:
                print(f"HTTP Error: {e}")
                return []
        except Exception as e:
            print(f"Error: {e}")
            return []
        
        
    def _make_request(self, url: str, headers: Dict, params: Dict = None, max_retries: int = 5) -> Dict:
     retry_count = 0
     base_wait_time = 1.5  # Starting wait time
    
     while retry_count < max_retries:
        try:
            # Small delay to prevent API spam
            if retry_count == 0:  # Only add base delay on first attempt
                time.sleep(self.base_delay)
                
            response = self.session.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            # If successful, return the data
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Rate limit hit
                retry_count += 1
                if retry_count == max_retries:
                    print(f"Failed after {max_retries} retries due to rate limiting")
                    return {'data': []}
                    
                # Linear increase in wait time - add 0.5 seconds each retry
                wait_time = base_wait_time + (0.5 * (retry_count - 1))
                print(f"Rate limit (429) hit, waiting {wait_time} seconds before retry {retry_count}")
                time.sleep(wait_time)
                continue
            else:
                # Other HTTP errors, return empty result
                print(f"HTTP Error: {e.response.status_code}")
                return {'data': []}
                
        except Exception as e:
            print(f"Request error: {str(e)}")
            return {'data': []}
            
     return {'data': []}
            
            
    def search_papers_parallel(self, search_terms: List[Dict], api_key: str) -> List[Dict]:
     """
    Parallel search that tracks which terms found which papers
    Returns papers with information about which search terms found them
    Ensures all returned papers have valid abstracts
     """
     def search_single_term(term_info: Dict) -> List[Dict]:
         try:
            if not term_info or 'term' not in term_info:
                print(f"Invalid term_info: {term_info}")
                return []
                
            url = "https://api.semanticscholar.org/graph/v1/paper/search"
            headers = {"x-api-key": api_key}
            fields = "title,abstract,year,citationCount,authors.name,citations.title,references.title"
            
            params = {
                "query": term_info['term'],
                "limit": 20,  # Increased from 10 to have more candidates to filter
                "fields": fields
            }
            
            result = self._make_request(url, headers, params)
            
            if not result:
                print(f"No result for term: {term_info['term']}")
                return []
                
            papers_data = result.get('data', [])
            
            # Filter for papers with valid abstracts
            valid_papers = []
            for paper in papers_data:
                if not paper:
                    continue
                
                # Check if this paper has a valid abstract
                abstract = paper.get("abstract")
                if abstract and isinstance(abstract, str) and len(abstract) >= 15:
                    # Create a source info dictionary to track where this paper came from
                    source_info = {
                        "search_term": term_info['term'],
                        "search_type": term_info['type']
                    }
                    
                    paper_info = {
                        "source_info": source_info,  # Add the source tracking information
                        "paper_info": {
                            "title": paper.get("title", "N/A"),
                            "abstract": abstract,
                            "year": paper.get("year", "N/A"),
                            "citation_count": paper.get("citationCount", 0),
                            "reference_count": paper.get("referenceCount", 0),
                            "authors": [author.get("name", "N/A") for author in paper.get("authors", [])],
                            "citations": [citation.get("title", "N/A") for citation in paper.get("citations", [])],
                            "references": [reference.get("title", "N/A") for reference in paper.get("references", [])]
                        }
                    }
                    valid_papers.append(paper_info)

            if len(valid_papers) < 5 and papers_data:
                for paper in papers_data:
                    # Skip papers we've already processed
                    if any(p['paper_info']['title'] == paper.get('title') for p in valid_papers):
                        continue
                        
                    paper_id = paper.get('paperId')
                    if not paper_id:
                        continue
                    
                    # Get detailed paper info
                    paper_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
                    paper_params = {"fields": fields}
                    
                    paper_result = self._make_request(paper_url, headers, paper_params)
                    if not paper_result:
                        continue
                    
                    # Check for valid abstract
                    detailed_abstract = paper_result.get("abstract")
                    if detailed_abstract and isinstance(detailed_abstract, str) and len(detailed_abstract) >= 15:
                        source_info = {
                            "search_term": term_info['term'],
                            "search_type": term_info['type']
                        }
                        
                        paper_info = {
                            "source_info": source_info,
                            "paper_info": {
                                "title": paper_result.get("title", "N/A"),
                                "abstract": detailed_abstract,
                                "year": paper_result.get("year", "N/A"),
                                "citation_count": paper_result.get("citationCount", 0),
                                "reference_count": paper_result.get("referenceCount", 0),
                                "authors": [author.get("name", "N/A") for author in paper_result.get("authors", [])],
                                "citations": [citation.get("title", "N/A") for citation in paper_result.get("citations", [])],
                                "references": [reference.get("title", "N/A") for reference in paper_result.get("references", [])]
                            }
                        }
                        valid_papers.append(paper_info)
                        
                        # Once we have enough valid papers, stop looking
                        if len(valid_papers) >= 5:
                            break
            

            final_valid_papers = []
            for paper in valid_papers:
                abstract = paper['paper_info'].get('abstract')
                if abstract and isinstance(abstract, str) and len(abstract) >= 15:
                    final_valid_papers.append(paper)
                    
            print(f"Found {len(final_valid_papers)} valid papers for term: {term_info['term']}")
            return final_valid_papers
            
         except Exception as e:
            print(f"Error processing term {term_info.get('term', 'unknown')}: {str(e)}")
            return []

     try:
        # Validate input
        if not search_terms:
            print("No search terms provided")
            return []
            
        print(f"Starting parallel search with {len(search_terms)} terms")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            all_results = list(executor.map(search_single_term, search_terms))
        
        # Flatten and filter results
        flattened = [paper for sublist in all_results if sublist for paper in sublist]
        
        # Group papers by title to handle duplicates found by multiple terms
        papers_by_title = {}
        for paper in flattened:
            title = paper['paper_info']['title']
            if title not in papers_by_title:
                papers_by_title[title] = paper
                papers_by_title[title]['source_info'] = [paper['source_info']]
            else:
                # If we've seen this paper before, add the new source info
                papers_by_title[title]['source_info'].append(paper['source_info'])
        
        # Convert back to list
        final_papers = list(papers_by_title.values())
        
        # Final verification that all papers have valid abstracts
        valid_final_papers = []
        for paper in final_papers:
            abstract = paper['paper_info'].get('abstract')
            if abstract and isinstance(abstract, str) and len(abstract) >= 15:
                valid_final_papers.append(paper)
            else:
                print(f"Removing paper with invalid abstract: {paper['paper_info'].get('title')}")
        
        print(f"Total unique papers with valid abstracts: {len(valid_final_papers)}/{len(final_papers)}")
        return valid_final_papers
        
     except Exception as e:
        print(f"Error in search_papers_parallel: {str(e)}")
        return []
    
    
    
    
    
    
    
    
    
    def _make_request_ALEX(self, url: str, params: Dict) -> Dict:
     """Helper function that wraps requests.get with better error reporting."""
     try:
        print(f"Making request to {url} with params {params}")  # Debug logging
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        print(f"Got response with {len(data.get('results', []))} results")  # Debug logging
        return data
     except requests.exceptions.HTTPError as e:
        print(f"HTTP Error for URL {url}: {e.response.status_code} - {e.response.text}")
        return {"results": []}
     except Exception as e:
        print(f"Request failed for URL {url}: {str(e)}")
        return {"results": []}

    def decode_inverted_index(self, inverted_index: Dict) -> str:
        """
        Reconstruct the abstract from OpenAlex's inverted index format.
        Returns a plain text abstract or None if not possible.
        """
        if not inverted_index:
            return None
        max_index = 0
        for positions in inverted_index.values():
            if positions:
                max_index = max(max_index, max(positions))
        words_list = [""] * (max_index + 1)
        for word, positions in inverted_index.items():
            for pos in positions:
                words_list[pos] = word
        abstract_text = " ".join(words_list).strip()
        return abstract_text if abstract_text else None


    def get_citations(self, work_id: str, limit: int = 20) -> List[str]:
     """
    Retrieves a list of citation titles for a given work.
    """
     base_url = "https://api.openalex.org/works"
    # Ensure work_id is properly formatted
     if not work_id.startswith("https://openalex.org/"):
        work_id = f"https://openalex.org/{work_id}"
        
     params = {
        "filter": f"cites:{work_id}",
        "per-page": limit,
        "fields": "display_name"
    }
    
     print(f"Fetching citations for {work_id}")  # Debug
     result = self._make_request_ALEX(base_url, params)
     if not result or "results" not in result:
        return []
        
     citations = result.get("results", [])
     citation_titles = [citation.get("display_name", "N/A") for citation in citations]
     return citation_titles

    def search_papers_parallel_ALEX(self, search_terms: List[Dict], desired_papers: int) -> List[Dict]:
        """
        Parallel search using OpenAlex.
        For each search term (a dict with keys "term", "type", and optionally "weight"),
        queries OpenAlex for papers and returns a list of dictionaries with valid paper information.
        Ensures all returned papers have a valid abstract (>=15 characters).
        Each returned record has:
            - source_info: { "search_term": ..., "search_type": ... }
            - paper_info: {
                  "title": ..., "abstract": ..., "year": ...,
                  "citation_count": ...,
                  "authors": [...],
                  "references": [...],
                  "citations": [...]  # List of citing paper titles
              }
        """
        base_url = "https://api.openalex.org/works"

        fields = "display_name,abstract_inverted_index,publication_year,cited_by_count,authorships,referenced_works"
        per_page = 20  # Adjust as needed

        def search_single_term(term_info: Dict) -> List[Dict]:
            valid_papers = []
            if not term_info or 'term' not in term_info:
                print(f"Invalid term_info: {term_info}")
                return []
            page = 1
            # Continue paginating until we collect the desired number of valid papers
            while len(valid_papers) < desired_papers:
                params = {
                    "search": term_info['term'],
                    "page": page,
                    "per-page": per_page,
                    "select": fields
                }
                result = self._make_request_ALEX(base_url, params)
                if not result:
                    #print(f"No result for term: {term_info['term']} on page {page}")
                    break
                works = result.get("results", [])
                if not works:
                    # No more results available
                    break

                for work in works:
                    if not work:
                        continue
                    abstract_inverted = work.get("abstract_inverted_index")
                    abstract = self.decode_inverted_index(abstract_inverted)
                    if not abstract or not (isinstance(abstract, str) and len(abstract) >= 15):
                        continue  # Skip if abstract is missing or too short

                    # Build source info for tracking
                    source_info = {
                        "search_term": term_info['term'],
                        "search_type": term_info.get("type", "N/A")
                    }
                    # Extract basic fields from OpenAlex
                    title = work.get("display_name", "N/A")
                    year = work.get("publication_year", "N/A")
                    citation_count = work.get("cited_by_count", 0)
                    # Extract authors from the authorships list
                    authors_list = work.get("authorships", [])
                    authors = []
                    for auth in authors_list:
                        author = auth.get("author", {})
                        if "display_name" in author:
                            authors.append(author["display_name"])
                    # References: these are provided as a list of OpenAlex IDs
                    references = work.get("referenced_works", [])

                    # Get citations (list of citing paper titles) via an extra request.
                    # Note: This extra request is made per paper.
                    work_id = work.get("id", "")
                    citations = self.get_citations(work_id) if work_id else []

                    paper_info = {
                        "source_info": source_info,
                        "paper_info": {
                            "title": title,
                            "abstract": abstract,
                            "year": year,
                            "citation_count": citation_count,
                            "authors": authors,
                            "references": references,
                            "citations": citations
                        }
                    }
                    valid_papers.append(paper_info)
                    if len(valid_papers) >= desired_papers:
                        break

                #print(f"Term '{term_info['term']}' page {page} processed: {len(valid_papers)} valid papers collected so far.")
                page += 1
                time.sleep(1)  # Be polite to the API
            #print(f"Found {len(valid_papers)} valid papers for term: {term_info['term']}")
            return valid_papers

        all_valid_papers = []
        # Use ThreadPoolExecutor to run the term searches in parallel.
        with ThreadPoolExecutor(max_workers=len(search_terms)) as executor:
            futures = {executor.submit(search_single_term, term): term for term in search_terms}
            for future in as_completed(futures):
                term = futures[future]
                try:
                    papers = future.result()
                    all_valid_papers.extend(papers)
                except Exception as e:
                    print(f"Error processing term {term.get('term', 'unknown')}: {e}")
        return all_valid_papers
        