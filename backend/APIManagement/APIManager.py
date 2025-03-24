# backend/routes/upload_routes.py
# APP.py script
import logging
import time
from bs4 import BeautifulSoup
from flask import Blueprint, request, jsonify
from typing import Dict, Any, List
import numpy as np
import requests
import torch
from werkzeug.utils import secure_filename
# backend/app.py
from pdfProcessing.pdfProcessor import PDFProcessor  # Note the lowercase 'p' in processor


# Import for model runners
from modelFolder.modelRunners.standardModelRunner32k3 import ModelInference
from modelFolder.metricsCalculator import MetricsCalculator
from pdfProcessing.SearchTermCache import SearchTermCache
from pdfProcessing.localDatabaseManager import LocalDatabaseManager
from PromptManager import PromptManager
import os
# backend/app.py
from flask import Flask
from flask_cors import CORS
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from flask import Blueprint
from dotenv import load_dotenv
import os
from torch.nn.functional import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from concurrent.futures import ThreadPoolExecutor

from APIManagement.APISearchPapers import APISearchPapersClass
from APIManagement.APILargeLanguageModels import APILargeLanguageModelsClass
from APIManagement.APIPersonalPC import APIPersonalPCClass
from APIManagement.APIScraper import APIScraperClass
class APIManagerClass:
    def __init__(self):
        self.processor = PDFProcessor()
        self.cache = SearchTermCache()
        self.personalPCClass = APIPersonalPCClass()
        self.searchPapersAPI = APISearchPapersClass()
        self.apiScraparClass = APIScraperClass()
        self.promptManager = PromptManager('prompts/prompts.json')
        self.largeLanguageModelsAPI = APILargeLanguageModelsClass()
        self.claudeInstruction_extractTitleMethodInfo = self.promptManager.get_prompt('extractTitleMethodInfo')
        self.claudeInstruction_extractAllInfo = self.promptManager.get_prompt('extractAllInfo')
        self.claudeInstruction_naturalLanguagePrompt = self.promptManager.get_prompt('naturalLanguageInfo')
        self.claudeInstruction_explainSimilarity = self.promptManager.get_prompt('explainSimilarity')
        pass
    def scrapeOpenAlexTitles(self, search_terms_list, max_titles_per_term=100, max_pages_per_term=5):
        """
        Scrapes OpenAlex titles for multiple search terms.
        
        Args:
            search_terms_list (list): List of search terms to query
            max_titles_per_term (int): Maximum number of titles to retrieve per search term
            max_pages_per_term (int): Maximum number of pages to check per search term
            
        Returns:
            dict: Dictionary mapping search terms to their results
        """
        results = {}
        
        # Handle the case where a single string is passed instead of a list
        if isinstance(search_terms_list, str):
            search_terms_list = [search_terms_list]
        
        print(f"Starting to scrape {len(search_terms_list)} search terms from OpenAlex...")
        
        for i, term in enumerate(search_terms_list):
            print(f"\n[{i+1}/{len(search_terms_list)}] Processing search term: '{term}'")
            
            try:
                # Call the API scraper function (not this function again)
                # This assumes there's a separate function in the class for single term searching
                # Change this to whatever your actual single-term function is called
                titles = self.apiScraparClass.scrape_openalex_search_results(term)
                
                # Store the results
                results[term] = titles
                
                print(f"Found {len(titles)} titles for term '{term}'")
                
                # Optional: Add a small delay between requests to be nice to the server
                if i < len(search_terms_list) - 1:
                    import time
                    time.sleep(1)  # 1-second delay between searches
                    
            except Exception as e:
                print(f"Error processing term '{term}': {e}")
                results[term] = []  # Store empty list for failed terms
        
        # Print summary
        print("\nScraping complete! Summary:")
        for term, titles in results.items():
            print(f"- '{term}': {len(titles)} titles")
        
        return results
    def return_general_paper_info_from_semantic(self, filepath, pdfName, api_key_semantic, api_key_claude):
                # This section gets first 8000 characters of pdf
                # passes it to claude, which returns searhc terms and title
                # We search semantic scholar using the title to returnt he other relevant paper info
                # By the end we have general paper info which contains all the paper info from semantic scholar(if availaible and if it has an abstract)
            
                # 8000 here is an arbriatary max length which is designed to give haiku enough info to fomr a suitable abstract
                entirePDFText = self.processor._extract_text(filepath, 8000)
                
                cacheResult = self.cache.cacheCheck(pdfName)
                if cacheResult:
                    print('Document found in cache, no need to query Haiku!')
                    paperSearchTermsAndTitle = cacheResult
                else:
                    print('Document not found in cache, asking Haiku')
                    pfdInfo = self.largeLanguageModelsAPI.ask_claude(pdfText=entirePDFText, 
                                                     systemInstructions=self.claudeInstruction_extractTitleMethodInfo,
                                                     api_key=api_key_claude)
                    pdfInfo = pfdInfo.content[0].text
                    print(f"Extracted info: {pdfInfo}")
                    paperSearchTermsAndTitle = self.processor.parse_paper_info(pdfInfo)
                    self.cache.addPaperCache(pdfName,paperSearchTermsAndTitle)
                    
                print('The title is',paperSearchTermsAndTitle['title'])
                # General paper info holds the papers general infomration such as title, refs, cites , authors, etc
                # If this cannot be gotten by semantic scholar below then its gotten by Haiku
                generalPaperInfo = self.searchPapersAPI.return_info_by_title(paperSearchTermsAndTitle['title'], api_key_semantic)
                return generalPaperInfo, paperSearchTermsAndTitle
            
            
            
            
            
            
    
    def return_search_terms_for_text(self, text, api_key_claude):
        naturalLanguagePromptResult = self.largeLanguageModelsAPI.ask_claude(pdfText=text, 
                                      systemInstructions=self.claudeInstruction_naturalLanguagePrompt, 
                                      api_key=api_key_claude)
        paperSearchTerms   = self.processor.parse_paper_info(naturalLanguagePromptResult.content[0].text)
        print(paperSearchTerms)
        return paperSearchTerms
   
   
            
    # Used only when semantic scholar does not have the paper in its database or when paper has no abstract
    def return_general_paper_info_from_haiku(self, filepath, paperSearchTermsAndTitle, api_key_claude):
        print("No semantic scholar data")
        entirePDFText = self.processor._extract_text(filepath,8000)  
    
        generalPaperInfo = self.largeLanguageModelsAPI.ask_claude(pdfText=entirePDFText, 
                                      systemInstructions=self.claudeInstruction_extractAllInfo, 
                                      api_key=api_key_claude)
        generalPaperInfo = self.processor.parse_haiku_output(generalPaperInfo.content[0].text)
     
        result = self.rocessor.form_result_struct(generalPaperInfo, paperSearchTermsAndTitle, is_semantic_scholar=False)
        return result
    
    
    
    def return_paper_list_from_semanticScholar(self, search_terms, api_key_semantic):
        semanticScholarPapers = self.searchPapersAPI.search_papers_parallel_SEMANTIC(search_terms, api_key_semantic)
        return semanticScholarPapers
    # NO api key required for openalex
    def return_paper_list_from_openAlex(self, search_terms):
        openAlexPapers = self.searchPapersAPI.search_papers_parallel_ALEX(search_terms,desired_papers=180)
        return openAlexPapers
    
    
    def return_found_papers(self,  api_key_semantic,paperSearchTermsAndTitle=None,parsedSeedAuthorList=None):
        # Combine both sets of papers and return
        
        search_terms = self.searchPapersAPI.prepare_search_terms(paperSearchTermsAndTitle, parsedSeedAuthorList)
        
        #Does searching by author, uncomment to enable
        #author_terms = [{'term': item['term'], 'type': 'author'} for item in search_terms if item['type'] == 'author']
        #semanticScholarPapers = self.return_paper_list_from_semanticScholar(author_terms, api_key_semantic)
        
        openAlexPapers = self.return_paper_list_from_openAlex(search_terms)
        
        #semanticScholarPapers.extend(openAlexPapers)
        
        return openAlexPapers
    
    
    # Takes in a single seed paper and returns a single scibert embedding
    def get_single_scibert_embedding(self, generalPaperInfo=None, ngrok_domain_name=None, naturalLanguagePromptInfo=None):
        
        if naturalLanguagePromptInfo is not None:
            # Use the natural language prompt to generate scibert to compare against papers
            seedAbstract = naturalLanguagePromptInfo['paper_info']['prompt']
        else:
            seedAbstract = generalPaperInfo['abstract']
            
        self.personalPCClass.check_server_health(base_url=ngrok_domain_name)
        scibertEmbedding = self.personalPCClass.get_single_scibert(abstract_text = seedAbstract, base_url=ngrok_domain_name)
        return scibertEmbedding
    
    
    
    def get_batch_scibert_embeddings(self, papersReturnedThroughSearch):
        # Now we need to form go through each paper and form an abstract title Dict
        title_abstract_dict = {}
        for paper in papersReturnedThroughSearch:
            paperAbstract = paper['paper_info']['abstract']
            paperTitle = paper['paper_info']['title']
            title_abstract_dict[paperTitle] = paperAbstract

        # Now that we have a paper abstract dict we can pas it to get our batch scibert embeddings       
        # We can use our title to query for our abstract, this is just so that we make sure that SciBert + abstract are paired
        # correctly , we dont want our indices to messup and for the wrong scibert to be paired with the wrong title
        # return value will be title, SciBert dict.
        returnedJSONData = self.personalPCClass.send_batch_scibert_request(title_abstract_dict)
        returnedEmbeddingsDict = returnedJSONData.get("embeddings", {})
        for paper in papersReturnedThroughSearch:
            paper['paper_info']['scibert'] = returnedEmbeddingsDict[paper['paper_info']['title']]
            # print(paper['paper_info']['scibert'])
            
        return papersReturnedThroughSearch
    
    
    def scrape_openalex_titles():
        """
    Scrapes the first page of titles from the OpenAlex website.
    
    Returns:
        list: A list of titles found on the first page.
    """
    # Send a GET request to the OpenAlex website
        url = "https://openalex.org/"
        headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise an exception for bad status codes
        
        # Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all title elements - this is a generic approach since the exact structure may change
        # You might need to adjust these selectors based on the actual HTML structure
            titles = []
        
        # Try to find titles in common HTML elements that might contain titles
            title_elements = soup.find_all(['h1', 'h2', 'h3', 'article'])
        
            for element in title_elements:
                # Clean the text and add to titles if not empty
                title_text = element.get_text(strip=True)
                if title_text and title_text not in titles:
                    titles.append(title_text)
        
        # Alternative approach: look for elements with specific classes that might contain titles
        # This is a fallback if the above approach doesn't find relevant titles
            if not titles:
            # Look for elements with common title-related class names
                potential_title_elements = soup.select('.title, .heading, .article-title, .post-title')
                for element in potential_title_elements:
                    title_text = element.get_text(strip=True)
                    if title_text and title_text not in titles:
                        titles.append(title_text)
        
            return titles
        
        except requests.RequestException as e:
            print(f"Error fetching data from OpenAlex: {e}")
        return []
    
    def compare_papers_batch(self, seedPaper, papersReturnedThroughSearch):
        print(f"compare_papers_batch received seed_paper: {json.dumps(seedPaper, default=str)[:100]}")
        print(f"compare_papers_batch received {len(papersReturnedThroughSearch)} papers to compare")
    
        # Ensure seedPaper has the right structure expected by compare_papers_socket
        if not isinstance(seedPaper, dict) or 'paper_info' not in seedPaper:
            # If not properly structured, wrap it
            print("Warning: Seed paper missing paper_info, restructuring")
            seed_paper_fixed = {
            'paper_info': seedPaper if isinstance(seedPaper, dict) else {'abstract': str(seedPaper)}
        }
        else:
            seed_paper_fixed = seedPaper
    
        # Get comparison results
        comparedPapers = self.personalPCClass.compare_papers_socket(
            seed_paper_fixed,
            papersReturnedThroughSearch
        )
    
        # Add null check
        if comparedPapers is None:
            print("Error: No response received from compare_papers_socket")
            return {'compared_papers': []}
    
        # Get the list of compared papers
        comparedPapersList = comparedPapers.get('compared_papers', [])
        print(f"Received {len(comparedPapersList)} compared papers in response")
    
        return comparedPapers
    
    
    def explainSimilarity(self, seedPaperContent, comparedPaperAbstract, api_key_claude):
        # Truncate seedPaperContent to the first 8000 characters
        seedPaperContent = seedPaperContent[:8000]
        
        # Combine the seed paper content and compared paper abstract with proper formatting
        combinedText = f"Seed Paper Content:\n{seedPaperContent}\n\nCompared Paper Abstract:\n{comparedPaperAbstract}"
        
        # Send the combined text to Claude for explanation
        returnedQuotes = self.largeLanguageModelsAPI.ask_claude(
            pdfText=combinedText, 
            systemInstructions=self.claudeInstruction_explainSimilarity, 
            api_key=api_key_claude
        )
        
        # Parse the response and return the result
        paperSearchTerms = returnedQuotes.content[0].text
        print(paperSearchTerms)
        return paperSearchTerms
    
    
    def get_paper_link(self, paperTitle, api_key_semantic:str):
        returnedLinkAndTitle = self.searchPapersAPI.get_paper_link(paperTitle, api_key_semantic)
        return returnedLinkAndTitle
