# backend/routes/upload_routes.py
# APP.py script
import logging
import time
from flask import Blueprint, request, jsonify
from typing import Dict, Any
import numpy as np
import torch
from werkzeug.utils import secure_filename
# backend/app.py
from pdfProcessing.pdfProcessor import PDFProcessor  # Note the lowercase 'p' in processor
from pdfProcessing.APISearch import APISearchClass

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


class FetchResultsManagerClass:
    def __init__(self):
        self.api_key_claude = os.getenv('HAIKU_API_KEY')
        self.api_key_semantic = os.getenv('SEMANTIC_API_KEY')
        self.processor = PDFProcessor()
        self.cache = SearchTermCache()
        self.APISearch = APISearchClass()
        self.promptManager = PromptManager('prompts/prompts.json')
        
        self.claudeInstruction_extractTitleMethodInfo = self.promptManager.get_prompt('extractTitleMethodInfo')
        self.claudeInstruction_extractAllInfo = self.promptManager.get_prompt('extractAllInfo')
        pass
    
    
    
    def return_general_paper_info_from_semantic(self,filepath,pdfName):
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
                    pfdInfo = self.processor.ask_claude(pdfText=entirePDFText, 
                                                     systemInstructions=self.claudeInstruction_extractTitleMethodInfo,
                                                     api_key=self.api_key_claude)
                    pdfInfo = pfdInfo.content[0].text
                    print(f"Extracted info: {pdfInfo}")
                    paperSearchTermsAndTitle = self.processor.parse_paper_info(pdfInfo)
                    self.cache.addPaperCache(pdfName,paperSearchTermsAndTitle)
                
                startTime = time.time()
                print('The title is',paperSearchTermsAndTitle['title'])
                # General paper info holds the papers general infomration such as title, refs, cites , authors, etc
                # If this cannot be gotten by semantic scholar below then its gotten by Haiku
                generalPaperInfo = self.APISearch.return_info_by_title(paperSearchTermsAndTitle['title'], self.api_key_semantic)
                return generalPaperInfo, paperSearchTermsAndTitle
            
            
            
    def return_general_paper_info_from_haiku(self, filepath, paperSearchTermsAndTitle):
        print("No semantic scholar data")
        entirePDFText = self.processor._extract_text(filepath)  
    
        generalPaperInfo = self.processor.ask_claude(pdfText=entirePDFText, 
                                      systemInstructions=self.claudeInstruction_extractAllInfo, 
                                      api_key=self.api_key_claude)
        generalPaperInfo = self.processor.parse_haiku_output(generalPaperInfo.content[0].text)
     
        result = self.rocessor.form_result_struct(generalPaperInfo, paperSearchTermsAndTitle, is_semantic_scholar=False)
        return result
        