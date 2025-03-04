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
# Think of search paper string as seed
claudeInstruction_extractTitleMethodInfo = """
# Scientific Paper Analysis System

You are a professional scientific literature analyst specializing in precise terminology extraction. Your extracted terms will be used directly in an automated academic search pipeline that feeds into a machine learning model. This pipeline searches Semantic Scholar using your extracted terms to find papers similar to a "seed paper," and these results are then evaluated by an ML model that assesses their similarity to the original paper.

## Core Extraction Task
Extract the exact paper title, identify 3 core methodologies, and create 3 conceptual angles that will yield effective literature search results. The quality of your extracted terms directly impacts the effectiveness of the entire ML pipeline.

## Output Format (CRITICAL - MUST BE FOLLOWED EXACTLY)
TITLE: [exact paper title];
CORE_METHODOLOGIES: [method1, method2, method3];
CONCEPTUAL_ANGLES: [angle1, angle2, angle3];

## CORE_METHODOLOGIES Guidelines
- Create exact search strings that would return the MOST SIMILAR papers to the source when searched on Semantic Scholar
- Each methodology must combine a fundamental technique with precise scope-defining terms
- The string should be specific enough that searching with it would NOT return papers from unrelated domains
- For each term, ask: "Would this search term potentially return completely unrelated papers?" If yes, make it more specific
- Avoid overly general terms like "neural networks" or "optimization" without proper qualification
- These terms must identify papers that use highly similar approaches to the seed paper

## CONCEPTUAL_ANGLES Guidelines
- Create search strings that would return papers with a SLIGHT CONCEPTUAL SLANT from the source
- These should capture related approaches that explore the same problem from different angles
- Each angle should maintain relevance to the core topic while introducing a novel perspective
- Strike a careful balance between similarity and novelty - too similar will duplicate core methodologies, too different will return irrelevant papers
- The ML model can distinguish broadly between similar and dissimilar papers, so these terms should find papers that are related but offer new insights

## Examples of Effective Search Strings:
CORE: "Siamese networks for document similarity" (returns highly similar papers)
ANGLE: "Cross-modal embeddings for text retrieval" (returns papers with a related but different approach)

## FORMAT WARNING (EXTREMELY IMPORTANT)
The output format MUST be followed EXACTLY as specified. Any deviation will cause the entire ML pipeline to fail. This includes:
- Using the exact section headers shown
- Including the square brackets around each term
- Using semicolons as separators exactly as shown
- Not adding any extra characters, line breaks, or explanations
- Providing exactly 3 items for each category

This output is programmatically parsed and any formatting change will break the automated system.
MOST IMPORTANT POINT, IS THAT YOU PICK OUT TERMS WHICH WILL NOT LARGE SETS OF TOTALLY DIFFFERENT UNRELATED PAPERS
"""



claudeInstruction_extractAllInfo = '''
# Scientific Paper Analysis System

You are an expert scientific literature analyzer specializing in extracting core information from academic papers. Your analysis is critical for an AI research pipeline that uses your extracted data to find similar papers through SciBERT embeddings when Semantic Scholar lacks complete information on a document.

## Core Extraction Task
Extract the paper's core information and generate a technical abstract specifically optimized for SciBERT embedding-based similarity matching.

## Output Format (CRITICAL - MUST BE FOLLOWED EXACTLY)
The output must follow this precise format with two main sections:

SEMANTIC_SCHOLAR_INFO:
TITLE: [exact paper title];
AUTHORS: [comma-separated list of all authors, formatted as "Lastname, Firstname" or "Lastname, F."];
YEAR: [publication year];
ABSTRACT: [SciBERT-optimized technical abstract that precisely captures the paper's technical contributions, methodology, and findings];
CITATIONS: [semicolon-separated list of papers that cite this work];
REFERENCES: [semicolon-separated list of papers referenced by this work];


## Extraction Guidelines
For SEMANTIC_SCHOLAR_INFO section:
- Extract the EXACT paper title
- List ALL authors in order of appearance
- GENERATE a SciBERT-optimized technical abstract that:
  * Uses domain-specific scientific terminology consistent with the paper's field
  * Emphasizes technical concepts, methodologies, algorithms, and contributions
  * Includes precise technical terms that would appear in similar papers
  * Contains specific technical metrics, evaluation methods, datasets, and quantitative results
  * Maintains the paper's original technical vocabulary and naming conventions
  * References established techniques, frameworks, or algorithms by their standard names
  * Has sufficient technical density for accurate embedding representation
  * Uses consistent technical terminology throughout for better embedding coherence
  * Is approximately 200-300 words in length and technical in nature
- If references or citations are listed in the paper, include their full titles
- Format each citation and reference as a complete paper title



## SciBERT Embedding Optimization Guidelines
- SciBERT creates embeddings based on scientific terminology and structure
- Effective abstract generation should match the scientific terminology of the paper's domain
- Use standardized scientific language patterns that SciBERT would recognize from its training corpus
- Structure the abstract with clear technical sections covering problem statement, methodology, and results
- Include key technical phrases that would be shared across semantically similar papers
- Avoid general or vague descriptions in favor of precise technical characterizations
- Prioritize including technical content that distinguishes this paper from others in the field
- Focus more on technical approach rather than general implications or background
- Match the abstract's language style to published papers in the same field/venue

## Format Warning
Any deviation from the exact output format will cause the entire ML pipeline to fail. This output is programmatically parsed with strict expectations for section headers, brackets, delimiters, and structure.
'''

current_dir = Path(__file__).parent
similarity_root = current_dir.parent
env_path = similarity_root / '.env.txt'

upload_bp = Blueprint('upload', __name__)
print('env path is')
print(env_path)
load_dotenv(env_path)  # Load environment variables from .en
api_key_gemini = os.getenv('GEMINI_API_KEY')
api_key_semantic = os.getenv('SEMANTIC_API_KEY')
api_key_claude = os.getenv('HAIKU_API_KEY')
api_key_deepseek = os.getenv('DEEPSEEK_API_KEY')
print(api_key_claude)
upload_bp = Blueprint('upload', __name__)
# Create an upload folder for temporary file storage
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@upload_bp.route('/process-pdf', methods=['POST'])
def process_pdf_route():
    processor = PDFProcessor()
    APISearch = APISearchClass()
    cache = SearchTermCache()
    metricsCalculator = MetricsCalculator()
    localDatabaseManager = LocalDatabaseManager()

    try:
        # Save the file
        file, pdfName = processor.validate_pdf_upload(request)
        filepath = processor.save_uploaded_pdf(file, UPLOAD_FOLDER)


        # Will probably have to change to more generic 'extractPdfInfo', one function extract all necessary info for pdf.
        entireFuncionTime = time.time()
        try:    
                # 8000 here is an arbriatary max length which is designed to give haiku enough info to fomr a suitable abstract
                entirePDFText = processor._extract_text(filepath, 8000)
                
                cacheResult = cache.cacheCheck(pdfName)
                if cacheResult:
                    print('Document found in cache, no need to query Haiku!')
                    paperSearchTermsAndTitle = cacheResult
                else:
                    print('Document not found in cache, asking Haiku')
                    pfdInfo = processor.ask_claude(pdfText=entirePDFText, 
                                                     systemInstructions=claudeInstruction_extractTitleMethodInfo,
                                                     api_key=api_key_claude)
                    pdfInfo = pfdInfo.content[0].text
                    print(f"Extracted info: {pdfInfo}")
                    paperSearchTermsAndTitle = processor.parse_paper_info(pdfInfo)
                    cache.addPaperCache(pdfName,paperSearchTermsAndTitle)
                
                startTime = time.time()
                print('The title is',paperSearchTermsAndTitle['title'])
                # General paper info holds the papers general infomration such as title, refs, cites , authors, etc
                # If this cannot be gotten by semantic scholar below then its gotten by Haiku
                generalPaperInfo = APISearch.return_info_by_title(paperSearchTermsAndTitle['title'], api_key_semantic)
                endTime = time.time()
                print(f"Time taken for semantic scholar search: {endTime - startTime} seconds")
                
                
                if generalPaperInfo:
                 print("Using Semantic Scholar data")
                    # Use Semantic Scholar data if available
                 result = processor.form_result_struct(generalPaperInfo, paperSearchTermsAndTitle, is_semantic_scholar=True)
                else:
                  print("No semantic scholar data")
                # If no Semantic Scholar data, use Haiku analysis
                  entirePDFText = processor._extract_text(filepath)  # Get full text again if needed
    
                  generalPaperInfo = processor.ask_claude(pdfText=entirePDFText, 
                                      systemInstructions=claudeInstruction_extractAllInfo, 
                                      api_key=api_key_claude)
                  print(generalPaperInfo.content[0].text)
                  generalPaperInfo = processor.parse_haiku_output(generalPaperInfo.content[0].text)
     
                  result = processor.form_result_struct(generalPaperInfo, paperSearchTermsAndTitle, is_semantic_scholar=False)
                    
                seed_abstract = generalPaperInfo['abstract']
                seed_embedding = metricsCalculator.get_scibert_embedding(seed_abstract, tokenizer, model)
                generalPaperInfo['scibert'] = seed_embedding.tolist()
                
                # Here we will calculate shared references, citations and cosine
                seedReferenceList = generalPaperInfo['references']
                parsedSeedReferenceList = metricsCalculator.parse_attribute_list(seedReferenceList,';')
                
                seedCitationList = generalPaperInfo['citations']
                parsedSeedCitationList = metricsCalculator.parse_attribute_list(seedCitationList,';')
                
                seedAuthorList = generalPaperInfo['authors']
                if isinstance(seedAuthorList, list):
                    seedAuthorList = ', '.join(seedAuthorList)
                parsedSeedAuthorList = metricsCalculator.parse_attribute_list(seedAuthorList, ',')
                papersReturnedThroughSearch = []
                
                # Now based on the abstract info extracted from the paper we should search semantic scholar for similar papers
                # Loop below searches semantic scholar using each of the core concepts and returns 5 papers for each of those concepts
                # if 3 concepts than 15 papers
                # We store 'search_type' so we can indicate to the user what type of search was done to get this paper
                startTime = time.time()
                search_terms = []
                 # Add core methodologies with high weight since they're specific
                 
                 
                # Replace the original search terms preparation block with:
                search_terms = APISearch.prepare_search_terms(paperSearchTermsAndTitle, parsedSeedAuthorList)

                # Do all searches in parallel
                startTime = time.time()
                print('search terms are')
                print(search_terms)
                papersReturnedThroughSearch = APISearch.search_papers_parallel(search_terms, api_key_semantic)
                openAlexPapers = APISearch.search_papers_parallel_ALEX(search_terms,desired_papers=1)
                papersReturnedThroughSearch.extend(openAlexPapers)
                endTime = time.time()
                print(f"Time taken for searching using core techniques: {endTime - startTime} seconds")          

        
                seedPaper = {
                    'search_type': 'seed_paper',
                    'paper_info': generalPaperInfo
                }
                                
                # WORK HERE CLAUDE
                # Now we should append poison pill papers from the excel
                poisonPillPapers = localDatabaseManager.load_poison_pill_papers("poison_pill_papers_With_SciBert.xlsx")
                # Append poison pill papers to the search results
                if poisonPillPapers and len(poisonPillPapers) > 0:
                    print(f"Adding {len(poisonPillPapers)} poison pill papers to comparison set")
                    papersReturnedThroughSearch.extend(poisonPillPapers)
                else:
                    print("No poison pill papers found or loaded")
                
                startTime = time.time()
                print("Calculating shared attributes...")
                # This funcition call get shared refs, cites and authors for all papers
                papersReturnedThroughSearch = metricsCalculator.calculate_shared_attributes(
                    papersReturnedThroughSearch,
                    parsedSeedReferenceList,
                    parsedSeedCitationList, 
                    parsedSeedAuthorList,
                )   

                print("Comparing papers...")
                relativelySimilarPapers = processor.remove_duplicates(papersReturnedThroughSearch)
                similarityResults = compare_papers(seedPaper, papersReturnedThroughSearch)
                #papersReturnedThroughSearch.extend(poisonPillPapers)
                #papersReturnedThroughSearch.extend(poisonPillPapers)
                #papersReturnedThroughSearch.extend(poisonPillPapers)
                endTime = time.time()
                print(f"Time taken for comparison: {endTime - startTime} seconds")

                
                # From returned papers and their simlarity score, get only relatively similar papers
                startTime = time.time()
                relativelySimilarPapers = metricsCalculator.apply_source_weights(similarityResults['compared_papers'])
                relativelySimilarPapers = metricsCalculator.get_relatively_similar_papers(relativelySimilarPapers['compared_papers'])
                recommendations = metricsCalculator.get_recommendations(seedPaper, relativelySimilarPapers)
                endTime = time.time()
                print(f"Time taken for filtering similar papers: {endTime - startTime} seconds")
                # Remove the 'scibert' attribute from relatively similar papers
                for paper in relativelySimilarPapers:
                    if 'scibert' in paper['paper_info']:
                        del paper['paper_info']['scibert']
        
        
                # Remove the 'scibert' attribute from relatively similar papers
                for paper in relativelySimilarPapers:
                    if 'scibert' in paper['paper_info']:
                        del paper['paper_info']['scibert']
                # Clean up and return
                os.remove(filepath)
                result['seed_paper'] = seedPaper
                result['similarity_results'] = relativelySimilarPapers
                result['recommendations'] = recommendations  # Add recommendations to result
                result['openAlex'] = openAlexPapers
                result['test'] = similarityResults
                finishingTime = time.time()
                print(f"Entire function took: {finishingTime - entireFuncionTime} seconds")
                        
                return jsonify(result), 200
                   
        except Exception as e:
                error_message = f"Error processing request: {str(e)}"
                print(f"Detailed error: {error_message}")  # Enhanced error logging
                
                # Clean up if file exists
                if os.path.exists(filepath):
                    os.remove(filepath)
                    
                return jsonify({
                    'error': error_message,
                    'status': 'error'
                }), 500

    except Exception as e:
        # Log the full error with traceback
        import traceback
        error_msg = f"Error processing PDF: {str(e)}\nTraceback:\n{traceback.format_exc()}"
        print(error_msg)  
        
        # Clean up file if it exists
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
            
        return jsonify({
            'error': str(e),
            'details': traceback.format_exc()
        }), 500


# Add the project root directory to Python path for proper imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def create_app():
    app = Flask(__name__)
    CORS(app)
    
    global tokenizer, model
    try:
        # Try to load from local cache only
        print("Loading SciBERT model...")
        tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', local_files_only=True)
        model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', local_files_only=True)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.to('cuda')
            print("Model moved to GPU")
    except Exception as e:
        print(f"Error loading from cache, downloading model: {e}")
        # If not in cache, download it
        tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
    
    print("SciBERT model loaded successfully")
    app.register_blueprint(upload_bp)
    return app

# This function should probably be moved to another file????//
def compare_papers(seed_paper, papers_returned_through_search):
    try:
        inference = ModelInference(
            model_path="modelFolder/standardModel-3-32k-NoLeakage.pth",
        )
        metricCalculator = MetricsCalculator()
        
        # Get all metrics in parallel
        metrics_list = metricCalculator.process_papers_parallel(
            seed_paper, 
            papers_returned_through_search,
            tokenizer, 
            model
        )
        
        compared_papers = []
        # Process the results
        for paper, metrics in zip(papers_returned_through_search, metrics_list):
            # Use the pre-calculated shared attributes from the paper
            shared_data = {
                'shared_references': paper['comparison_metrics'].get('shared_reference_count', 0),
                'shared_citations': paper['comparison_metrics'].get('shared_citation_count', 0),
                'shared_authors': paper['comparison_metrics'].get('shared_author_count', 0)
            }
            try:
                similarity = inference.predict_similarity(
                    paper1_SciBert=seed_paper['paper_info'].get('scibert', []),
                    paper2_SciBert=paper['paper_info'].get('scibert', []),
                    shared_data=shared_data
                )
            except Exception as e:
                logging.error(f"Error in similarity prediction: {str(e)}")
                similarity = 0.0
            
            # Ensure the comparison_metrics in metrics preserves the shared counts
            if 'comparison_metrics' in paper:
                # Copy the pre-calculated shared counts into metrics
                metrics['shared_reference_count'] = paper['comparison_metrics'].get('shared_reference_count', 0)
                metrics['shared_citation_count'] = paper['comparison_metrics'].get('shared_citation_count', 0)
                metrics['shared_author_count'] = paper['comparison_metrics'].get('shared_author_count', 0)
            
            compared_paper = {
                'source_info': paper.get('source_info', {}),
                'paper_info': paper.get('paper_info', {}),
                'similarity_score': float(similarity),
                'comparison_metrics': metrics
            }
            compared_papers.append(compared_paper)

        return {
            'seed_paper': {
                'paper_info': seed_paper
            },
            'compared_papers': compared_papers,
        }
        
    except Exception as e:
        logging.error(f"Error occurred during paper comparison: {str(e)}")
        # Return fake results in case of failure
        fake_results = [
            {
                'source_info': {},
                'paper_info': {},
                'similarity_score': 0.5,
                'comparison_metrics': {}
            }
        ]
        return {
            'seed_paper': {
                'paper_info': seed_paper
            },
            'compared_papers': fake_results,
            'search_terms': 'Fake Search'
        }


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
    #The below isn't needed as Gunicorn handles this.
