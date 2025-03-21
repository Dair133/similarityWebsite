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
from APIManagement.APIManager import APIManagerClass

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
ngrok_domain_name = os.getenv('NGROK_DOMAIN')
upload_bp = Blueprint('upload', __name__)
# Create an upload folder for temporary file storage
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
processor = PDFProcessor()
metricsCalculator = MetricsCalculator()
localDatabaseManager = LocalDatabaseManager()
apiManagerClass = APIManagerClass()

@upload_bp.route('/process-pdf', methods=['POST'])
def process_pdf_route():


    try:
        # Save the file
        file, pdfName = processor.validate_pdf_upload(request)
        filepath = processor.save_uploaded_pdf(file, UPLOAD_FOLDER)


        entireFuncionTime = time.time()
        try:    
            # For using a returned paper as  a seed paper I could clikc 'use as seed apper' upon which I pass all the info about the  paper already gotten to the backend
            # Simply skipping the pdf saving bit, then setting generalPaperInfo to the seed paper info
            # The one issue here would be thatw e would NOT have access to the full pdf and hence the search terms would not be as accurate and the user could not be shown the pdf of the paper.
                startTime = time.time()
                generalPaperInfo,paperSearchTermsAndTitle = apiManagerClass.return_general_paper_info_from_semantic(filepath, pdfName, api_key_semantic, api_key_claude)
                endTime = time.time()
                print(f"Time taken for semantic scholar search of seed paper: {endTime - startTime} seconds")
                
                
                if generalPaperInfo:
                    print("Using Semantic Scholar data")
                    result = processor.form_result_struct(generalPaperInfo, paperSearchTermsAndTitle, is_semantic_scholar=True)
                else:
                    print('No Semantic Scholar data, getting ALL paper info from Haiku!')
                    result = apiManagerClass.return_general_paper_info_from_haiku(filepath,pdfName)
                  
            
                generalPaperInfo['scibert'] = apiManagerClass.get_single_scibert_embedding(generalPaperInfo, ngrok_domain_name)
                # generalPaperInfo['scibert'] = metricsCalculator.return_scibert_embeddings(generalPaperInfo, tokenizer, model)
                
                # Returns the references citaitons and authors in a list, making themr eady to work with later on
                parsedSeedReferenceList, parsedSeedCitationList,  parsedSeedAuthorList = metricsCalculator.return_attributes_lists(generalPaperInfo)
                
            
                papersReturnedThroughSearch = apiManagerClass.return_found_papers(api_key_semantic=api_key_semantic, paperSearchTermsAndTitle=paperSearchTermsAndTitle,parsedSeedAuthorList=parsedSeedAuthorList)

        
                seedPaper = {
                    'search_type': 'seed_paper',
                    'paper_info': generalPaperInfo
                }
                print('loading poison pill')
                papersReturnedThroughSearch = localDatabaseManager.load_poison_pill_papers(papersReturnedThroughSearch,"poison_pill_papers_With_SciBert.xlsx")
 
                
                # This funcition call get shared refs, cites and authors for all papers
                papersReturnedThroughSearch = apiManagerClass.get_batch_scibert_embeddings(papersReturnedThroughSearch)
                papersReturnedThroughSearch = metricsCalculator.calculate_shared_attributes(papersReturnedThroughSearch,parsedSeedReferenceList,parsedSeedCitationList, parsedSeedAuthorList,)   

                print("Comparing papers...")
                relativelySimilarPapers = processor.remove_duplicates(papersReturnedThroughSearch)
   
                
                similarityResults = apiManagerClass.compare_papers_batch(seedPaper, papersReturnedThroughSearch)
                #print("External similarity results are", externalSimilarityResults)
                endTime = time.time()
                print(f"Time taken for comparison: {endTime - startTime} seconds")

                
                relativelySimilarPapers = metricsCalculator.get_relatively_similar_papers(similarityResults['compared_papers'])
                recommendations = metricsCalculator.get_recommendations(seedPaper, relativelySimilarPapers)
        
                # Remove the 'scibert' attribute from relatively similar papers
                for paper in relativelySimilarPapers:
                    if 'scibert' in paper['paper_info']:
                        del paper['paper_info']['scibert']
                # Clean up and return
                os.remove(filepath)
                result['seed_paper'] = seedPaper
                result['similarity_results'] = relativelySimilarPapers
                result['recommendations'] = recommendations  # Add recommendations to result
                result['test'] = similarityResults
                finishingTime = time.time()
                print(f"Entire function took: {finishingTime - entireFuncionTime} seconds")
                    
                
                    
                return jsonify(result), 200
                   
        except Exception as e:
                import traceback
                error_message = f"Error processing request: {str(e)}"
                detailed_error = traceback.format_exc()
                print(f"Detailed error: {detailed_error}")  # Enhanced error logging
                
                # Clean up if file exists
                if os.path.exists(filepath):
                    os.remove(filepath)
                    
                return jsonify({
                    'error': error_message,
                    'status': 'error',
                    'traceback': detailed_error
                }), 500

    except Exception as e:
        # Log the full error with traceback

        error_msg = f"Error processing PDF: {str(e)}\nTraceback:\n{traceback.format_exc()}"
        print(error_msg)  
        
        # Clean up file if it exists
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
            
        return jsonify({
            'error': str(e),
            'details': traceback.format_exc()
        }), 500

@upload_bp.route('/explain-similarity', methods=['POST'])
def explain_similarity():

    try:
        print("Inside explain similarity route")
        data = request.get_json()
        
        # Log the received data for debugging
        print("Received data:", data)
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Extract the required information
        seed_paper = data.get('seed_paper', {})
        current_paper = data.get('current_paper', {})
        similarity_metrics = data.get('similarity_metrics', {})
        
        seedPaperAbstract = seed_paper.get('abstract', '')
        print("Seed paper abstract:", seedPaperAbstract)
        currentPaperAbstract = current_paper.get('abstract', '')
        explanation = apiManagerClass.explainSimilarity(seedPaperAbstract, currentPaperAbstract, api_key_claude)
    
        return jsonify({
            "explanation": explanation,
            "success": True
        })
        
    except Exception as e:
        print(f"Error in explain_similarity: {str(e)}")
        return jsonify({"error": str(e)}), 500

@upload_bp.route('/get-paper-link', methods=['POST'])
def get_paper_link():
    # Expect a JSON payload with a key 'paper_title'
    data = request.get_json()
    paper_title = data.get('paper_title', "")
    print("Paper title is", paper_title)
    # Call your helper function to search by paper title
    paperLinkAndTitle = apiManagerClass.get_paper_link(paper_title, api_key_semantic)
    return jsonify(paperLinkAndTitle)




@upload_bp.route('/natural-language-search', methods=['POST'])
def semantic_search():
    
    naturalLanguagePrompt = request.json['query']
    print(naturalLanguagePrompt)
    searchTerms = apiManagerClass.return_search_terms_for_text(naturalLanguagePrompt, api_key_claude)

    # Fix the structure - ensure unique keys
    naturalLanguagePromptInfo = {
        'search_type': 'natural_language_prompt',
        'paper_info': {
            'title': 'Natural Language Prompt',
            'abstract': naturalLanguagePrompt,
            'prompt': naturalLanguagePrompt,
            'scibert': None,
            'authors': 'None',
            'year': 'None',
            'citation_count': 0,
            'reference_count': 0,
            'citations': 'None',
            'references': 'None',
        },
        'comparison_metrics': {
            'shared_reference_count': 0,
            'shared_citation_count': 0,
            'shared_author_count': 0
        }
    }      
    
    naturalLanguagePromptInfo['paper_info']['scibert'] = apiManagerClass.get_single_scibert_embedding(
        naturalLanguagePromptInfo=naturalLanguagePromptInfo, ngrok_domain_name=ngrok_domain_name)
    
    papersReturnedThroughSearch = apiManagerClass.return_found_papers(
        api_key_semantic=api_key_semantic, paperSearchTermsAndTitle=searchTerms)
    
    papersReturnedThroughSearch = localDatabaseManager.load_poison_pill_papers(
        papersReturnedThroughSearch, "poison_pill_papers_With_SciBert.xlsx")
    
    papersReturnedThroughSearch = apiManagerClass.get_batch_scibert_embeddings(papersReturnedThroughSearch)
    
    # Create a clean seed paper structure that matches what the endpoint expects
    clean_seed_paper = {
        'paper_info': naturalLanguagePromptInfo['paper_info'],
        'comparison_metrics': naturalLanguagePromptInfo['comparison_metrics']
    }
    
    # Print structure for debugging
    print("Sending clean seed paper structure:", json.dumps(clean_seed_paper, default=str)[:100])
    
    similarityResults = apiManagerClass.compare_papers_batch(clean_seed_paper, papersReturnedThroughSearch)
    
    if similarityResults and 'compared_papers' in similarityResults:
        # Process results if successful
        relativelySimilarPapers = metricsCalculator.get_relatively_similar_papers(
            similarityResults['compared_papers'])
        
        return jsonify({
            'seed_paper': clean_seed_paper,
            'similarity_results': relativelySimilarPapers
        })
    else:
        print('In natural language search - error case')
        return jsonify({
            'status': 'error',
            'message': 'Failed to compare papers'
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
