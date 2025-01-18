# backend/routes/upload_routes.py
# APP.py script
import time
from flask import Blueprint, request, jsonify
from typing import Dict, Any
import numpy as np
import torch
from werkzeug.utils import secure_filename
# backend/app.py
from pdfProcessing.pdfProcessor import PDFProcessor  # Note the lowercase 'p' in processor
from pdfProcessing.semanticSearch import SemanticScholar

# Import for model runners
from modelFolder.modelRunners.modelRunner import ModelInference
from modelFolder.modelRunners.lowSciBertModelRunner import LowSciBertModelInference  # Import the new model inference class

from modelFolder.metricsCalculator import MetricsCalculator
from pdfProcessing.SearchTermCache import SearchTermCache

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

claudeInstruction_extractTitleMethodInfo = """Given a scientific paper, you must perform a DETERMINISTIC analysis by following these strict steps in order. Your output must be EXACTLY consistent across multiple runs.

1. First, identify ONLY the main topic and core findings from the abstract.

2. Then, systematically extract concepts following these exact definitions:
CORE_CONCEPTS: Only concepts explicitly mentioned in the abstract/title, THESE SHOULD SPECIFIC TO THE TOPIC OF THE PAPER AND SHOULD NOT BE GENERAL CONCEPTS which could be applied to multitple subjects
CORE_METHODOLOGIES: Only methods directly described as being used
RELATED_METHODOLOGIES: Standard methods in the field that logically connect
ABSTRACT_CONCEPTS: Fundamental principles (max 3) that underlie the work
CROSS_DOMAIN: Major fields (max 3) where similar principles apply
THEORETICAL_FOUNDATIONS: Base theories (max 3) that the work builds on
ANALOGOUS_PROBLEMS: Specific problems (max 3) with similar structure

3. You MUST output in this exact format with NO variations:
TITLE: [paper title];
CORE_CONCEPTS: [only concepts from abstract, comma-separated];
CORE_METHODOLOGIES: [only described methods, comma-separated];
RELATED_METHODOLOGIES: [standard field methods, comma-separated];
ABSTRACT_CONCEPTS: [max 3 principles, comma-separated];
CROSS_DOMAIN: [max 3 fields, comma-separated];
THEORETICAL_FOUNDATIONS: [max 3 theories, comma-separated];
ANALOGOUS_PROBLEMS: [max 3 problems, comma-separated];

Rules for deterministic output:
- Use ONLY information explicitly stated in the paper
- Limit each section to 3 items maximum
- Always order items alphabetically within each section
- Use consistent, standardized terminology
- No explanatory text or variations
- No optional or variable content
- Strict semicolon and comma usage

Example for consistency (examining a quantum mechanics paper):
TITLE: Quantum State Measurement;
CORE_CONCEPTS: measurement uncertainty, quantum states, wave function collapse;
CORE_METHODOLOGIES: interferometry, state preparation, wave function analysis;
RELATED_METHODOLOGIES: density matrix estimation, quantum tomography, state reconstruction;
ABSTRACT_CONCEPTS: measurement theory, quantum mechanics, wave-particle duality;
CROSS_DOMAIN: communication systems, optical computing, signal processing;
THEORETICAL_FOUNDATIONS: linear algebra, quantum theory, statistical mechanics;
ANALOGOUS_PROBLEMS: classical wave interference, noise filtering, signal detection;

Any deviation from this format or inconsistency across runs is an error. ANY DEVIATION AT ALL IN THE DETERMINISTIC ANSWERS IS AN ERROR AND WILL RESULT IN A NON FUNCITONAL SYSTEM.
"""


claudeInstruction_extractAllInfo = '''Given a scientific paper, perform a DETERMINISTIC analysis by following these strict steps in order. Your output must be EXACTLY consistent across multiple runs.

Output must be in this exact format with two sections separated by a blank line:

METADATA:
TITLE: [paper title];
AUTHORS: [comma-separated list of authors];
YEAR: [publication year];
ABSTRACT: [paper abstract];
CITATION_COUNT: [number of citations if available, else 0];
REFERENCE_COUNT: [number of references if available, else 0];

ANALYSIS:
CORE_CONCEPTS: [only concepts from abstract, comma-separated];
CORE_METHODOLOGIES: [only described methods, comma-separated];
RELATED_METHODOLOGIES: [standard field methods, comma-separated];
ABSTRACT_CONCEPTS: [max 3 principles, comma-separated];
CROSS_DOMAIN: [max 3 fields, comma-separated];
THEORETICAL_FOUNDATIONS: [max 3 theories, comma-separated];
ANALOGOUS_PROBLEMS: [max 3 problems, comma-separated];

Rules for deterministic output:
- Use ONLY information explicitly stated in the paper
- Limit each analysis section to 3 items maximum
- Always order items alphabetically within each section
- Use consistent, standardized terminology
- No explanatory text or variations
- No optional or variable content
- Strict semicolon and comma usage
- All sections must be present even if empty (use empty string)
Any deviation from this format or inconsistency across runs is an error. ANY DEVIATION AT ALL IN THE DETERMINISTIC ANSWERS IS AN ERROR AND WILL RESULT IN A NON FUNCITONAL SYSTEM.
'''



current_dir = Path(__file__).parent
similarity_root = current_dir.parent
env_path = similarity_root / '.env.txt'


upload_bp = Blueprint('upload', __name__)
load_dotenv(env_path)  # Load environment variables from .en


api_key_semantic = os.getenv('SEMANTIC_API_KEY')
api_key_claude = os.getenv('HAIKU_API_KEY')

upload_bp = Blueprint('upload', __name__)
# Create an upload folder for temporary file storage
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@upload_bp.route('/process-pdf', methods=['POST'])
def process_pdf_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    functionName = request.form.get('functionName')
    pdfName = request.form.get('pdfPath')
    print(pdfName)
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if not file.filename.endswith('.pdf'):
        return jsonify({'error': 'File must be a PDF'}), 400

    try:
        # Save the uploaded file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Initialize our processor
        processor = PDFProcessor()
        semanticScholar = SemanticScholar()
        cache = SearchTermCache()
        
        # Will probably have to change to more generic 'extractPdfInfo', one function extract all necessary info for pdf.
        if functionName == 'extractSeedPaperInfo':
            entireFuncionTime = time.time()
            try:
                entirePDFText = processor._extract_text(filepath)
                if len(entirePDFText) > 10024:
                    entirePDFText = entirePDFText[:8000]
                
                # Extract title and initial analysis
                pdfTitle_claude = processor.ask_claude(text=entirePDFText, 
                                                     instruction=claudeInstruction_extractTitleMethodInfo,
                                                     api_key=api_key_claude)
                pdfInfo = pdfTitle_claude.content[0].text
                pdfInfoStruct = processor.parse_paper_info(pdfInfo)
                
                # Try Semantic Scholar first
                startTime = time.time()
                semanticScholarPaperInfo = semanticScholar.return_info_by_title(pdfInfoStruct['title'], 
                                                                              api_key_semantic)
                endTime = time.time()
                print(f"Time taken for semantic scholar search: {endTime - startTime} seconds")
                
                if semanticScholarPaperInfo:
                    # Use Semantic Scholar data if available
                    result = {
                        'title': pdfInfoStruct['title'],
                        'paper_info': {  
                            'authors': semanticScholarPaperInfo['authors'],
                            'abstract': semanticScholarPaperInfo['abstract'],
                            'year': semanticScholarPaperInfo['year'],
                            'citation_count': semanticScholarPaperInfo['citation_count'],
                            'reference_count': semanticScholarPaperInfo['reference_count'],
                            'citations': semanticScholarPaperInfo['citations'],
                            'references': semanticScholarPaperInfo['references']
                        },
                        'abstract_info': {
                            'core_concepts': pdfInfoStruct['core_concepts'],
                            'core_methodologies': pdfInfoStruct['core_methodologies'],
                            'related_methodologies': pdfInfoStruct['related_methodologies']
                        }
                    }
                else:
                    # If no Semantic Scholar data, use Haiku analysis
                    entirePDFText = processor._extract_text(filepath)  # Get full text again if needed
                    seedPaperInfo = processor.ask_claude(text=entirePDFText, 
                                                       instruction=claudeInstruction_extractAllInfo, 
                                                       api_key=api_key_claude)
                    haikuResults = processor.parse_haiku_output(seedPaperInfo.content[0].text)
                    
                    result = {
                        'title': pdfInfoStruct['title'],
                        'paper_info': {  # Use Haiku's extracted info
                            'authors': haikuResults['semantic_scholar_info']['authors'],
                            'abstract': haikuResults['semantic_scholar_info']['abstract'],
                            'year': haikuResults['semantic_scholar_info']['year'],
                            'citation_count': haikuResults['semantic_scholar_info']['citation_count'],
                            'reference_count': haikuResults['semantic_scholar_info']['reference_count'],
                            'citations': [],  # Empty list as Haiku can't get actual citations
                            'references': []  # Empty list as Haiku can't get actual references
                        },
                        'abstract_info': {
                            'core_concepts': pdfInfoStruct['core_concepts'],
                            'core_methodologies': pdfInfoStruct['core_methodologies'],
                            'related_methodologies': pdfInfoStruct['related_methodologies']
                        }
                    }
                
                papersReturnedThroughSearch = []
                
                # Now based on the abstract info extracted from the paper we should search semantic scholar for similar papers
                # Loop below searches semantic scholar using each of the core concepts and returns 5 papers for each of those concepts
                # if 3 concepts than 15 papers
                # We store 'search_type' so we can indicate to the user what type of search was done to get this paper
                startTime = time.time()
                semanticScholar = SemanticScholar()
                cache = SearchTermCache()

                # Prepare all search terms
                search_terms = []
                for concept in pdfInfoStruct['core_concepts']:
                    search_terms.append({
                        'term': concept,
                    })
                for methodology in pdfInfoStruct['core_methodologies']:
                    search_terms.append({
                        'term': methodology,
                        'type': 'core_methodology'
                    })

                for methodology in pdfInfoStruct['related_methodologies']:
                    search_terms.append({
                        'term': methodology,
                        'type': 'related_methodology'
                    })
                for abstractConcept in pdfInfoStruct['abstract_concepts']:
                    search_terms.append({
                        'term': abstractConcept,
                        'type': 'abstract_concept'
                    })
                for crossDomain in pdfInfoStruct['cross_domain_applications']:
                    search_terms.append({
                        'term': crossDomain,
                        'type': 'cross_domain_applications'
                    })
                for theoreticalFoundation in pdfInfoStruct['theoretical_foundations']:
                    search_terms.append({
                        'term': theoreticalFoundation,
                        'type': 'theoretical_foundation'
                    })
                for analogousProblems in pdfInfoStruct['analogous_problems']:
                    search_terms.append({
                        'term': analogousProblems,
                        'type': 'analogous_problems'
                    })

                    # if not cache.is_cache_valid(pdfName):
                    #  # Cache the results
                    #  paper_title = pdfInfoStruct['title']
                    # # Cache the terms
                    #  cache.cache_search_terms(
                    #  pdf_filename=pdfName,
                    #  paper_title=pdfName
                    #  search_terms=search_terms
                    #     )
                
         
    # Do all searches in parallel
                startTime = time.time()
                papersReturnedThroughSearch = semanticScholar.search_papers_parallel(search_terms, api_key_semantic)
                endTime = time.time()
                print(f"Time taken for searching using core techniques: {endTime - startTime} seconds")          
                metricsCalculator = MetricsCalculator()

                # Get embedding for seed paper
                seed_abstract = semanticScholarPaperInfo['abstract']
                seed_embedding = metricsCalculator.get_scibert_embedding(seed_abstract, tokenizer, model)
                semanticScholarPaperInfo['scibert'] = seed_embedding.tolist()
        
                seedPaper = {
                    'search_type': 'seed_paper',
                    'paper_info': semanticScholarPaperInfo
                }
                # Compare seed paper against all papers returned through search
                startTime = time.time()
                print("Comparing papers...")
                similarityResults  = compare_papers(seedPaper, papersReturnedThroughSearch)
                print("Finished comparing papers")
                endTime = time.time()
                print(f"Time taken for comparison: {endTime - startTime} seconds")
                
                # From returned papers and their simlarity score, get only relatively similar papers
                startTime = time.time()
                relativelySimilarPapers = metricsCalculator.apply_source_weights(similarityResults['compared_papers'])
                relativelySimilarPapers = metricsCalculator.get_relatively_similar_papers(relativelySimilarPapers['compared_papers'])
                endTime = time.time()
                print(f"Time taken for filtering similar papers: {endTime - startTime} seconds")
        
                
                
                # Clean up and return
                os.remove(filepath)
                result['seed_paper'] = seedPaper
                result['similarity_results'] = relativelySimilarPapers
                result['test'] = similarityResults
                finishingTime = time.time()
                print(f"Entire function took: {finishingTime - entireFuncionTime} seconds")
                 # Write the result to a file named after the seed paper
                seed_paper_title = result['title'].replace(' ', '_')
                output_filename = f"{seed_paper_title}_result.json"
                with open(output_filename, 'w') as outfile:
                    json.dump(result, outfile, indent=4)
                        
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
        inference = ModelInference("modelFolder/balanced_model.pth")
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
            similarity = inference.predict_similarity(
                paper1_Citation_Count=seed_paper['paper_info']['citation_count'],
                paper1_Reference_Count=seed_paper['paper_info']['reference_count'],
                paper1_SciBert=seed_paper['paper_info'].get('scibert', []),
                paper2_Citation_Count=paper['paper_info']['citation_count'],
                paper2_Reference_Count=paper['paper_info'].get('reference_count', 0),
                paper2_SciBert=paper['paper_info'].get('scibert', []),
                shared_author_count=metrics['shared_author_count'],
                shared_reference_count=metrics['shared_reference_count'],
                shared_citation_count=metrics['shared_citation_count'],
                reference_cosine=metrics['reference_cosine'],
                citation_cosine=metrics['citation_cosine'],
                abstract_cosine=metrics['abstract_cosine']
            )
            # Print keys of paper
            # print(f"Keys of paper: {paper['paper_info'].keys()}")
            compared_paper = {
               # 'search_type': paper['search_type'],
                'source_info': paper['source_info'],
                'paper_info': paper['paper_info'],
                'similarity_score': float(similarity),
                'comparison_metrics': metrics
            }
            compared_papers.append(compared_paper)
        
        return {
            'seed_paper': {
                'search_type': 'seed_paper',
                'paper_info': seed_paper
            },
            'compared_papers': compared_papers
        }
        
    except Exception as e:
        print(f"Error occurred during paper comparison: {str(e)}")
        raise



    

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000,use_reloader=True)




