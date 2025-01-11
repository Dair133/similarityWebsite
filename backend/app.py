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
from modelFolder.modelRunner import ModelInference
from modelFolder.metricsCalculator import MetricsCalculator
import os
# backend/app.py
from flask import Flask
from flask_cors import CORS
import os
import sys
import json
from pathlib import Path
from flask import Blueprint
from dotenv import load_dotenv
import os
from torch.nn.functional import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from concurrent.futures import ThreadPoolExecutor

claudeInstruction_extractInfo = """
Analyze this scientific paper and extract information in EXACTLY the following format, with NO deviation:

TITLE:'[Insert the main research title - not journal name, section headers, or running headers]';CORE_CONCEPTS:[concept1],[concept2],[concept3];CORE_METHODOLOGIES:[method1],[method2],[method3];RELATED_METHODOLOGIES:[method1],[method2],[method3];

Important Formatting Rules:
ONLY RETURN A MAXIMUM OF 2 ENTRIES FOR EACH OF CORE CONCEPTS, CORE METHODOLOGIES, AND RELATED METHODOLOGIES
Everything must be on one line
Use semicolons (;) to separate major sections
Use commas (,) to separate items within sections
Do not use bullet points
Do not use line breaks
Include square brackets around each individual item
No spaces between commas and the next item
Use capital letters at the beginning of each word, i.e., Medicine,Biology
This information will be used for automated systems and must remain presentable
Key Objectives:
Accuracy To The Paper:
Extract only information that is explicitly mentioned or strongly implied in the paper.
Ensure terms align with the paper's content without adding unrelated material.
Searchability For Similar Papers:
Select terms and phrases that are likely to retrieve similar or related papers when used as search queries.
DO NOT INCLUDE GENERAL TERMS THAT WOULD RETURN PAPERS UNRELATED TO THE RESEARCH. SUCH AS COVID -19, PANDEMIC, ETC.
Be concise and consistent with terminology used in the field.
Core Concepts:
Identify the central ideas or themes of the research.
Ensure selected concepts describe both the topic and its broader context.
Core Methodologies:
Highlight the primary research methods explicitly mentioned in the paper.
Focus on methods fundamental to the research outcomes.
Related Methodologies:
Include auxiliary or secondary methods related to the core methodologies, even if not directly central to the research.

For the TITLE:
example: TITLE:'A Novel Approach to Treating Cancer';Core Concepts:[concept1],[concept2];Core Methodologies:[method1],[method2];Related Methodologies:[method1],[method2];
Extract only the main research title.
Exclude journal names, section headers, or running headers.
If no title is found, return 'Title Not Found'.
For concepts and methodologies:

List in order of importance.
Be specific but concise.
Ensure terms are relevant to the paper’s content.
Select terms that, when searched, will return similar or related papers.

Remember: The exact format is critical for automated processing. Any deviation from this format will cause errors in the system.
"""



current_dir = Path(__file__).parent
similarity_root = current_dir.parent
env_path = similarity_root / '.env.txt'


upload_bp = Blueprint('upload', __name__)
load_dotenv(env_path)  # Load environment variables from .en


api_key_semantic= os.getenv('SEMANTIC_API_KEY')
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
        # Will probably have ti chaneg to more generic 'extractPdfInfo', one function extract all necessary info for pdf.
        if functionName == 'extractSeedPaperInfo':
            entireFuncionTime = time.time()
            try:
                entirePDFText = processor._extract_text(filepath)
                if len(entirePDFText) > 10024:
                    entirePDFText = entirePDFText[:8000]
                
                # Get Claude's response - this will now be a string
                pdfTitle_claude = processor.ask_claude(
                    text=entirePDFText,
                    instruction=claudeInstruction_extractInfo,
                    api_key=api_key_claude
                )
                # Extract just the title text from the textblock and clean it up
                pdfInfo = pdfTitle_claude.content[0].text
            
                pdfInfoStruct = processor.parse_paper_info(pdfInfo)
                
                # Now pass the clean title to semantic scholar
                startTime = time.time()
                semanticScholarPaperInfo = semanticScholar.return_info_by_title(pdfInfoStruct['title'], api_key_semantic)
                endTime = time.time()  
                print(f"Time taken for semantic scholar search: {endTime - startTime} seconds")
                # Now search semantic scholar for that paper. If no result returned then
                # we will form the info we need ourselves using haiku.
                
                
                result = {
                    # Info about seed paper eneterd
                'title': pdfInfoStruct['title'],
                'semantic_scholar_info': {
                'authors': semanticScholarPaperInfo['authors'], 
                'abstract': semanticScholarPaperInfo['abstract'],
                'year': semanticScholarPaperInfo['year'],
                'citation_count': semanticScholarPaperInfo['citation_count'],
                'reference_count': semanticScholarPaperInfo['reference_count'],
                'citations': semanticScholarPaperInfo['citations'],
                'references': semanticScholarPaperInfo['references']
                    },
                'abstract_info':{
                    'core_concepts': pdfInfoStruct['core_concepts'],
                    'core_methodologies': pdfInfoStruct['core_methodologies'],
                    'related_methodologies': pdfInfoStruct['related_methodologies']
                    }
                
                }
                papersReturnedThroughSearch = []
                
                # Now based on the abstract info extracted from the paper we should search semantic scholar for similar papers
                # Loop below searches semantic scholar using each of the core concepts and returns 5 papers for each of those concepts, if 3 concepts than 15 papers
                # We store 'search_type' so we can indicate to the user what type of search was done to get this paper and compare its similarity to the seed paper.
                startTime = time.time()
                # In your app.py route
                semanticScholar = SemanticScholar()

                # Prepare all search terms
                search_terms = []

                # Add core concepts
                for concept in pdfInfoStruct['core_concepts']:
                    search_terms.append({
                        'term': concept,
                        'type': 'core_concept'
                    })

                # Add methodologies
                for methodology in pdfInfoStruct['core_methodologies']:
                    search_terms.append({
                        'term': methodology,
                        'type': 'core_methodology'
                    })

                # Add related methodologies
                for methodology in pdfInfoStruct['related_methodologies']:
                    search_terms.append({
                        'term': methodology,
                        'type': 'related_methodology'
                    })

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
                relativelySimilarPapers = metricsCalculator.get_relatively_similar_papers(similarityResults['compared_papers'])
                endTime = time.time()
                print(f"Time taken for filtering similar papers: {endTime - startTime} seconds")
        
                
                
                # Clean up and return
                os.remove(filepath)
                result['similarity_results'] = relativelySimilarPapers
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
            
            compared_paper = {
                'search_type': paper['search_type'],
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
