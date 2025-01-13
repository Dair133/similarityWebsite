# Import for model runners
import unicodedata
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
# For some reason just using '..' does not want to import correctly therefore we need to do weird hack below
# Get the absolute path of the parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))  # Gets testing folder path
parent_dir = os.path.dirname(current_dir)  # Gets backend folder path
sys.path.append(parent_dir)  # Adds backend folder to Python path

from modelFolder.modelRunners.modelRunner import ModelInference
# Helper function to convert string to array (moved outside)
def convert_string_to_array(vec_str):
    if isinstance(vec_str, str):
        try:
            # Handle np.str_ format
            if vec_str.startswith('np.str_'):
                vec_str = vec_str[8:-1]  
            # Remove brackets and split by comma
            vec_list = vec_str.strip('[]').split(',')
            # Convert to float array
            return np.array([float(x.strip()) for x in vec_list])
        except:
            return np.zeros(768)  # Return zero vector if conversion fails
    return vec_str


def calculate_cosine_similarity(vec1, vec2):
    # Convert string representation of array to actual array
    def convert_string_to_array(vec):
        if isinstance(vec, str):
            try:
                # Handle np.str_ format specifically
                if vec.startswith('np.str_'):
                    vec = vec[8:-1]  # Remove np.str_(' ') wrapper
                # Remove brackets and split by comma
                vec_str = vec.strip('[]').split(',')
                # Convert to float array
                return np.array([float(x.strip()) for x in vec_str])
            except:
                return None
        return vec

    # Convert inputs if they're strings and check for None/empty
    vec1 = convert_string_to_array(vec1)
    vec2 = convert_string_to_array(vec2)
    
    # Return 0 if either vector is None, empty or not convertible
    if vec1 is None or vec2 is None or len(vec1) == 0 or len(vec2) == 0:
        return 0.0

    # Reshape if needed
    if not isinstance(vec1, np.ndarray):
        vec1 = np.array(vec1).reshape(1, -1)
    else:
        vec1 = vec1.reshape(1, -1)
    
    if not isinstance(vec2, np.ndarray):
        vec2 = np.array(vec2).reshape(1, -1)
    else:
        vec2 = vec2.reshape(1, -1)
    
    # Check if vectors are empty after conversion
    if vec1.size == 0 or vec2.size == 0:
        return 0.0
        
    try:
        return float(cosine_similarity(vec1, vec2)[0][0])
    except:
        return 0.0




def get_shared_count(list1, list2):
    # Handle empty, None, or non-string inputs
    if not list1 or not list2 or not isinstance(list1, str) or not isinstance(list2, str):
        return 0
    
    # Split strings and clean items
    try:
        set1 = set(x.strip() for x in list1.split(';'))
        set2 = set(x.strip() for x in list2.split(';'))
        # Remove empty strings
        set1 = {x for x in set1 if x}
        set2 = {x for x in set2 if x}
        return len(set1.intersection(set2))
    except:
        return 0

def calculate_text_cosine(text1, text2):
    """Calculate cosine similarity between two text strings using TF-IDF."""
    if not isinstance(text1, str) or not isinstance(text2, str):
        return 0
    
    # Clean and validate texts
    text1 = text1.strip().lower()
    text2 = text2.strip().lower()
    if not text1 or not text2:
        return 0
    
    # Use TF-IDF for cosine similarity
    try:
        vectorizer = TfidfVectorizer().fit([text1, text2])
        vectors = vectorizer.transform([text1, text2])
        return cosine_similarity(vectors[0], vectors[1])[0][0]
    except Exception as e:
        print(f"Error in text cosine calculation: {str(e)}")
        return 0

def compare_papers(seed_paper_df, returned_papers_df, model_path):
    try:
        # Initialize model
        inference = ModelInference(model_path)
        
        # Get seed paper
        seed_paper = seed_paper_df.iloc[0]
        
        print("\nComparing Papers:")
        print(f"Seed Paper Title: {seed_paper.get('Title', 'Unknown Title')}\n")
        
        # Debug print
        print("DEBUG: Seed paper Citations and References:")
        print(f"Citations: {seed_paper.get('Citations', '')}")
        print(f"References: {seed_paper.get('References', '')}")
        
        for idx, returned_paper in returned_papers_df.iterrows():
            print(f"\nProcessing paper {idx}")
            
            # Convert SciBert strings to numpy arrays for model input
            try:
                seed_scibert = convert_string_to_array(seed_paper.get('SciBert', '[]'))
                returned_scibert = convert_string_to_array(returned_paper.get('SciBert', '[]'))
                
                # Debug print SciBert shapes
                print(f"SciBert shapes - Seed: {seed_scibert.shape}, Returned: {returned_scibert.shape}")
            except Exception as e:
                print(f"Error converting SciBert embeddings: {str(e)}")
                continue

            # Calculate abstract similarity using SciBert
            abstract_cosine = calculate_cosine_similarity(
                seed_paper.get('SciBert', []), 
                returned_paper.get('SciBert', [])
            )
            print(f"Abstract cosine: {abstract_cosine}")
            
            # Debug print for Citations and References content
            print("\nDEBUG Citations:")
            print(f"Paper 1: {seed_paper.get('Citations', '')[:100]}...")
            print(f"Paper 2: {returned_paper.get('Citations', '')[:100]}...")
            
            # Calculate cosines using TF-IDF for Citations and References
            citation_cosine = calculate_text_cosine(
                str(seed_paper.get('Citations', '')), 
                str(returned_paper.get('Citations', ''))
            )
            print(f"Citation cosine: {citation_cosine}")
            
            reference_cosine = calculate_text_cosine(
                str(seed_paper.get('References', '')), 
                str(returned_paper.get('References', ''))
            )
            print(f"Reference cosine: {reference_cosine}")
            
            # Get shared counts
            shared_authors = get_shared_count(
                seed_paper.get('Authors', ''), 
                returned_paper.get('Authors', '')
            )
            
            shared_citations = get_shared_count(
                seed_paper.get('Citations', ''), 
                returned_paper.get('Citations', '')
            )
            
            shared_references = get_shared_count(
                seed_paper.get('References', ''), 
                returned_paper.get('References', '')
            )
            
            # Calculate citation and reference cosine based on overlap
            try:
                # Normalize by the total number of items
                total_citations = len(seed_paper.get('Citations', '').split(';')) + len(returned_paper.get('Citations', '').split(';'))
                citation_cosine = (2 * shared_citations) / total_citations if total_citations > 0 else 0
                
                total_references = len(seed_paper.get('References', '').split(';')) + len(returned_paper.get('References', '').split(';'))
                reference_cosine = (2 * shared_references) / total_references if total_references > 0 else 0
            except:
                citation_cosine = 0
                reference_cosine = 0
            
            # Convert counts to integers if they're strings
            citation_count1 = int(seed_paper.get('Citation_Count', 0))
            reference_count1 = int(seed_paper.get('Reference_Count', 0))
            citation_count2 = int(returned_paper.get('Citation_Count', 0))
            reference_count2 = int(returned_paper.get('Reference_Count', 0))
            
            # Get model prediction with properly typed inputs
            similarity_score = inference.predict_similarity(
                paper1_Citation_Count=citation_count1,
                paper1_Reference_Count=reference_count1,
                paper1_SciBert=seed_scibert,
                paper2_Citation_Count=citation_count2,
                paper2_Reference_Count=reference_count2,
                paper2_SciBert=returned_scibert,
                shared_author_count=shared_authors,
                shared_reference_count=shared_references,
                shared_citation_count=shared_citations,
                reference_cosine=reference_cosine,
                citation_cosine=citation_cosine,
                abstract_cosine=abstract_cosine
            )

            # Print results
            print(f"Compared with: {returned_paper.get('Title', 'Unknown Title')}")
            print(f"Similarity Score: {similarity_score:.4f}")
            print(f"Metrics:")
            print(f"  Abstract Cosine: {abstract_cosine:.4f}")
            print(f"  Reference Cosine: {reference_cosine:.4f}")
            print(f"  Citation Cosine: {citation_cosine:.4f}")
            print(f"  Shared Authors: {shared_authors}")
            print(f"  Shared Citations: {shared_citations}")
            print(f"  Shared References: {shared_references}")
            print("-" * 80 + "\n")

    except Exception as e:
        print(f"Error during comparison: {str(e)}")
        raise

def save_similarity_results(seed_paper_df, returned_papers_df, model_path, model_name, output_path=None):
            # If no output path specified, create one using model name
            if output_path is None:
                base_path = f"similarity_results_{model_name}"
                extension = ".txt"
                counter = 1
                final_path = f"{base_path}{extension}"
                
                while os.path.exists(final_path):
                    counter += 1
                    final_path = f"{base_path}{counter}{extension}"
            else:
                final_path = output_path
            
            try:
                # Initialize model
                inference = ModelInference(model_path)
                
                # Get seed paper
                seed_paper = seed_paper_df.iloc[0]
                
                # Open file to write results
                with open(final_path, 'w', encoding='utf-8-sig') as f:
                    # Write model path at top
                    f.write(f"Model Path: {model_path}\n")
                    f.write("=" * 80 + "\n\n")
                    
                    f.write(f"Seed Paper: {seed_paper.get('Title', 'Unknown Title')}\n")
                    f.write(f"Abstract: {seed_paper.get('Abstract', 'No Abstract')}\n")
                    f.write("=" * 80 + "\n\n")
                    
                    results = []
                    
                    for idx, returned_paper in returned_papers_df.iterrows():
                        try:
                            seed_scibert = convert_string_to_array(seed_paper.get('SciBert', '[]'))
                            returned_scibert = convert_string_to_array(returned_paper.get('SciBert', '[]'))

                            abstract_cosine = calculate_cosine_similarity(
                                seed_paper.get('SciBert', []), 
                                returned_paper.get('SciBert', [])
                            )
                            
                            # Calculate cosines using TF-IDF for Citations and References
                            citation_cosine = calculate_text_cosine(
                                str(seed_paper.get('Citations', '')), 
                                str(returned_paper.get('Citations', ''))
                            )
                            
                            reference_cosine = calculate_text_cosine(
                                str(seed_paper.get('References', '')), 
                                str(returned_paper.get('References', ''))
                            )
                            
                            shared_authors = get_shared_count(
                                seed_paper.get('Authors', ''), 
                                returned_paper.get('Authors', '')
                            )
                            
                            shared_citations = get_shared_count(
                                seed_paper.get('Citations', ''), 
                                returned_paper.get('Citations', '')
                            )
                            
                            shared_references = get_shared_count(
                                seed_paper.get('References', ''), 
                                returned_paper.get('References', '')
                            )

                            similarity_score = inference.predict_similarity(
                                paper1_Citation_Count=int(seed_paper.get('Citation_Count', 0)),
                                paper1_Reference_Count=int(seed_paper.get('Reference_Count', 0)),
                                paper1_SciBert=seed_scibert,
                                paper2_Citation_Count=int(returned_paper.get('Citation_Count', 0)),
                                paper2_Reference_Count=int(returned_paper.get('Reference_Count', 0)),
                                paper2_SciBert=returned_scibert,
                                shared_author_count=shared_authors,
                                shared_reference_count=shared_references,
                                shared_citation_count=shared_citations,
                                reference_cosine=reference_cosine,
                                citation_cosine=citation_cosine,
                                abstract_cosine=abstract_cosine
                            )
                            
                            results.append({
                                'title': returned_paper.get('Title', 'Unknown Title'),
                                'abstract': returned_paper.get('Abstract', 'No Abstract'),
                                'score': similarity_score,
                                'metrics': {
                                    'abstract_cosine': abstract_cosine,
                                    'reference_cosine': reference_cosine,
                                    'citation_cosine': citation_cosine,
                                    'shared_authors': shared_authors,
                                    'shared_citations': shared_citations,
                                    'shared_references': shared_references
                                }
                            })
                            
                        except Exception as e:
                            print(f"Error processing paper {idx}: {str(e)}")
                            continue
                    
                    # Sort results by similarity score
                    results.sort(key=lambda x: x['score'], reverse=True)
                    
                    # Write sorted results to file
                    for i, result in enumerate(results, 1):
                        f.write(f"Rank {i}:\n")
                        f.write(f"Abstract: {normalize_text(result['abstract'])}\n")
                        f.write(f"Title: {normalize_text(result['title'])}\n")
                        f.write(f"Similarity Score: {result['score']:.4f}\n")
                        f.write("Detailed Metrics:\n")
                        f.write(f"  Abstract Cosine: {result['metrics']['abstract_cosine']:.4f}\n")
                        f.write(f"  Reference Cosine: {result['metrics']['reference_cosine']:.4f}\n")
                        f.write(f"  Citation Cosine: {result['metrics']['citation_cosine']:.4f}\n")
                        f.write(f"  Shared Authors: {result['metrics']['shared_authors']}\n")
                        f.write(f"  Shared Citations: {result['metrics']['shared_citations']}\n")
                        f.write(f"  Shared References: {result['metrics']['shared_references']}\n")
                        f.write("-" * 80 + "\n\n")
                    
                    print(f"Results saved to {final_path}")
                    
            except Exception as e:
                print(f"Error saving results: {str(e)}")
                raise
def normalize_text(text):
    if not isinstance(text, str):
        return text
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

if __name__ == "__main__":
    # Load data
    seed_paper_df = pd.read_excel('C:\\reactOffline\\similarity\\backend\\testing\\seedPaper.xlsx')
    returned_papers_df = pd.read_excel('C:\\reactOffline\\similarity\\backend\\testing\\returnedPaper.xlsx')
    
    # Extract model name from path
    modelName = 'best_model_fold_4.pth' # actually sued for logic, change this if want to change model
    model_name = 'best_model_fold_4'# Used for file creation without extension
    
    # Run comparison
    compare_papers(
        seed_paper_df,
        returned_papers_df,
        model_path="C:\\reactOffline\\similarity\\backend\\modelFolder\\" + modelName
    )
    
    # Save results to file
    save_similarity_results(
        seed_paper_df,
        returned_papers_df,
        model_path="C:\\reactOffline\\similarity\\backend\\modelFolder\\" + modelName,
        model_name=model_name  # Add this parameter
    )