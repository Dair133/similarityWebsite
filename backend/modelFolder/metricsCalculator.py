import numpy as np
import torch
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MetricsCalculator:
    def __init__(self):
        self.batch_size = 8
        self.max_workers = 4  
        self.thread_local = threading.local()
        
        # Check if CUDA is available and set device accordingly
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
    def _process_single_paper(self, args) -> Dict:
        """Process a single paper - used for parallel processing"""
        seed_paper, comparison_paper, tokenizer, model = args
        return self.calculate_paper_comparison_metrics(seed_paper, comparison_paper, tokenizer, model)
        
    def process_papers_parallel(self, seed_paper: Dict, papers: List[Dict], tokenizer, model) -> List[Dict]:
        """Process multiple papers in parallel"""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Create args for each paper
            args_list = [(seed_paper, paper, tokenizer, model) for paper in papers]
            # Process papers in parallel
            results = list(executor.map(self._process_single_paper, args_list))
            return results
            
    def batch_get_embeddings(self, abstracts: List[str], tokenizer, model) -> List[torch.Tensor]:
        """Process multiple abstracts in batches with CUDA support"""
        embeddings = []
        for i in range(0, len(abstracts), self.batch_size):
            batch = abstracts[i:i + self.batch_size]
            valid_abstracts = [text for text in batch if text and isinstance(text, str) and text.strip()]
            
            if not valid_abstracts:
                # Create zero tensors on the correct device
                embeddings.extend([torch.zeros(768, device=self.device) for _ in batch])
                continue
                
            with torch.no_grad():
                inputs = tokenizer(
                    valid_abstracts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                
                # Move input tensors to the appropriate device (CPU or CUDA)
                inputs = {key: val.to(self.device) for key, val in inputs.items()}
                
                # Model forward pass
                outputs = model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :]
                
            # Add embeddings and zeros for invalid abstracts
            curr_valid_idx = 0
            for text in batch:
                if text and isinstance(text, str) and text.strip():
                    embeddings.append(batch_embeddings[curr_valid_idx])
                    curr_valid_idx += 1
                else:
                    embeddings.append(torch.zeros(768, device=self.device))
                    
        return embeddings

    @torch.no_grad()
    def get_scibert_embedding(self, text: str, tokenizer, model) -> torch.Tensor:
        """Single text processing with CUDA support"""
        if not text or not isinstance(text, str) or not text.strip():
            return torch.zeros(768, device=self.device)
            
        try:
            inputs = tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Move input tensors to the appropriate device (CPU or CUDA)
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            
            outputs = model(**inputs)
            return outputs.last_hidden_state[0, 0, :]
            
        except Exception as e:
            print(f"Error generating SciBERT embedding: {str(e)}")
            return torch.zeros(768, device=self.device)

    def calculate_paper_comparison_metrics(self, seed_paper: Dict[str, Any], comparison_paper: Dict[str, Any], tokenizer, model) -> Dict[str, Any]:
        try:
            metrics = {
                'shared_author_count': 0,
                'shared_reference_count': 0,
                'shared_citation_count': 0,
                'reference_cosine': 0.0,
                'citation_cosine': 0.0,
                'abstract_cosine': 0.0
            }
            
            paper_info = comparison_paper['paper_info']
            seed_info = seed_paper['paper_info']
            
            # Process abstract and get embedding
            comp_abstract = paper_info.get('abstract', '')
            comp_embedding = self.get_scibert_embedding(comp_abstract, tokenizer, model)
            
            # Convert tensor to list for JSON serialization, ensuring it's moved to CPU first
            paper_info['scibert'] = comp_embedding.cpu().tolist()
            
            # Use set operations for faster computation
            seed_authors = frozenset(seed_info.get('authors', []))
            comp_authors = frozenset(paper_info.get('authors', []))
            seed_refs = frozenset(seed_info.get('references', []))
            comp_refs = frozenset(paper_info.get('references', []))
            seed_cites = frozenset(seed_info.get('citations', []))
            comp_cites = frozenset(paper_info.get('citations', []))
            
            # Compute set metrics
            metrics.update({
                'shared_author_count': len(seed_authors & comp_authors),
                'shared_reference_count': len(seed_refs & comp_refs),
                'shared_citation_count': len(seed_cites & comp_cites)
            })
            
            # Convert lists of strings to one long string for references and citations
            seed_references = ' '.join(seed_info.get('references', [])) if isinstance(seed_info.get('references', []), list) else seed_info.get('references', '')
            comp_references = ' '.join(paper_info.get('references', [])) if isinstance(paper_info.get('references', []), list) else paper_info.get('references', '')
            seed_citations = ' '.join(seed_info.get('citations', [])) if isinstance(seed_info.get('citations', []), list) else seed_info.get('citations', '')
            comp_citations = ' '.join(paper_info.get('citations', [])) if isinstance(paper_info.get('citations', []), list) else paper_info.get('citations', '')
            
            # Calculate cosine similarities in parallel
            if seed_references and comp_references:
                metrics['reference_cosine'] = self.calculate_text_cosine(seed_references, comp_references)
            if seed_citations and comp_citations:
                metrics['citation_cosine'] = self.calculate_text_cosine(seed_citations, comp_citations)
            
            # Calculate abstract cosine similarity
            seed_abstract = seed_info.get('abstract', '')
            comp_abstract = paper_info.get('abstract', '')
            if seed_abstract and comp_abstract:
                metrics['abstract_cosine'] = self.calculate_text_cosine(seed_abstract, comp_abstract)
            
            return metrics
            
        except Exception as e:
            print(f"Error in metrics calculation: {str(e)}")
            raise

    def calculate_text_cosine(self, text1, text2):
        """Calculate cosine similarity between two text strings using TF-IDF."""
        if not isinstance(text1, str) or not isinstance(text2, str):
            print('text1 and text2 are not strings, they are ', type(text1), type(text2))
            return 0
        
        # Clean and validate texts
        text1 = text1.strip().lower()
        text2 = text2.strip().lower()
        if not text1 or not text2:
            print('text1 or text2 is empty')
            return 0
        
        # Use TF-IDF for cosine similarity
        try:
            vectorizer = TfidfVectorizer().fit([text1, text2])
            vectors = vectorizer.transform([text1, text2])
            return cosine_similarity(vectors[0], vectors[1])[0][0]
        except Exception as e:
            print(f"Error in text cosine calculation: {str(e)}")
            return 0

    def get_relatively_similar_papers(self, papers, min_results=50, max_results=80):
        # First, ensure we have valid similarity scores and can access them
        try:
            # Handle both dictionary and list formats
            if isinstance(papers, list):
                # Make sure each paper has a similarity_score
                papers = [p for p in papers if isinstance(p, dict) and 'similarity_score' in p]
            else:
                print("Papers input is not a list, type:", type(papers))
                return []
                
            # If we have very few papers, return them all
            if len(papers) <= min_results:
                return papers
            
            # Sort papers by similarity score
            sorted_papers = sorted(papers, key=lambda x: float(x.get('similarity_score', 0)), reverse=True)
            sorted_scores = [float(p.get('similarity_score', 0)) for p in sorted_papers]
            
            # Calculate gaps between consecutive scores
            gaps = [sorted_scores[i] - sorted_scores[i+1] for i in range(len(sorted_scores)-1)]
            
            if len(gaps) > 0:
                # Consider a larger portion of papers for gap analysis (90%)
                consider_until = max(int(len(gaps) * 0.9), min_results)
                largest_gap = max(gaps[:consider_until])
                gap_index = gaps.index(largest_gap)
                
                # Make gap significance threshold more lenient (3%)
                score_range = max(sorted_scores) - min(sorted_scores)
                if largest_gap > score_range * 0.03:
                    # Ensure we return at least min_results papers
                    result_count = max(gap_index + 1, min_results)
                    result_count = min(result_count, max_results)  # But not more than max_results
                    return sorted_papers[:result_count]
                else:
                    # More lenient statistical approach
                    mean = np.mean(sorted_scores)
                    std = np.std(sorted_scores)
                    
                    # Try increasingly lenient thresholds until we get enough results
                    for threshold_multiplier in [0.0, -0.25, -0.5, -0.75, -1.0]:
                        threshold = mean + threshold_multiplier * std
                        filtered_papers = [p for p in sorted_papers if float(p.get('similarity_score', 0)) > threshold]
                        
                        if len(filtered_papers) >= min_results:
                            # If we have enough papers, return them (up to max_results)
                            return filtered_papers[:max_results]
                    
                    # If we still don't have enough papers, just return top min_results
                    return sorted_papers[:min_results]
            
            # If no gaps (unlikely), return top papers up to max_results
            return sorted_papers[:max_results]
            
        except Exception as e:
            print("Error in get_relatively_similar_papers:", str(e))
            # Add some debug printing
            print("Papers type:", type(papers))
            if isinstance(papers, list) and len(papers) > 0:
                print("First paper type:", type(papers[0]))
                print("First paper content:", papers[0])
            return []
     

     
    def parse_attribute_list(self, attributeList: str, delimiter: str) -> List[str]:
        listResults = attributeList.split(delimiter)
        return listResults
        
    def compareAttribute(self, seedAttributeArray, secondAttributeArray): 
     # Ensure secondAttributeArray is a list and not a string
     if isinstance(secondAttributeArray, str):
        secondAttributeArray = secondAttributeArray.split(',')
    
     # Normalize both arrays to ensure fair comparison
     normalized_seed = [attr.strip().lower() for attr in seedAttributeArray if attr.strip()]
     normalized_second = [attr.strip().lower() for attr in secondAttributeArray if attr.strip()]

     shared_items = set(normalized_seed).intersection(set(normalized_second))
     return len(shared_items)

                    
    def recommend_authors(self, seed_paper: dict, similar_papers: list, min_appearances: int = 2) -> list:
        # Extract seed paper authors
        seed_authors = set()
        if isinstance(seed_paper['paper_info'].get('authors', ''), list):
            seed_authors = {author.lower() for author in seed_paper['paper_info'].get('authors', [])}
        else:
            # If authors is a string, split by comma
            seed_authors = {author.strip().lower() for author in seed_paper['paper_info'].get('authors', '').split(',') if author.strip()}
        
        # Count author appearances in similar papers
        author_counts = {}
        for paper in similar_papers:
            paper_authors = []
            if isinstance(paper['paper_info'].get('authors', ''), list):
                paper_authors = paper['paper_info'].get('authors', [])
            else:
                # If authors is a string, split by comma
                paper_authors = [author.strip() for author in paper['paper_info'].get('authors', '').split(',') if author.strip()]
            
            # Count each unique author once per paper
            for author in paper_authors:
                if not author or author.lower() in seed_authors:
                    continue
                
                author_counts[author] = author_counts.get(author, 0) + 1
        
        # Filter by minimum appearances and sort by count
        recommendations = [
            {"author": author, "appearances": count, "rank": 0}
            for author, count in author_counts.items()
            if count >= min_appearances
        ]
        
        # Sort by appearance count (descending)
        recommendations.sort(key=lambda x: x["appearances"], reverse=True)
        
        # Add ranking
        for i, rec in enumerate(recommendations):
            rec["rank"] = i + 1
        
        return recommendations

    def recommend_citations_references(self, seed_paper: dict, similar_papers: list, field: str = 'citations', min_appearances: int = 3) -> list:
        # Validate field
        if field not in ['citations', 'references']:
            raise ValueError("Field must be either 'citations' or 'references'")
        
        # Extract seed paper citations/references
        seed_items = set()
        seed_items_raw = seed_paper['paper_info'].get(field, '')
        
        if isinstance(seed_items_raw, list):
            seed_items = {item.lower() for item in seed_items_raw}
        else:
            # If it's a string, split by semicolon
            seed_items = {item.strip().lower() for item in seed_items_raw.split(';') if item.strip()}
        
        # Count citation/reference appearances in similar papers
        item_counts = {}
        for paper in similar_papers:
            paper_items = []
            
            paper_items_raw = paper['paper_info'].get(field, '')
            if isinstance(paper_items_raw, list):
                paper_items = paper_items_raw
            else:
                # If it's a string, split by semicolon
                paper_items = [item.strip() for item in paper_items_raw.split(';') if item.strip()]
            
            # Count each unique citation/reference once per paper
            for item in paper_items:
                if not item or item.lower() in seed_items:
                    continue
                
                item_counts[item] = item_counts.get(item, 0) + 1
        
        # Filter by minimum appearances and sort by count
        recommendations = [
            {"title": item, "appearances": count, "rank": 0}
            for item, count in item_counts.items()
            if count >= min_appearances
        ]
        
        # Sort by appearance count (descending)
        recommendations.sort(key=lambda x: x["appearances"], reverse=True)
        
        # Add ranking
        for i, rec in enumerate(recommendations):
            rec["rank"] = i + 1
        
        return recommendations

    def get_recommendations(self, seed_paper: dict, similar_papers: list) -> dict:
        return {
            "recommended_authors": self.recommend_authors(seed_paper, similar_papers, min_appearances=2),
            "recommended_citations": self.recommend_citations_references(seed_paper, similar_papers, field='citations', min_appearances=3),
            "recommended_references": self.recommend_citations_references(seed_paper, similar_papers, field='references', min_appearances=3)
        }
        
        
        
    def calculate_shared_attributes(self, papers, parsed_seed_references, parsed_seed_citations, parsed_seed_authors):
     for paper in papers:
        # Get attributes from paper
        reference_list = paper['paper_info'].get('references', [])
        citation_list = paper['paper_info'].get('citations', [])
        author_list = paper['paper_info'].get('authors', [])
        
        # Calculate shared references
        shared_reference_count = self.compareAttribute(parsed_seed_references, reference_list)
        
        # Calculate shared citations
        shared_citation_count = self.compareAttribute(parsed_seed_citations, citation_list)
        
        # Format author list if it's a list and calculate shared authors
        if isinstance(author_list, list):
            author_list = ', '.join(author_list)  # Convert to string first if it's a list
        shared_author_count = self.compareAttribute(parsed_seed_authors, author_list)
        
        # Store comparison metrics in paper
        if 'comparison_metrics' not in paper:
            paper['comparison_metrics'] = {}
            
        paper['comparison_metrics']['shared_reference_count'] = shared_reference_count
        paper['comparison_metrics']['shared_citation_count'] = shared_citation_count
        paper['comparison_metrics']['shared_author_count'] = shared_author_count
    
     return papers
 
    def return_scibert_embeddings(self, generalPaperInfo, tokenizer, model):
        seed_abstract = generalPaperInfo['abstract']
        seed_embedding = self.get_scibert_embedding(seed_abstract, tokenizer, model)
        return seed_embedding.tolist()
        
 
    def return_attributes_lists(self, generalPaperInfo):
        # Here we will calculate shared references, citations and cosine
        seedReferenceList = generalPaperInfo['references']
        parsedSeedReferenceList = self.parse_attribute_list(seedReferenceList,';')
                
        seedCitationList = generalPaperInfo['citations']
        parsedSeedCitationList = self.parse_attribute_list(seedCitationList,';')
                
        seedAuthorList = generalPaperInfo['authors']
        if isinstance(seedAuthorList, list):
                seedAuthorList = ', '.join(seedAuthorList)
        parsedSeedAuthorList = self.parse_attribute_list(seedAuthorList, ',')
            
        return parsedSeedReferenceList, parsedSeedCitationList, parsedSeedAuthorList
        
        
        
        
    def extract_all_values(self, input_dict):
            """
            Extract all values from a dictionary into a single flat array.
            Ignores keys and just returns all values.
            
            Args:
                input_dict (dict): Dictionary with nested values
                
            Returns:
                list: A flat list of all values with cleaned formatting
            """
            all_values = []
            
            # Process all values in the dictionary
            for value in input_dict.values():
                if isinstance(value, list):
                    # If the value is a list, process each item
                    for item in value:
                        # Clean up the item
                        clean_item = self.clean_value(item)
                        if clean_item:
                            all_values.append(clean_item)
                else:
                    # If it's not a list, just clean and add it
                    clean_item = self.clean_value(value)
                    if clean_item:
                        all_values.append(clean_item)
            
            return all_values

    def clean_value(self, value):
            """Helper function to clean up a value string."""
            if not isinstance(value, str):
                return value
                
            # Remove brackets, quotes, and trailing punctuation
            result = value.strip()
            if result.startswith('['):
                result = result[1:]
            if result.endswith('];'):
                result = result[:-2]
            elif result.endswith(']'):
                result = result[:-1]
                
            # Remove any remaining quotes and extra whitespace
            result = result.strip('"\'').strip()
            
            return result
    def mark_gem_papers(self, relatively_similar_papers, scraped_titles):
        """
        Marks papers as GEM (true) if their title doesn't appear in scraped_titles.
        Includes thorough title normalization to prevent false positives.
        
        Args:
            relatively_similar_papers (list): List of paper objects from similarity results
            scraped_titles (list or dict): List or dictionary of titles returned from OpenAlex scraping
            
        Returns:
            list: The original papers list with GEM status added to each paper
        """
        # Convert scraped_titles to a list if it's not already
        titles_list = []
        if isinstance(scraped_titles, dict):
            # If it's a dictionary, try to extract titles
            if 'all_titles' in scraped_titles:
                titles_list = scraped_titles['all_titles']
            elif 'by_term' in scraped_titles:
                # Flatten the nested dictionary
                for term_titles in scraped_titles['by_term'].values():
                    titles_list.extend(term_titles)
            else:
                # Just use values directly
                for value in scraped_titles.values():
                    if isinstance(value, list):
                        titles_list.extend(value)
                    elif isinstance(value, str):
                        titles_list.append(value)
        elif isinstance(scraped_titles, list):
            titles_list = scraped_titles
        else:
            # Try to convert to a list if it's some other type
            try:
                titles_list = list(scraped_titles)
            except:
                titles_list = [str(scraped_titles)]
        
        # Function to normalize titles for comparison
        def normalize_title(title):
            if not title:
                return ""
            if not isinstance(title, str):
                title = str(title)
            
            # Convert to lowercase
            normalized = title.lower()
            
            # Remove special characters and extra whitespace
            import re
            # Remove punctuation
            normalized = re.sub(r'[^\w\s]', ' ', normalized)
            # Replace multiple spaces with a single space
            normalized = re.sub(r'\s+', ' ', normalized)
            # Remove leading/trailing whitespace
            normalized = normalized.strip()
            
            return normalized
        
        # Normalize the scraped titles for better comparison
        scraped_titles_normalized = [normalize_title(title) for title in titles_list]
        
        # Debug: Print some normalized scraped titles
        print(f"\nFound {len(titles_list)} scraped titles to check against")
        print("\nExample normalized scraped titles:")
        for i in range(min(3, len(titles_list))):
            print(f"Original: '{titles_list[i]}'")
            print(f"Normalized: '{scraped_titles_normalized[i]}'")
        
        # Counter for stats
        gem_count = 0
        total_count = 0
        
        # Process each paper
        for i, paper in enumerate(relatively_similar_papers):
            total_count += 1
            
            # Extract the paper title - handle different possible structures
            paper_title = None
            
            if 'paper_info' in paper and isinstance(paper['paper_info'], dict):
                if 'title' in paper['paper_info']:
                    paper_title = paper['paper_info']['title']
            elif 'title' in paper:
                paper_title = paper['title']
                
            # If we couldn't find a title, default to non-GEM
            if not paper_title:
                paper['is_gem'] = False
                continue
            
            # Normalize the paper title
            normalized_paper_title = normalize_title(paper_title)
            
            # Debug: Print paper title normalization
            if i < 3:  # Just print first few for debugging
                print(f"\nPaper #{i+1}:")
                print(f"Original title: '{paper_title}'")
                print(f"Normalized title: '{normalized_paper_title}'")
            
            # Check if this title appears in the scraped titles using normalized comparison
            # First try exact match
            title_in_scraped = normalized_paper_title in scraped_titles_normalized
            
            # If no exact match, try substring match
            if not title_in_scraped and len(normalized_paper_title) > 10:
                for scraped in scraped_titles_normalized:
                    # Check if paper title is contained within scraped title or vice versa
                    if normalized_paper_title in scraped or scraped in normalized_paper_title:
                        if i < 3:
                            print(f"Substring match found: '{scraped}'")
                        title_in_scraped = True
                        break
                    
                    # Simple word overlap check
                    if len(normalized_paper_title) > 0 and len(scraped) > 0:
                        paper_words = set(normalized_paper_title.split())
                        scraped_words = set(scraped.split())
                        
                        # If 70% of words match, consider it a match
                        if len(paper_words) > 0 and len(scraped_words) > 0:
                            overlap = len(paper_words.intersection(scraped_words))
                            smaller_set_size = min(len(paper_words), len(scraped_words))
                            
                            if overlap >= 0.7 * smaller_set_size and smaller_set_size >= 3:
                                if i < 3:
                                    print(f"Word overlap match: {overlap}/{smaller_set_size} words with: '{scraped}'")
                                title_in_scraped = True
                                break
            
            # Mark as GEM if the title is NOT in scraped titles
            paper['is_gem'] = not title_in_scraped
            
            if paper['is_gem']:
                gem_count += 1
                if i < 10:  # Print first few GEM papers
                    print(f"Marked as GEM: '{paper_title}'")
        
        # Print summary
        print(f"\nGEM Paper Analysis:")
        print(f"Found {gem_count} GEM papers out of {total_count} total papers ({(gem_count/total_count)*100:.1f}%)")
        
        return relatively_similar_papers
