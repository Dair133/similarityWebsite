from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import json
import time
import sys


class APIScraperClass:
    def __init__(self):
        pass

    def scrape_openalex_search_results(self, search_term, max_titles=100, max_pages=5, debug=True):
        """
        Searches OpenAlex for a given term and scrapes the titles from multiple pages of search results
        using Selenium WebDriver to fully render the JavaScript content.
        
        Args:
            search_term (str): The term to search for
            max_titles (int): Maximum number of titles to retrieve in total
            max_pages (int): Maximum number of pages to scrape (default: 5)
            debug (bool): If True, print verbose debugging info
            
        Returns:
            list: A list of titles found in the search results
        """
        # Convert parameters to integers to ensure they're the right type
        max_titles = int(max_titles)
        max_pages = int(max_pages)
        
        if debug:
            print("Setting up Chrome options...")
            
        # Configure Chrome options
        chrome_options = Options()
        #chrome_options.add_argument("--headless")  # Run in headless mode (no browser UI)
        chrome_options.add_argument("--disable-gpu")  # Disable GPU hardware acceleration
        chrome_options.add_argument("--window-size=1920,1080")  # Set window size
        chrome_options.add_argument("--no-sandbox")  # Bypass OS security model
        chrome_options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource problems
        
        # Create a new Chrome WebDriver without using webdriver_manager
        try:
            if debug:
                print("Initializing Chrome WebDriver...")
                
            # Initialize driver directly without webdriver_manager
            driver = webdriver.Chrome(options=chrome_options)
            
            if debug:
                print("WebDriver initialized successfully!")
        except Exception as e:
            print(f"Error initializing WebDriver: {e}")
            print("\nYou might need to download ChromeDriver manually:")
            print("1. Visit: https://chromedriver.chromium.org/downloads")
            print("2. Download the version matching your Chrome browser")
            print("3. Place the chromedriver executable in your PATH")
            return []
        
        all_titles = []
        current_page = 1
        
        try:
            while current_page <= max_pages and len(all_titles) < max_titles:
                # Construct search URL with the correct query parameter format and page number
                base_url = "https://openalex.org/works"
                search_url = f"{base_url}?page={current_page}&filter=title_and_abstract.search%3A{search_term.replace(' ', '%20')}"
                
                print(f"\nSearching OpenAlex for: {search_term} - Page {current_page}/{max_pages}")
                print(f"Opening URL: {search_url}")
                
                # Navigate to the search URL
                driver.get(search_url)
                
                if debug:
                    print("URL loaded, waiting for content to appear...")
                
                # Wait for the page to load (adjust timeout as needed)
                wait = WebDriverWait(driver, 20)
                try:
                    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.v-list-item__title")))
                    if debug:
                        print("Initial elements found, waiting 3 seconds for full page load...")
                        
                    # Give a bit more time for all content to load
                    time.sleep(0.75)
                    
                    print(f"Page {current_page} loaded, extracting titles...")
                    
                    # Find all title elements using the exact format you provided
                    title_elements = driver.find_elements(By.CSS_SELECTOR, 'div.v-list-item__title > div')
                    
                    page_titles = []
                    if title_elements:
                        for element in title_elements:
                            title_text = element.text.strip()
                            if title_text and title_text not in page_titles:
                                page_titles.append(title_text)
                    else:
                        print("No titles found with the exact format. Trying alternative approaches...")
                        
                        # Try parent elements
                        parent_elements = driver.find_elements(By.CSS_SELECTOR, 'div.v-list-item__title')
                        for element in parent_elements:
                            title_text = element.text.strip()
                            if title_text and title_text not in page_titles:
                                page_titles.append(title_text)
                    
                    # Add new unique titles to the all_titles list
                    new_titles_count = 0
                    for title in page_titles:
                        if title not in all_titles:
                            all_titles.append(title)
                            new_titles_count += 1
                            if len(all_titles) >= max_titles:
                                break
                    
                    print(f"Found {len(page_titles)} titles on page {current_page} ({new_titles_count} new)")
                    
                    # Check if we found any new titles on this page
                    if new_titles_count == 0:
                        print(f"No new titles found on page {current_page}. There might not be more results.")
                        break
                    
                    # Check if we've reached the maximum number of titles
                    if len(all_titles) >= max_titles:
                        print(f"Reached the maximum number of titles ({max_titles}). Stopping search.")
                        break
                    
                    # Check if there are more pages by looking for pagination buttons
                    try:
                        # First, try to find any pagination-related elements for debugging
                        if debug:
                            print("Searching for pagination elements...")
                            all_pagination = driver.find_elements(By.CSS_SELECTOR, '.v-pagination, .pagination, nav.v-pagination, .v-pagination__item, [aria-label*="page"], [aria-label*="Page"]')
                            print(f"Found {len(all_pagination)} potential pagination elements")
                            for idx, elem in enumerate(all_pagination[:5]):  # Show first 5 to avoid cluttering output
                                try:
                                    elem_html = elem.get_attribute('outerHTML')
                                    print(f"Pagination element {idx}: {elem_html[:100]}...")  # Show first 100 chars
                                except:
                                    print(f"Pagination element {idx}: [Error getting HTML]")
                        
                        # Look for numerical page buttons like the one provided
                        pagination_buttons = driver.find_elements(By.CSS_SELECTOR, 'button.v-pagination__item, button[aria-label*="Goto Page"]')
                        
                        if debug:
                            print(f"Found {len(pagination_buttons)} specific pagination buttons")
                        
                        # Filter out the current page button
                        next_page_exists = False
                        for button in pagination_buttons:
                            button_text = button.text.strip()
                            aria_label = button.get_attribute('aria-label') or ''
                            
                            if button_text.isdigit() and int(button_text) > current_page:
                                next_page_exists = True
                                if debug:
                                    print(f"Found button for page {button_text}")
                                break
                            elif 'Goto Page' in aria_label:
                                page_num = aria_label.replace('Goto Page', '').strip()
                                if page_num.isdigit() and int(page_num) > current_page:
                                    next_page_exists = True
                                    if debug:
                                        print(f"Found button with aria-label for page {page_num}")
                                    break
                        
                        if not next_page_exists:
                            # One more attempt: try to find specific "next page" buttons
                            next_buttons = driver.find_elements(By.CSS_SELECTOR, 'button.v-pagination__item--next, [aria-label="Next page"], [aria-label="next"], button.next')
                            if next_buttons and not "disabled" in next_buttons[0].get_attribute("class"):
                                next_page_exists = True
                                if debug:
                                    print("Found 'Next page' button")
                        
                        if not next_page_exists:
                            print("No more pages available (no pagination buttons for next pages found).")
                            break
                            
                    except Exception as pagination_error:
                        print(f"Error checking pagination: {pagination_error}")
                        print("Will try to continue to next page anyway...")
                    
                    # Move to the next page
                    current_page += 1
                    
                except Exception as page_error:
                    print(f"Error loading page {current_page}: {page_error}")
                    # Try to save the page source for debugging
                    try:
                        with open(f"openalex_page{current_page}_source.html", "w", encoding="utf-8") as f:
                            f.write(driver.page_source)
                        print(f"Page source saved to openalex_page{current_page}_source.html for inspection")
                    except:
                        pass
                    break
            
            # Print a sample of found titles
            if all_titles:
                print(f"\nFound {len(all_titles)} total unique titles across {current_page} pages:")
                for i, title in enumerate(all_titles[:5], 1):
                    print(f"  {i}. {title}")
                if len(all_titles) > 5:
                    print(f"  ... and {len(all_titles)-5} more")
            else:
                print("No titles found across all pages.")
                
            return all_titles[:max_titles]
            
        except Exception as e:
            print(f"Error scraping OpenAlex: {e}")
            return all_titles  # Return any titles we've found so far
        
        finally:
            # Always close the driver to avoid memory leaks
            if debug:
                print("Closing WebDriver...")
            driver.quit()