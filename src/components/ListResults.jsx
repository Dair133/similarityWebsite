import React, { useState, useRef, useEffect } from 'react';
import FadeIn from 'react-fade-in';
import EnhancedPaperView from './EnhancedPaperView';
import SimplePulseButton from './module/buttons/PulseButton';
import Spinner from './module/animations/Spinner';

function ListResults({ results, toggleGraphView, setParentResults, showGraph, onClearPdf, hoveredNodeIndex }) {
  const [localResults, setLocalResults] = useState(null);
  const [tooltipVisible, setTooltipVisible] = useState({});
  const [selectedPaper, setSelectedPaper] = useState(null);
  const [isProcessingSeedPaper, setIsProcessingSeedPaper] = useState(false);
  const [processingError, setProcessingError] = useState(null);
  
  // Create refs for each paper item
  const paperRefs = useRef([]);
  
  const displayResults = localResults || results;
  
  // Reset paper refs when results change
  useEffect(() => {
    if (displayResults && displayResults.similarity_results) {
      paperRefs.current = displayResults.similarity_results.map(() => React.createRef());
    }
  }, [displayResults]);
  
  // Scroll to paper when hoveredNodeIndex changes
  useEffect(() => {
    if (hoveredNodeIndex !== null && hoveredNodeIndex !== undefined && 
        paperRefs.current[hoveredNodeIndex] && 
        paperRefs.current[hoveredNodeIndex].current) {
      
      paperRefs.current[hoveredNodeIndex].current.scrollIntoView({
        behavior: 'smooth',
        block: 'center'
      });
    }
  }, [hoveredNodeIndex]);

  const styles = {
    container: {
      width: '100%',
      height: '100%',
      backgroundColor: '#10253E',
      color: '#F7F3E9',
      padding: '2%',
      boxSizing: 'border-box',
      overflow: 'auto',
      fontFamily: '"Source Sans Pro", sans-serif',
    },
    title: {
      fontSize: '24px',
      color: '#F7F3E9',
      marginBottom: '20px',
      fontFamily: '"Montserrat", sans-serif',
      fontWeight: '600',
    },
    toggleContainer: {
      marginBottom: '20px',
    },
    switchContainer: {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      width: '260px',
      height: '40px',
      backgroundColor: '#14304D',
      borderRadius: '20px',
      padding: '2px',
      position: 'relative',
    },
    switchOption: {
      flex: 1,
      textAlign: 'center',
      padding: '8px 16px',
      cursor: 'pointer',
      zIndex: 2,
      transition: 'color 0.3s',
      userSelect: 'none',
      color: '#81A4CD',
    },
    activeOption: {
      color: 'white',
    },
    slider: {
      position: 'absolute',
      top: '2px',
      bottom: '2px',
      width: '20%',
      backgroundColor: '#3E7CB9',
      borderRadius: '18px',
      transition: 'left 0.3s ease-in-out',
      zIndex: 1,
    },
    uploadArea: {
      width: '100%',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
    },
    input: {
      width: '100%',
      padding: '10px',
      border: '2px dashed #81A4CD',
      borderRadius: '4px',
      cursor: 'pointer',
      backgroundColor: '#14304D',
      color: '#F7F3E9',
      fontFamily: '"Source Sans Pro", sans-serif',
    },
    fileInfo: {
      marginTop: '10px',
      color: '#EEE8D9',
      fontSize: '14px',
      fontFamily: '"Source Sans Pro", sans-serif',
    },
    results: {
      marginTop: '20px',
      width: '100%',
      textAlign: 'left',
    },
    button: {
      backgroundColor: '#3E7CB9',
      border: 'none',
      color: '#F7F3E9',
      padding: '10px 20px',
      textAlign: 'center',
      textDecoration: 'none',
      display: 'inline-block',
      fontSize: '16px',
      margin: '4px 2px',
      cursor: 'pointer',
      borderRadius: '4px',
      fontFamily: '"Montserrat", sans-serif',
      fontWeight: '500',
      transition: 'background-color 0.3s',
    },
    buttonHover: {
      backgroundColor: '#2A5986',
    },
    NodeGraphContainer: {
      visibility: 'hidden',
    },
    sectionHeading: {
      fontFamily: '"Montserrat", sans-serif',
      fontSize: '20px',
      fontWeight: '600',
      color: '#F7F3E9',
      marginTop: '25px',
      marginBottom: '15px',
    },
    listItem: {
      backgroundColor: '#14304D',
      padding: '15px',
      marginBottom: '15px',
      borderRadius: '6px',
      borderLeft: '3px solid #3E7CB9',
      transition: 'all 0.3s ease',
    },
    highlightedItem: {
      boxShadow: '0 0 15px #3E7CB9',
      border: '2px solid #3E7CB9',
      backgroundColor: '#1A3A5F',
    },
    paperTitle: {
      fontFamily: '"Montserrat", sans-serif',
      fontSize: '18px',
      fontWeight: '600',
      color: '#EEE8D9',
      marginBottom: '10px',
    },
    paperInfo: {
      fontFamily: '"Source Sans Pro", sans-serif',
      fontSize: '14px',
      lineHeight: '1.6',
      marginBottom: '8px',
      color: '#F7F3E9',
    },
    paperMetric: {
      color: '#81A4CD',
      fontWeight: '600',
    },
    paperAbstract: {
      fontFamily: '"Source Sans Pro", sans-serif',
      fontSize: '15px',
      lineHeight: '1.6',
      color: '#F7F3E9',
      marginTop: '10px',
      marginBottom: '5px',
    },
    overlapContainer: {
      backgroundColor: '#14304D',
      borderRadius: '6px',
      border: '1px solid #3E7CB9',
      marginBottom: '15px',
      marginTop: '10px',
    },
    overlapHeader: {
      display: 'flex',
      alignItems: 'center',
      padding: '8px 12px',
      backgroundColor: '#1A3A5F',
      borderBottom: '1px solid #3E7CB9',
      borderRadius: '6px 6px 0 0',
    },
    overlapTitle: {
      fontFamily: '"Montserrat", sans-serif',
      fontWeight: '600',
      color: '#F7F3E9',
      margin: 0,
      fontSize: '14px',
    },
    helpIcon: {
      marginLeft: '8px',
      width: '16px',
      height: '16px',
      backgroundColor: '#3E7CB9',
      color: '#F7F3E9',
      borderRadius: '50%',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontSize: '12px',
      cursor: 'pointer',
      position: 'relative',
    },
    tooltip: {
      position: 'absolute',
      backgroundColor: '#14304D',
      border: '1px solid #3E7CB9',
      borderRadius: '4px',
      padding: '8px 12px',
      color: '#F7F3E9',
      fontSize: '12px',
      width: '200px',
      top: '20px',
      left: '10px',
      zIndex: 100,
      boxShadow: '0 2px 8px rgba(0,0,0,0.2)',
      opacity: 0,
      visibility: 'hidden',
      transition: 'opacity 0.2s, visibility 0.2s',
      pointerEvents: 'none',
    },
    overlapContent: {
      padding: '12px',
    },
    overlapTag: {
      display: 'inline-block',
      padding: '5px 10px',
      backgroundColor: '#38729E',
      color: '#F7F3E9',
      borderRadius: '4px',
      fontSize: '13px',
      fontFamily: '"Source Sans Pro", sans-serif',
      margin: '2px 4px 2px 0',
    },
    overlapTypeLabel: {
      marginLeft: '8px',
      fontSize: '12px',
      color: '#81A4CD',
    },
    overlapItem: {
      marginBottom: '8px',
    },
    divider: {
      height: '1px',
      backgroundColor: '#2C4C72',
      margin: '12px 0',
      opacity: 0.7,
    },
    sectionLabel: {
      fontSize: '13px',
      color: '#81A4CD',
      fontFamily: '"Montserrat", sans-serif',
      fontWeight: '500',
      marginBottom: '8px',
    },
    authorTag: {
      display: 'inline-block',
      padding: '5px 10px',
      backgroundColor: '#38729E',
      color: '#F7F3E9',
      borderRadius: '4px',
      fontSize: '13px',
      fontFamily: '"Source Sans Pro", sans-serif',
      margin: '2px 4px 2px 0',
    },
    enhancedViewButton: {
      backgroundColor: '#A4BEDC',
      color: '#10253E',
      border: 'none',
      padding: '8px 16px',
      textAlign: 'center',
      textDecoration: 'none',
      display: 'block',
      width: '100%',
      fontSize: '14px',
      margin: '10px 0 0 0',
      cursor: 'pointer',
      borderRadius: '4px',
      fontFamily: '"Montserrat", sans-serif',
      fontWeight: '500',
      transition: 'background-color 0.3s',
    },
    loadingContainer: {
      display: 'flex', 
      flexDirection: 'column', 
      alignItems: 'center', 
      justifyContent: 'center',
      height: '50%'
    },
    errorMessage: { 
      padding: '10px', 
      backgroundColor: '#ff6b6b30', 
      borderLeft: '3px solid #ff6b6b',
      marginBottom: '15px',
      color: '#F7F3E9'
    }
  };

  // Add keyframe animation for pulse effect
  useEffect(() => {
    // Create style element 
    const styleEl = document.createElement('style');
    
    // Define the keyframe animation
    const keyframes = `
      @keyframes pulse {
        0% { box-shadow: 0 0 5px rgba(62, 124, 185, 0.8); }
        50% { box-shadow: 0 0 20px rgba(62, 124, 185, 1); }
        100% { box-shadow: 0 0 5px rgba(62, 124, 185, 0.8); }
      }
    `;
    
    styleEl.innerHTML = keyframes;
    document.head.appendChild(styleEl);
    
    // Clean up
    return () => {
      document.head.removeChild(styleEl);
    };
  }, []);
  
  // Listen for custom event to show enhanced view
  useEffect(() => {
    const handleShowEnhancedViewEvent = (event) => {
      const { paper } = event.detail;
      if (paper) {
        handleShowEnhancedView(paper);
      }
    };
    
    // Add event listener
    document.addEventListener('showEnhancedView', handleShowEnhancedViewEvent);
    
    // Clean up
    return () => {
      document.removeEventListener('showEnhancedView', handleShowEnhancedViewEvent);
    };
  }, []);
  
  // Helper function to format authors
  const formatAuthors = (authors) => {
    if (!authors) return '';
    if (typeof authors === 'string') return authors;
    if (Array.isArray(authors)) return authors.join(', ');
    return JSON.stringify(authors);
  };

  // Helper function to format source type text
  const formatSourceType = (sourceType) => {
    if (!sourceType) return '';
    return sourceType
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  // Overlap Box Component
  const OverlapBox = ({ paper, index }) => {
    const hasMethodology = paper.source_info && (
      (Array.isArray(paper.source_info.search_term) && paper.source_info.search_term.length > 0) ||
      (!Array.isArray(paper.source_info.search_term) && paper.source_info.search_term)
    );

    const hasSharedAuthors = paper.comparison_metrics &&
      paper.comparison_metrics.shared_authors &&
      paper.comparison_metrics.shared_authors.length > 0;

    const showDivider = hasMethodology && hasSharedAuthors;

    return (
      <div style={styles.overlapContainer}>
        <div style={styles.overlapHeader}>
          <h4 style={styles.overlapTitle}>Overlap</h4>
          <div
            style={styles.helpIcon}
            onMouseEnter={() => {
              setTooltipVisible(prev => ({ ...prev, [index]: true }));
            }}
            onMouseLeave={() => {
              setTooltipVisible(prev => ({ ...prev, [index]: false }));
            }}
          >
            ?
            <div style={{
              ...styles.tooltip,
              opacity: tooltipVisible[index] ? 1 : 0,
              visibility: tooltipVisible[index] ? 'visible' : 'hidden'
            }}>
              Areas where both papers share methodology or concepts
            </div>
          </div>
        </div>
        <div style={styles.overlapContent}>
          {hasMethodology && (
            <>
              <div style={styles.sectionLabel}>Shared Methodology:</div>
              <div style={styles.overlapItem}>
                <span style={styles.overlapTag}>
                  {Array.isArray(paper.source_info.search_term)
                    ? paper.source_info.search_term[0]
                    : paper.source_info.search_term}
                </span>
                <span style={styles.overlapTypeLabel}>
                  ({formatSourceType(paper.source_info.search_type)})
                </span>
              </div>
            </>
          )}
          {hasMethodology && hasSharedAuthors && <div style={styles.divider}></div>}
          {hasSharedAuthors && (
            <>
              <div style={styles.sectionLabel}>Shared Authors:</div>
              <div style={styles.overlapItem}>
                <div style={{ display: 'flex', flexWrap: 'wrap' }}>
                  {paper.comparison_metrics.shared_authors.map((author, i) => (
                    <span key={i} style={styles.authorTag}>{author}</span>
                  ))}
                </div>
              </div>
            </>
          )}
          {paper.comparison_metrics &&
            paper.comparison_metrics.shared_references &&
            paper.comparison_metrics.shared_references.length > 0 && (
              <>
                {(hasMethodology || hasSharedAuthors) && <div style={styles.divider}></div>}
                <div style={styles.sectionLabel}>Shared References:</div>
                <div style={styles.overlapItem}>
                  <div style={{ display: 'flex', flexWrap: 'wrap' }}>
                    {paper.comparison_metrics.shared_references.slice(0, 5).map((reference, i) => (
                      <span key={i} style={{ ...styles.authorTag, backgroundColor: '#2A5278' }}>
                        {reference}
                      </span>
                    ))}
                    {paper.comparison_metrics.shared_references.length > 5 && (
                      <span style={{ color: '#81A4CD', fontSize: '13px', marginLeft: '5px', alignSelf: 'center' }}>
                        ... {paper.comparison_metrics.shared_references.length - 5} more
                      </span>
                    )}
                  </div>
                </div>
              </>
            )}
          {(!hasMethodology && !hasSharedAuthors) && (
            <div style={{ color: '#81A4CD', fontStyle: 'italic', fontSize: '13px' }}>
              No direct methodological or authorship overlap detected
            </div>
          )}
        </div>
      </div>
    );
  };

  // Function to handle showing the enhanced view
  const handleShowEnhancedView = (paper) => {
    setSelectedPaper(paper);
  };
  
  // Function to handle setting a paper as a seed paper
  const handleUseAsSeedPaper = async (paperData) => {
    // Show the loading state
    setIsProcessingSeedPaper(true);
    setProcessingError(null);
    
    try {
      // Prepare paper data to send to backend
      const payload = {
        paper_info: paperData.paper_info,
        search_terms: paperData.source_info?.search_term || [],
        search_type: paperData.source_info?.search_type || 'core_methodology'
      };
  
      // Make API call to the backend
      const response = await fetch('http://localhost:5000/use-as-seed-paper', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });
  
      if (!response.ok) {
        throw new Error(`Request failed with status: ${response.status}`);
      }
  
      const data = await response.json();
      
      // If successful, update the UI with new results
      if (data.success) {
        // Update both local state and parent state
        setLocalResults(data.results);
        if (setParentResults && typeof setParentResults === 'function') {
          setParentResults(data.results);
        }
      } else {
        throw new Error(data.error || 'Failed to set as seed paper');
      }
    } catch (error) {
      console.error('Error setting paper as seed:', error);
      setProcessingError(error.message || 'Failed to set paper as seed');
    } finally {
      setIsProcessingSeedPaper(false);
    }
  };
  
  const handleBackToList = (data = null) => {
    // First clear the selected paper
    setSelectedPaper(null);
    
    // Check if this is a special action to use a paper as seed paper
    if (data && data.action === 'use_as_seed_paper' && data.paperData) {
      // Process the paper as a seed paper
      handleUseAsSeedPaper(data.paperData);
    }
  };

  if (selectedPaper) {
    const uniqueKey = selectedPaper.paper_info?.doi || selectedPaper.paper_info?.title || `paper-${JSON.stringify(selectedPaper)}`; // Choose a reliable unique identifier
    return (
      <div style={styles.container}>
        <EnhancedPaperView
          // ---- Add the key prop here ----
          key={uniqueKey}
          // ---- Pass other props as before ----
          paper={selectedPaper}
          onBack={handleBackToList}
          seedPaper={localResults?.seed_paper || displayResults?.seed_paper} // Add optional chaining for displayResults
          onClearPdf={onClearPdf}
        />
      </div>
    );
  }

  // Helper function to generate fake results for testing
  const generateFakeResults = () => {
    const authorPool = [
      "Zhang, L.",
      "Smith, J.",
      "Johnson, K.",
      "Williams, R.",
      "Brown, M.",
      "Davis, T.",
      "Miller, S.",
      "Wilson, A.",
      "Moore, D.",
      "Taylor, P."
    ];

    // Create a simple soil science seed paper
    const seedPaper = {
      paper_info: {
        title: "Effects of Climate Change on Soil Microbial Communities",
        authors: ["Brown, M.", "Davis, T.", "Wilson, A."],
        abstract: "This study examines how rising temperatures and changing precipitation patterns affect soil microbial communities. We collected samples from various ecosystems and analyzed microbial diversity and activity under different simulated climate conditions.",
        full_content: "This is the full content of the soil science paper. It contains detailed information about soil microbial communities and how they respond to climate change. The methodology involved collecting soil samples from different ecosystems including forests, grasslands, and agricultural fields. We measured microbial biomass, respiration rates, and community composition using DNA sequencing. Our results indicate that warming temperatures generally increase microbial activity but decrease diversity in most soil types. Drought conditions had more variable effects depending on the ecosystem type. These findings have important implications for carbon cycling and ecosystem functions under future climate scenarios."
      }
    };

    const fakePapers = Array.from({ length: 10 }, (_, i) => {
      const paperAuthorCount = Math.floor(Math.random() * 3) + 1;
      const paperAuthors = [];
      for (let j = 0; j < paperAuthorCount; j++) {
        const randomIndex = Math.floor(Math.random() * authorPool.length);
        if (!paperAuthors.includes(authorPool[randomIndex])) {
          paperAuthors.push(authorPool[randomIndex]);
        }
      }

      const sharedAuthors = i % 2 === 1 ? [authorPool[0], authorPool[1]].slice(0, i % 3) : [];

      // Make one paper related to soil science
      let title, abstract;
      if (i === 2) {
        title = "Agricultural Management Practices and Soil Health Indicators";
        abstract = "Our study investigates how different farming methods affect soil quality. We found that crop rotation and reduced tillage significantly improve soil microbial activity compared to conventional practices.";
      } else {
        title = `Fake Paper Title ${i + 1}`;
        abstract = `This is a fake abstract for paper ${i + 1}. It's short and sweet.`;
      }

      return {
        paper_info: {
          title: title,
          abstract: abstract,
          authors: paperAuthors,
        },
        similarity_score: (1 - i * 0.1).toFixed(2),
        source_info: {
          search_term: i % 2 === 0 ? "minimal perturbation computation" : ["adversarial example generation", "neural network vulnerability"],
          search_type: i % 2 === 0 ? "core_methodology" : "conceptual_angles",
        },
        comparison_metrics: {
          shared_reference_count: Math.floor(Math.random() * 10),
          shared_citation_count: Math.floor(Math.random() * 10),
          shared_author_count: sharedAuthors.length,
          shared_authors: sharedAuthors,
          shared_references: i % 2 === 0 ?
            ["Smith et al. (2019)", "Johnson & Lee (2020)", "Williams (2018)"] :
            ["Brown et al. (2021)", "Davis (2019)", "Miller & Taylor (2020)",
              "Wilson (2017)", "Moore & Zhang (2022)", "Additional paper 1", "Additional paper 2"],
        },
      };
    });

    const fakeResultsData = {
      seed_paper: seedPaper,
      abstract_info: "Fake abstract information.",
      similarity_results: fakePapers,
      test: {
        compared_papers: fakePapers,
      },
    };

    setLocalResults(fakeResultsData);
    if (setParentResults) {
      setParentResults(fakeResultsData);
    }
  };

  return (
    <div style={styles.container}>
      {/* Show loading state when processing a seed paper */}
      {isProcessingSeedPaper ? (
        <div style={styles.loadingContainer}>
          <h2 style={styles.title}>Processing New Seed Paper</h2>
          <div style={{ marginBottom: '20px' }}>
            <Spinner size={32} color="#3498db" />
          </div>
          <p style={{ color: '#F7F3E9', textAlign: 'center' }}>
            Please wait while we find papers similar to your new seed paper...
          </p>
        </div>
      ) : (
        // Otherwise show the regular list view
        <>
          {/* Toggle component moved here from ParentDashboard */}
          <div style={styles.toggleContainer}>
            <div style={styles.switchContainer}>
              <span
                style={{
                  ...styles.switchOption,
                  ...(!showGraph ? styles.activeOption : {}),
                }}
                onClick={() => {
                  if (showGraph) toggleGraphView();
                }}
              >
                PDF View
              </span>
              <span
                style={{
                  ...styles.switchOption,
                  ...(showGraph ? styles.activeOption : {}),
                }}
                onClick={() => {
                  if (!showGraph) toggleGraphView();
                }}
              >
                Node Graph View
              </span>
              <div
                style={{
                  ...styles.slider,
                  left: showGraph ? 'calc(50% - 2px)' : '2px',
                }}
              />
            </div>
          </div>
  
          <h2 style={styles.title}>List Of Results</h2>
          
          {/* Show error message if there was an error processing the seed paper */}
          {processingError && (
            <div style={styles.errorMessage}>
              <strong>Error:</strong> {processingError}
            </div>
          )}
          
          <button 
            style={styles.button} 
            onClick={generateFakeResults}
            onMouseOver={(e) => {
              e.currentTarget.style.backgroundColor = '#2A5986';
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.backgroundColor = '#3E7CB9';
            }}
          >
            Generate Examples
          </button>
  
          {displayResults && (
            <div style={styles.results}>
              <h3 style={styles.sectionHeading}>Seed Paper</h3>
              <div style={styles.listItem}>
                <div style={styles.paperTitle}>{displayResults.seed_paper.paper_info.title}</div>
                {displayResults.seed_paper.paper_info.authors && (
                  <div style={styles.paperInfo}>
                    <strong>Authors: </strong>
                    {formatAuthors(displayResults.seed_paper.paper_info.authors)}
                  </div>
                )}
                <div style={styles.paperAbstract}>
                  {displayResults.seed_paper.paper_info.abstract}
                </div>
              </div>
  
              <h3 style={styles.sectionHeading}>Similar Papers</h3>
              <ol style={{ listStyle: 'none', padding: 0 }}>
                <FadeIn>
                  {displayResults.similarity_results.map((paper, index) => (
                    <li 
                      key={index} 
                      ref={paperRefs.current[index]}
                      style={{
                        ...styles.listItem,
                        ...(hoveredNodeIndex === index ? styles.highlightedItem : {}),
                        animation: hoveredNodeIndex === index ? 'pulse 1.5s infinite' : 'none'
                      }}
                    >
                      <div style={styles.paperTitle}>{paper.paper_info.title}</div>
                      {paper.paper_info.authors && (
                        <div style={styles.paperInfo}>
                          <strong>Authors: </strong>
                          {formatAuthors(paper.paper_info.authors)}
                        </div>
                      )}
  
                      <OverlapBox paper={paper} index={index} />
  
                      <div style={styles.paperInfo}>
                        {/* Display Similarity Score */}
                        <strong>Similarity Score: </strong>
                        <span style={styles.paperMetric}>{paper.similarity_score}</span>
  
                        {/* Wrap the conditional GEM marker in a div for spacing */}
                        <div>
                          {/* Conditionally render "IS A GEM" with rainbow style */}
                          {paper.is_gem && (
                            <span style={styles.paperMetric} className='rainbowSpan'>
                              IS A GEM
                            </span>
                          )}
                          {/* Conditionally render "NOT A GEM" */}
                          {!paper.is_gem && (
                            <span style={styles.paperMetric}>
                              NOT A GEM
                            </span>
                          )}
                        </div>
                      </div>
  
  
                      <div style={styles.paperInfo}>
                        <strong>Shared: </strong>
                        <span style={styles.paperMetric}>{paper.comparison_metrics.shared_reference_count}</span> references,
                        <span style={styles.paperMetric}> {paper.comparison_metrics.shared_citation_count}</span> citations
                      </div>
  
                      <div style={styles.paperAbstract}>
                        <strong>Abstract: </strong> {paper.paper_info.abstract}
                      </div>
                      
                      <SimplePulseButton
                        buttonText={"Enhanced Paper View"}
                        onClick={() => handleShowEnhancedView(paper)}
                        customStyle={{
                          fontSize: '14px',
                          width: '200px',
                          fontWeight: '700',
                          backgroundColor: '#94B4DC',
                          width: '80%',
                        }}
                      />
                    </li>
                  ))}
                </FadeIn>
              </ol>
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default ListResults;