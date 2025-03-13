import React, { useState } from 'react';
import FadeIn from 'react-fade-in';

function ListResults({ results, toggleGraphView, setParentResults }) {
  console.log(results);
  // Using { results } is the equivalent of doing
  // const results = props.results;
  // Here we are using Javascript object destructuring

  const [localResults, setLocalResults] = useState(null);
  const [tooltipVisible, setTooltipVisible] = useState({});

  // Inline styling for the component
  const styles = {
    container: {
      width: '25%',
      height: '95vh',
      backgroundColor: '#10253E', // Deep Navy instead of black
      color: '#F7F3E9', // Ivory instead of white
      padding: '2%',
      boxSizing: 'border-box',
      overflow: 'auto',
      fontFamily: '"Source Sans Pro", sans-serif',
    },
    title: {
      fontSize: '24px',
      color: '#F7F3E9', // Ivory
      marginBottom: '20px',
      fontFamily: '"Montserrat", sans-serif',
      fontWeight: '600',
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
      border: '2px dashed #81A4CD', // Light blue for border
      borderRadius: '4px',
      cursor: 'pointer',
      backgroundColor: '#14304D', // Navy (lighter shade)
      color: '#F7F3E9', // Ivory
      fontFamily: '"Source Sans Pro", sans-serif',
    },
    fileInfo: {
      marginTop: '10px',
      color: '#EEE8D9', // Soft Ivory
      fontSize: '14px',
      fontFamily: '"Source Sans Pro", sans-serif',
    },
    results: {
      marginTop: '20px',
      width: '100%',
      textAlign: 'left',
    },
    button: {
      backgroundColor: '#3E7CB9', // Steel Blue instead of green
      border: 'none',
      color: '#F7F3E9', // Ivory
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
    },
    NodeGraphContainer: {
      visibility: 'hidden'
    },
    // New styles for better typography
    sectionHeading: {
      fontFamily: '"Montserrat", sans-serif',
      fontSize: '20px',
      fontWeight: '600',
      color: '#F7F3E9', // Ivory
      marginTop: '25px',
      marginBottom: '15px',
    },
    listItem: {
      backgroundColor: '#14304D', // Navy (lighter shade)
      padding: '15px',
      marginBottom: '15px',
      borderRadius: '6px',
      borderLeft: '3px solid #3E7CB9', // Steel Blue accent
    },
    paperTitle: {
      fontFamily: '"Montserrat", sans-serif',
      fontSize: '18px',
      fontWeight: '600',
      color: '#EEE8D9', // Soft Ivory for contrast
      marginBottom: '10px',
    },
    paperInfo: {
      fontFamily: '"Source Sans Pro", sans-serif',
      fontSize: '14px',
      lineHeight: '1.6',
      marginBottom: '8px',
      color: '#F7F3E9', // Ivory
    },
    paperMetric: {
      color: '#81A4CD', // Light Blue for metrics
      fontWeight: '600',
    },
    paperAbstract: {
      fontFamily: '"Source Sans Pro", sans-serif',
      fontSize: '15px',
      lineHeight: '1.6',
      color: '#F7F3E9', // Ivory
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
    // Replace your current tooltip style with this
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
      pointerEvents: 'none', // This prevents the tooltip from capturing mouse events
    },
    overlapContent: {
      padding: '12px',
    },
    overlapTag: {
      display: 'inline-block',
      padding: '6px 12px',
      backgroundColor: '#3E7CB9',
      color: '#F7F3E9',
      borderRadius: '4px',
      fontSize: '13px',
      fontFamily: '"Source Sans Pro", sans-serif',
      margin: '2px 0',
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
  };

  // A helper to render the Semantic Scholar info (or any info object)
  const renderSemanticScholarInfo = (info) => {
    console.log(info);
    if (!info) return null;
    if (typeof info === 'string') {
      return <div>{info}</div>;
    }
    return (
      <div>
        {Object.entries(info).map(([key, value]) => (
          <div key={key}>
            <strong>{key}:</strong> {JSON.stringify(value, null, 2)}
          </div>
        ))}
      </div>
    );
  };

  // Helper function to safely format authors
  const formatAuthors = (authors) => {
    if (!authors) return '';
    if (typeof authors === 'string') return authors;
    if (Array.isArray(authors)) return authors.join(', ');
    return JSON.stringify(authors);
  };

  const generateFakeResults = () => {
    // Create some fake author names
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

    const fakePapers = Array.from({ length: 10 }, (_, i) => {
      // Generate random authors for each paper
      const paperAuthorCount = Math.floor(Math.random() * 3) + 1;
      const paperAuthors = [];
      for (let j = 0; j < paperAuthorCount; j++) {
        const randomIndex = Math.floor(Math.random() * authorPool.length);
        if (!paperAuthors.includes(authorPool[randomIndex])) {
          paperAuthors.push(authorPool[randomIndex]);
        }
      }

      // Generate shared authors with seed paper (for odd indices, to vary the data)
      const sharedAuthors = i % 2 === 1 ? [authorPool[0], authorPool[1]].slice(0, i % 3) : [];

      return {
        title: `Fake Paper Title ${i + 1}`,
        id: `fake-id-${i + 1}`,
        similarity_score: (1 - i * 0.1).toFixed(2), // Decreasing similarity
        source_info: {
          search_term: i % 2 === 0 ? "minimal perturbation computation" : ["adversarial example generation", "neural network vulnerability"],
          search_type: i % 2 === 0 ? "core_methodology" : "conceptual_angles"
        },
        paper_info: {
          title: `Fake Paper Title ${i + 1}`,
          abstract: `This is a fake abstract for paper ${i + 1}. It's short and sweet.`,
          authors: paperAuthors
        },
        comparison_metrics: {
          shared_reference_count: Math.floor(Math.random() * 10),
          shared_citation_count: Math.floor(Math.random() * 10),
          shared_author_count: sharedAuthors.length,
          shared_authors: sharedAuthors,
          shared_references: i % 2 === 0 ?
            ["Smith et al. (2019)", "Johnson & Lee (2020)", "Williams (2018)"] :
            ["Brown et al. (2021)", "Davis (2019)", "Miller & Taylor (2020)",
              "Wilson (2017)", "Moore & Zhang (2022)", "Additional paper 1", "Additional paper 2"]
        }
      };
    });

    const fakeResultsData = {
      seed_paper: {
        paper_info: {
          title: "Seed Paper Title",
          abstract: "This is a fake abstract for the seed paper.",
          authors: [authorPool[0], authorPool[1], authorPool[2]]
        }
      },
      abstract_info: "Fake abstract information.",
      similarity_results: fakePapers,
      // Add this to match your expected structure
      test: {
        compared_papers: fakePapers
      }
    };

    setLocalResults(fakeResultsData);
    setParentResults(fakeResultsData);
  };

  const displayResults = localResults || results;

  // Helper function to format source type text
  const formatSourceType = (sourceType) => {
    if (!sourceType) return '';

    // Replace underscores with spaces and capitalize each word
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

    // Only show divider if both methodology and shared authors exist
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
          {/* Methodology section */}
          {hasMethodology && (
            <>
              <div style={styles.sectionLabel}>Shared Methodology:</div>
              <div style={styles.overlapItem}>
                <span style={styles.overlapTag}>
                  {Array.isArray(paper.source_info.search_term)
                    ? paper.source_info.search_term[0] // Just take the first one if it's an array
                    : paper.source_info.search_term}
                </span>
                <span style={styles.overlapTypeLabel}>
                  ({formatSourceType(paper.source_info.search_type)})
                </span>
              </div>
            </>
          )}

          {/* Divider - only show if both methodology and authors exist */}
          {hasMethodology && hasSharedAuthors && <div style={styles.divider}></div>}

          {/* Shared Authors section */}
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
          {/* Shared References section */}
          {paper.comparison_metrics &&
            paper.comparison_metrics.shared_references &&
            paper.comparison_metrics.shared_references.length > 0 && (
              <>
                {/* Add divider if there's methodology or authors */}
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








          {/* Show message if no overlap */}
          {!hasMethodology && !hasSharedAuthors && (
            <div style={{ color: '#81A4CD', fontStyle: 'italic', fontSize: '13px' }}>
              No direct methodological or authorship overlap detected
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div style={styles.container}>
      <h2 style={styles.title}>List Of Results</h2>
      <button style={styles.button} onClick={toggleGraphView}>
        View Graph
      </button>
      <button style={styles.button} onClick={generateFakeResults}>
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
                <li key={index} style={styles.listItem}>
                  <div style={styles.paperTitle}>{paper.paper_info.title}</div>

                  {paper.paper_info.authors && (
                    <div style={styles.paperInfo}>
                      <strong>Authors: </strong>
                      {formatAuthors(paper.paper_info.authors)}
                    </div>
                  )}

                  {/* Add the OverlapBox component here */}
                  <OverlapBox paper={paper} index={index} />

                  <div style={styles.paperInfo}>
                    <strong>Similarity Score: </strong>
                    <span style={styles.paperMetric}>{paper.similarity_score}</span>
                  </div>

                  <div style={styles.paperInfo}>
                    <strong>Shared: </strong>
                    <span style={styles.paperMetric}>{paper.comparison_metrics.shared_reference_count}</span> references,
                    <span style={styles.paperMetric}> {paper.comparison_metrics.shared_citation_count}</span> citations
                  </div>

                  <div style={styles.paperAbstract}>
                    <strong>Abstract: </strong> {paper.paper_info.abstract}
                  </div>
                </li>
              ))}
            </FadeIn>
          </ol>
        </div>
      )}
    </div>
  );
}

export default ListResults;