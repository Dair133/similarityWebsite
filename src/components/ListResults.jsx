import React, { useState } from 'react';
import FadeIn from 'react-fade-in';
function ListResults({ results, toggleGraphView, setParentResults }) {
  console.log(results)
  // Using { results } is the equivalent of doing
  // const results = props.results;
  // Here we are using Javascript object destructuring


  const [localResults, setLocalResults] = useState(null);




  // Inline styling for the component
  const styles = {
    container: {
      width: '50%',
      height:'95vh',
      backgroundColor: 'yellow',
      padding: '2%',
      boxSizing: 'border-box',
      overflow: 'auto',
    },
    title: {
      fontSize: '24px',
      color: 'black',
      marginBottom: '20px',
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
      border: '2px dashed #ccc',
      borderRadius: '4px',
      cursor: 'pointer',
    },
    fileInfo: {
      marginTop: '10px',
      color: '#666',
      fontSize: '14px',
    },
    results: {
      marginTop: '20px',
      width: '100%',
      textAlign: 'left',
    },
    button: { // Added style for the button
      backgroundColor: '#4CAF50', // Green
      border: 'none',
      color: 'white',
      padding: '10px 20px',
      textAlign: 'center',
      textDecoration: 'none',
      display: 'inline-block',
      fontSize: '16px',
      margin: '4px 2px',
      cursor: 'pointer',
      borderRadius: '4px',
    },
    NodeGraphContainer: {
      visibility:'hidden'
    }
  };

  // A helper to render the Semantic Scholar info (or any info object)
  const renderSemanticScholarInfo = (info) => {
    console.log(info)
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


  const generateFakeResults = () => {
    const fakePapers = Array.from({ length: 10 }, (_, i) => ({
      title: `Fake Paper Title ${i + 1}`,
      id: `fake-id-${i + 1}`,
      similarity_score: (1 - i * 0.1).toFixed(2), // Decreasing similarity
      source_info: [{ search_term: "Fake Search", search_type: "Fake Type" }],
      paper_info: { abstract: `This is a fake abstract for paper ${i + 1}. It's short and sweet.` },
    }));

    const fakeResultsData = {
      seed_paper: {
        paper_info: {
          abstract: "This is a fake abstract for the seed paper."
        }
      },
      abstract_info: "Fake abstract information.",
      similarity_results: fakePapers
    };

    setLocalResults(fakeResultsData);

    // The below line should be removed as normally this componenet will never set result data
    // Its only so I can test the node graph
    setParentResults(fakeResultsData)
  };


  const displayResults = localResults || results;


  return (
    <div style={styles.container}>
      <h2 style={styles.title}>List Of Results</h2>
      <button onClick={toggleGraphView}>Swap</button>
       <button style={styles.button} onClick={generateFakeResults}>
        Fake Results
      </button>
      {displayResults && (
        <div style={styles.results}>
          <h3>Semantic Scholar Info:</h3>
          <strong>Title: </strong>{displayResults.seed_paper.paper_info.title}
          {renderSemanticScholarInfo(displayResults.seed_paper.paper_info.abstract)}
          <h3>Semantic Scholar Abstract Info:</h3>
          {renderSemanticScholarInfo(displayResults.abstract_info)}
          <h3>Similar Papers</h3>
          <ul>
           <FadeIn>
            {displayResults.similarity_results.map((paper, index) => (
              <li key={index}>
                <strong>Title:</strong> {paper.paper_info.title} <br />
                <strong>Similarity Score:</strong>{' '}
                {paper.similarity_score}
                <br />
                <strong>Source Info:</strong>{' '}
                {paper.source_info[0].search_term}
                {paper.source_info[0].search_type}
                <br />
                <strong>Shared Reference Count:</strong>
                {' '}{paper.comparison_metrics.shared_reference_count}
                <br />
                <strong>Shared Citation Count:</strong>
                {' '}{paper.comparison_metrics.shared_citation_count}
                <br />
                <strong>Shared Author Count:</strong>
                {' '}{paper.comparison_metrics.shared_author_count}
                <p>
                  <strong>Abstract:</strong> {paper.paper_info.abstract}
                </p>
              </li>
            ))}
           </FadeIn>
          </ul>
        </div>
      )}
    </div>
  );
}

export default ListResults;