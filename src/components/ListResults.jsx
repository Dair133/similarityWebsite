import React from 'react';




function ListResults({ results }){

// Inline styling for the component
const styles = {
  container: {
    width: '50%', // Set to 60% of parent width
    backgroundColor: 'white',
    padding: '2%',
    boxSizing: 'border-box',
    // Remove display: inline-block and float properties
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
  };

  // A helper to render the Semantic Scholar info (or any info object)
  const renderSemanticScholarInfo = (info) => {
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

    return(
        <div style={styles.container}>
            <h2 style={styles.title}>List Of Results</h2>
            {results && (
            <div style={styles.results}>
              <h3>Semantic Scholar Info:</h3>
              {renderSemanticScholarInfo(results.seed_paper.paper_info.abstract)}
              <h3>Semantic Scholar Abstract Info:</h3>
              {renderSemanticScholarInfo(results.abstract_info)}
              <h3>Similar Papers</h3>
              <ul>
                {results.similarity_results.map((paper, index) => (
                  <li key={index}>
                    <strong>Title:</strong> {paper.title} <br />
                    <strong>Paper ID:</strong> {paper.id} -{' '}
                    <strong>Similarity Score:</strong>{' '}
                    {paper.similarity_score}
                    <br />
                    <strong>Source Info:</strong>{' '}
                    {paper.source_info[0].search_term}
                    {paper.source_info[0].search_type}
                    <p>
                      <strong>Abstract:</strong> {paper.paper_info.abstract}
                    </p>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
    )
        















}

export default ListResults