import React, { useState } from 'react';
import UploadPDF from './UploadPDF';
import ListResults from './ListResults';
import NodeGraph from './NodeGraph';

function ParentDashboard() {
  const [results, setResults] = useState(null);
  const [pdfFile, setPdfFile] = useState(null);
  const [showGraph, setShowGraph] = useState(false);

  const handleResultsUpdate = (newResults) => {
    setResults(newResults);
  };

  const toggleGraphView = () => {
    setShowGraph((prev) => !prev);
  };

  const handlePdfUpload = (file) => {
    setPdfFile(file);
  };

  const handleClearPdf = () => {
    setPdfFile(null);
    setResults(null);
  };

  const styles = {
    container: {
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      width: '100%',
      minHeight: '100vh',
      backgroundColor: '#f5f5f5',
    },
    contentContainer: {
      display: 'flex',
      flexDirection: 'row',
      width: '100%',
      gap: '0',
    },
  };

  return (
    <div style={styles.container}>
      <div style={styles.contentContainer}>
        {showGraph ? (
          <NodeGraph results={results} toggleGraphView={toggleGraphView} />
        ) : (
          <UploadPDF
            onResultsUpdate={handleResultsUpdate}
            pdfFile={pdfFile}
            onPdfUpload={handlePdfUpload}
            onClearPdf={handleClearPdf}
            results={results}
          />
        )}
        <ListResults
          results={results}
          toggleGraphView={toggleGraphView}
          setParentResults={handleResultsUpdate}
          showGraph={showGraph} // Pass showGraph state to ListResults
        />
      </div>
    </div>
  );
}

export default ParentDashboard;