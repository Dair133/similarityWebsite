import React, { useState } from 'react';
import UploadPDF from './UploadPDF';
import ListResults from './ListResults';
import NodeGraph from './NodeGraph';
import UploadDashboard from './UploadDashboard';

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
      width: '100%',
      height: '100vh',
      backgroundColor: '#f5f5f5',
      overflow: 'hidden',
      margin: 0,
      padding: 0,
    },
    contentContainer: {
      display: 'flex',
      width: '100%',
      height: '100%',
      margin: 0,
      padding: 0,
      overflow: 'hidden',
      // The key fix: position relative to make absolute positioning work correctly
      position: 'relative',
    },
    mainContentArea: {
      width: '75%', // Match the width from ListResults component (it uses 25%)
      height: '100%',
      display: 'flex',
      overflow: 'hidden',
    }
    // Removed rightSidebar style as it conflicts with ListResults' own styling
  };

  return (
    <div style={styles.container}>
      <div style={styles.contentContainer}>
        <div style={styles.mainContentArea}>
          {showGraph ? (
            <NodeGraph 
              results={results} 
              toggleGraphView={toggleGraphView} 
            />
          ) : (
            <UploadDashboard
              onResultsUpdate={handleResultsUpdate}
              toggleGraphView={toggleGraphView}
              pdfFile={pdfFile}
              onPdfUpload={handlePdfUpload}
              onClearPdf={handleClearPdf}
              results={results}
              showGraph={showGraph}
            />
          )}
        </div>
        
        {/* ListResults has its own styling and doesn't need a container */}
        <ListResults
          results={results}
          toggleGraphView={toggleGraphView}
          setParentResults={handleResultsUpdate}
          showGraph={showGraph}
        />
      </div>
    </div>
  );
}

export default ParentDashboard;