import React, { useState } from 'react';
import NodeGraph from '../NodeGraph';
import ListResults from '../ListResults';
import UploadDashboard from '../UploadDashboard';

function ParentDashboard() {
  const [results, setResults] = useState(null);
  const [pdfFile, setPdfFile] = useState(null);
  const [showGraph, setShowGraph] = useState(true);
  const [showPdfDashboard, setShowPdfDashboard] = useState(false);
  const [hoveredNodeIndex, setHoveredNodeIndex] = useState(null);
  
  // Styles for the component
  const styles = {
    mainContainer: {
      display: 'flex',
      flexDirection: 'row',
      height: '100vh',
      width: '100%',
      backgroundColor: '#0A1929',
      overflow: 'hidden',
    },
    contentContainer: {
      display: 'flex',
      flex: 1,
      flexDirection: 'row',
    },
    graphContainer: {
      width: '75%',
      height: '100%',
      position: 'relative',
    },
    pdfDashboardContainer: {
      width: '75%',
      height: '100%',
      position: 'relative',
      backgroundColor: '#0F2741',
    },
    listContainer: {
      width: '25%',
      height: '100%',
      position: 'relative',
      boxShadow: '-4px 0 10px rgba(0, 0, 0, 0.2)',
    },
    noContentMessage: {
      width: '75%',
      height: '100%',
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      backgroundColor: '#0F2741',
      color: '#F7F3E9',
      fontSize: '18px',
      fontFamily: '"Montserrat", sans-serif',
    }
  };
  
  // Function to toggle between PDF dashboard and node graph view
  const toggleGraphView = () => {
    setShowGraph(prevState => !prevState);
    // If switching to graph view, ensure PDF dashboard is hidden
    if (!showGraph) {
      setShowPdfDashboard(false);
    }
  };

  // Function to toggle PDF dashboard visibility
  const togglePdfDashboard = () => {
    setShowPdfDashboard(prevState => !prevState);
    // If showing PDF dashboard, hide graph view
    if (!showPdfDashboard) {
      setShowGraph(false);
    }
  };
  
  // Function to handle PDF upload
  const handlePdfUpload = (file) => {
    setPdfFile(file);
    // Automatically show PDF dashboard when a file is uploaded
    setShowPdfDashboard(true);
    setShowGraph(false);
  };
  
  // Function to handle PDF clearing
  const handleClearPdf = () => {
    setPdfFile(null);
    // If we're in PDF dashboard view, switch back to graph or message
    if (showPdfDashboard) {
      setShowPdfDashboard(false);
      setShowGraph(true);
    }
  };
  
  // Handler for when a node is hovered in the graph
  const handleNodeHover = (paperIndex) => {
    setHoveredNodeIndex(paperIndex);
  };
  
  // Handler for when a node is clicked in the graph
  const handleNodeClick = (paper) => {
    // Send a message to the ListResults component to show enhanced view for this paper
    if (paper) {
      // We need a reference to the ListResults component to call its methods
      // For now, we'll use a custom event to communicate
      const event = new CustomEvent('showEnhancedView', { detail: { paper } });
      document.dispatchEvent(event);
    }
  };
  
  // Handler for updating results
  const handleSetResults = (newResults) => {
    setResults(newResults);
  };

  // Determine what to display in the main content area
  const renderMainContent = () => {
    if (!showGraph) {
      // Always show the UploadDashboard when not showing the graph
      return (
        <div style={styles.pdfDashboardContainer}>
          <UploadDashboard
            pdfFile={pdfFile} 
            onResultsUpdate={handleSetResults}
            toggleGraphView={toggleGraphView}
            onPdfUpload={handlePdfUpload}
            onClearPdf={handleClearPdf}
            results={results}
            showGraph={showGraph}
          />
        </div>
      );
    } else if (results) {
      // Show the graph when showGraph is true and we have results
      return (
        <div style={styles.graphContainer}>
          <NodeGraph 
            results={results}
            toggleGraphView={toggleGraphView}
            onNodeHover={handleNodeHover}
            onNodeClick={handleNodeClick}
          />
        </div>
      );
    } else {
      // Fall back to the message when there are no results to show in graph view
      return (
        <div style={styles.noContentMessage}>
          {results ? "Switch to Graph view to visualize paper relationships" : 
           "You are currently in Node View. No results to display. Upload a PDF to begin analysis."}
        </div>
      );
    }
  };

  return (
    <div style={styles.mainContainer}>
      <div style={styles.contentContainer}>
        {/* Left side - NodeGraph, PDFDashboard or placeholder (75%) */}
        {renderMainContent()}
        
        {/* Right side - ListResults (25%) */}
        <div style={styles.listContainer}>
          <ListResults 
            results={results}
            toggleGraphView={toggleGraphView}
            togglePdfDashboard={togglePdfDashboard}
            setParentResults={handleSetResults}
            showGraph={showGraph}
            showPdfDashboard={showPdfDashboard}
            onPdfUpload={handlePdfUpload}
            onClearPdf={handleClearPdf}
            hoveredNodeIndex={hoveredNodeIndex}
            pdfFile={pdfFile}
          />
        </div>
      </div>
    </div>
  );
}

export default ParentDashboard;