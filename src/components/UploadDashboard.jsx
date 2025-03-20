import React, { useState } from 'react';
import UploadPDF from './UploadPDF';
import PDFViewer from './PDFViewer';
import UploadText from './UploadText';

function UploadDashboard({ onResultsUpdate, toggleGraphView, pdfFile, onPdfUpload, onClearPdf, results, showGraph }) {
  // No need for previewUrl state since it's managed in UploadPDF

  const styles = {
    dashboardContainer: {
      display: 'flex',
      flexDirection: 'column',
      width: '100%',
      height: '100%',
      backgroundColor: '#f5f7fa',
      fontFamily: 'Arial, sans-serif',
      overflow: 'hidden', // Prevent scrollbars at this level
    },
    contentArea: {
      flex: 1,
      display: 'flex',
      width: '100%',
      height: 'calc(100% - 60px)', // Account for header height
      overflow: 'hidden',
    },
    uploadContainer: {
      flex: 1,
      height: '100%',
      padding: '0', // Remove padding to maximize space
      boxSizing: 'border-box',
      display: 'flex', // Use flex to ensure full height utilization
      overflow: 'hidden', // Hide overflow to prevent unwanted scrollbars
    }
  };

  return (
    <div style={styles.dashboardContainer}>
      <h1 style={{ 
        textAlign: 'center', 
        color: '#333', 
        margin: '15px 0',
        padding: '0 20px', 
        height: '30px',
        fontSize: '24px'
      }}>
        Research Paper Analysis
      </h1>
      
      <div style={styles.contentArea}>
        <div style={styles.uploadContainer}>
          <UploadPDF 
            onResultsUpdate={onResultsUpdate}
            toggleGraphView={toggleGraphView}
            pdfFile={pdfFile}
            onPdfUpload={onPdfUpload}
            onClearPdf={onClearPdf}
            results={results}
            showGraph={showGraph}
          />
        </div>
      </div>
    </div>
  );
}

export default UploadDashboard;