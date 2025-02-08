import React, { useState } from 'react';
import UploadPDF from './UploadPDF';
import ListResults from './ListResults';

function ParentDashboard() {
  // State for sharing results between siblings
  const [results, setResults] = useState(null);

  // Callback for UploadPDF to update results
  const handleResultsUpdate = (newResults) => {
    setResults(newResults);
  };

  // Dashboard container styles
  const styles = {
    container: {
      display: 'flex', // Use flexbox for side-by-side layout
      width: '100%',
      minHeight: '100vh',
      gap: '0', // Remove gap between components
      backgroundColor: '#f5f5f5' // Light background for the dashboard
    }
  };

  return (
    <div style={styles.container}>
      <UploadPDF onResultsUpdate={handleResultsUpdate} />
      <ListResults results={results} />
    </div>
  );
}

export default ParentDashboard;