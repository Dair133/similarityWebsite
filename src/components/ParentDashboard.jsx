import React, { useState } from 'react';
import UploadPDF from './UploadPDF';
import ListResults from './ListResults';
import NodeGraph from './NodeGraph';
function ParentDashboard() {
  // State for sharing results between siblings
  const [results, setResults] = useState(null);

  const [showGraph, setShowGraph] = useState(false)
  // Callback for UploadPDF to update results
  const handleResultsUpdate = (newResults) => {
    setResults(newResults);
  };
  const toggleGraphView = () =>{
    console.log('Calling toggle graph')
    setShowGraph(!showGraph)
  }

  // Dashboard container styles
  const styles = {
    container: {
      display: 'flex', // Use flexbox for side-by-side layout
      alignItems: 'flex-start', // Add this line!
      width: '100%',
      minHeight: '100vh',
      gap: '0', // Remove gap between components
      backgroundColor: '#f5f5f5' // Light background for the dashboard
    }
  };

  return (
    <div style={styles.container}>
         {showGraph ? (
           <UploadPDF onResultsUpdate={handleResultsUpdate} />
      ) : (
        <NodeGraph results={results} toggleGraphView={toggleGraphView} />
      )}
        { /* Conditionally render either ListResults or NodeGraph */}
        <ListResults results={results} toggleGraphView={toggleGraphView} setParentResults={handleResultsUpdate}/>
    </div>
  );
}

export default ParentDashboard;



// UploadPDF's Job: The UploadPDF component is likely responsible for handling the file upload, processing the PDF, and then extracting the relevant results data.
// State Ownership: The results state variable lives in ParentDashboard. UploadPDF cannot directly modify results because it's not its state. React enforces this one-way data flow.
// The Callback: handleResultsUpdate is a callback function. ParentDashboard defines the function, and UploadPDF calls the function when it has new results data. When UploadPDF calls handleResultsUpdate(newResults), it's essentially saying to ParentDashboard: "Hey, I've got some new results data (newResults). Please update your results state with this new data."
// setResults Inside handleResultsUpdate: Inside ParentDashboard, the handleResultsUpdate function calls setResults(newResults). This is the crucial step. This updates the results state in ParentDashboard, which triggers a re-render of ParentDashboard and its children (including ListResults). ListResults then receives the updated results data via its results prop.
// In short:  UploadPDF needs to update state that it doesn't own.  The only way to do this is by calling a function provided by the component that does own the state 