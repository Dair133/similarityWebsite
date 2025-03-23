// App.js
import React, { useState } from 'react';
import LandingPage from './components/Pages/LandingPage';
import ParentDashboard from './components/Pages/ParentDashboard';

function App() {
  const [showLanding, setShowLanding] = useState(true);

  const handleFindPapers = () => {
    setShowLanding(false);
  };

  return (
    <>
      {showLanding ? (
        <LandingPage onFindPapers={handleFindPapers} />
      ) : (
        <ParentDashboard />
      )}
    </>
  );
}

export default App;
