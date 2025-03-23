// LandingPage.jsx
import React from 'react';
import './LandingPage.css';
import BackgroundGraph from '../module/animations/BackgroundGraph';

const LandingPage = ({ onFindPapers }) => {
  return (
    <div className="landing-container">
      <BackgroundGraph />
      <div className="landing-content">
        <h1 className="landing-heading">Scientific Paper <span className="rainbowSpan">Gem</span> Finder</h1>
        <button className="landing-button" onClick={onFindPapers}>
          Find papers Now
        </button>
      </div>
    </div>
  );
};

export default LandingPage;
