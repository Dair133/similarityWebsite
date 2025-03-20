import React from 'react';
import './PulseButton.css';

function PulseButton({ onClick, buttonText, customStyle = {} }) {
  return (
    <button 
      className="pulse-button"
      onClick={onClick}
      style={customStyle}
    >
      {buttonText}
    </button>
  );
}

export default PulseButton;