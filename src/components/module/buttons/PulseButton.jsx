import React from 'react';
import './PulseButton.css';

function PulseButton({ onClick, buttonText, customStyle = {}, disabled = false }) {
  return (
    <button 
      className="pulse-button"
      onClick={onClick}
      style={customStyle}
      disabled={disabled}
    >
      {buttonText}
    </button>
  );
}

export default PulseButton;
