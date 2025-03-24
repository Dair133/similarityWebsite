// Spinner.js
import React from 'react';

const Spinner = ({ size = 16, color = "#3498db" }) => {
  const spinnerStyle = {
    border: `${size * 0.15}px solid #f3f3f3`,
    borderTop: `${size * 0.15}px solid ${color}`,
    borderRadius: "50%",
    width: size,
    height: size,
    animation: "spin 1s linear infinite",
    display: "block",      // make it a block element
    margin: "0 auto"       // center horizontally
  };

  return <div style={spinnerStyle}></div>;
};

export default Spinner;
