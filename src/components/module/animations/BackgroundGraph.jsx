// BackgroundGraph.js
import React, { useEffect, useRef, useState } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import * as d3 from 'd3-force';
import graphClusters from './graphData.json'; // Adjust path as necessary

// Utility: random number between min and max
const randomBetween = (min, max) => Math.random() * (max - min) + min;

const generateGraphData = () => {
  const nodes = [];
  const links = [];
  let globalChainCounter = 0; // For unique chain node IDs

  // Get current dimensions
  const width = window.innerWidth;
  const height = window.innerHeight;
  // Constrain cluster centers to a tight region: from 40% to 60% of window dimensions.
  const xMin = width * 0.4;
  const xMax = width * 0.6;
  const yMin = height * 0.4;
  const yMax = height * 0.6;

  // Loop through each cluster in the JSON file
  graphClusters.clusters.forEach((cluster, index) => {
    const clusterGroup = index + 1;
    // Random cluster center within the bounding box.
    const centerX = randomBetween(xMin, xMax);
    const centerY = randomBetween(yMin, yMax);
    
    // Create the seed node from the JSON.
    const seedId = cluster.seed;
    nodes.push({
      id: seedId,
      group: clusterGroup,
      isSeed: true,
      x: centerX,
      y: centerY,
      clusterCenterX: centerX,
      clusterCenterY: centerY,
      topic: cluster.topic,
      noisePhase: randomBetween(0, 2 * Math.PI)
    });
    
    // For each node in the cluster from the JSON, create a node with a random offset.
    const clusterNodeIds = [];
    cluster.nodes.forEach((nodeTitle) => {
      const nodeId = `${cluster.topic} - ${nodeTitle}`;
      clusterNodeIds.push(nodeId);
      // Increase offset range to spread nodes inside cluster more
      const offsetX = randomBetween(-200, 200);
      const offsetY = randomBetween(-200, 200);
      nodes.push({
        id: nodeId,
        group: clusterGroup,
        x: centerX + offsetX,
        y: centerY + offsetY,
        clusterCenterX: centerX,
        clusterCenterY: centerY,
        topic: cluster.topic,
        noisePhase: randomBetween(0, 2 * Math.PI)
      });
      // Link the seed node to this node.
      links.push({
        source: seedId,
        target: nodeId
      });
    });
    
    // Optionally, add a random extra intra-cluster link between two non-seed nodes.
    if (cluster.nodes.length > 1 && Math.random() < 0.5) {
      const n = cluster.nodes.length;
      const i1 = Math.floor(randomBetween(0, n));
      let i2 = Math.floor(randomBetween(0, n));
      while (i2 === i1) {
        i2 = Math.floor(randomBetween(0, n));
      }
      const nodeId1 = `${cluster.topic} - ${cluster.nodes[i1]}`;
      const nodeId2 = `${cluster.topic} - ${cluster.nodes[i2]}`;
      links.push({
        source: nodeId1,
        target: nodeId2
      });
    }
    
    // Additionally, add a random chain from this cluster.
    const numChains = Math.floor(randomBetween(0, 3)); // 0 to 2 chains
    for (let i = 0; i < numChains; i++) {
      const allClusterNodes = [seedId, ...clusterNodeIds];
      const sourceIndex = Math.floor(randomBetween(0, allClusterNodes.length));
      let currentSourceId = allClusterNodes[sourceIndex];
      const chainLength = Math.floor(randomBetween(1, 4)); // chain length: 1 to 3 nodes
      for (let j = 0; j < chainLength; j++) {
        const chainNodeId = `${cluster.topic} - chain${globalChainCounter++}`;
        const sourceNode = nodes.find(n => n.id === currentSourceId);
        // For chain nodes, use a moderate offset.
        const offsetX = randomBetween(-50, 50);
        const offsetY = randomBetween(-50, 50);
        nodes.push({
          id: chainNodeId,
          group: clusterGroup,
          x: sourceNode.x + offsetX,
          y: sourceNode.y + offsetY,
          clusterCenterX: centerX,
          clusterCenterY: centerY,
          topic: cluster.topic,
          noisePhase: randomBetween(0, 2 * Math.PI)
        });
        links.push({
          source: currentSourceId,
          target: chainNodeId
        });
        currentSourceId = chainNodeId;
      }
    }
  });

  return { nodes, links };
};

const BackgroundGraph = () => {
  const fgRef = useRef();
  const [graphData, setGraphData] = useState(generateGraphData());
  const [dimensions, setDimensions] = useState({
    width: window.innerWidth,
    height: window.innerHeight
  });

  // Update graph and dimensions on window resize.
  useEffect(() => {
    const handleResize = () => {
      setDimensions({
        width: window.innerWidth,
        height: window.innerHeight
      });
      setGraphData(generateGraphData());
      // Optionally, you can also reheat the simulation:
      if (fgRef.current) {
        fgRef.current.d3ReheatSimulation();
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Use a low global repulsion so clusters don't fly apart.
  useEffect(() => {
    if (fgRef.current) {
      fgRef.current.d3Force("charge").strength(-2000);
      // Add cluster forces to pull nodes toward their cluster center.
      fgRef.current.d3Force("clusterX", d3.forceX(d => d.clusterCenterX).strength(0.5));
      fgRef.current.d3Force("clusterY", d3.forceY(d => d.clusterCenterY).strength(0.5));
    }
  }, []);

// Use an effect to force re-rendering on window resize so that our noise animation is visible.
useEffect(() => {
    const handleResize = () => {
        if (fgRef.current) {
            fgRef.current.d3ReheatSimulation(0.05); // reheat simulation by a small amount
        }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
}, []);

  // Custom node renderer with noise animation.
  const nodeCanvasObject = (node, ctx, globalScale) => {
    const t = Date.now() / 1000; // time in seconds
    const amplitude = 5; // amplitude for noise effect
    const noiseX = Math.sin(t + node.noisePhase) * amplitude;
    const noiseY = Math.cos(t + node.noisePhase) * amplitude;
    
    const x = node.x + noiseX;
    const y = node.y + noiseY;
    
    const label = node.id;
    const fontSize = 10 / globalScale;
    ctx.font = `${fontSize}px Sans-Serif`;
    
    // Get node color based on group.
    const colors = [
      "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
      "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
      "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5"
    ];
    const color = colors[(node.group - 1) % colors.length];
    
    const radius = node.isSeed ? 8 : 5;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, 2 * Math.PI);
    ctx.fill();

    // Draw node label.
    ctx.fillStyle = 'black';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText(label, x, y + radius + 2);
  };

  return (
    <div className="background-graph">
      <ForceGraph2D
        ref={fgRef}
        graphData={graphData}
        width={dimensions.width}
        height={dimensions.height}
        nodeCanvasObject={nodeCanvasObject}
        backgroundColor="transparent"
        enableZoomPanInteraction={false}
      />
    </div>
  );
};

export default BackgroundGraph;
