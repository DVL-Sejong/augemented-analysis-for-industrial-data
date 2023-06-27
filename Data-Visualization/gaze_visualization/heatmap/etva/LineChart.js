import React, { useEffect, useRef } from 'react';

function LineChart(props) {
  const { width, height, patchDataList, colorEncoding } = props;
  const svgRef = useRef();
  const d3 = window.d3v4;
  const FEATURE_ordered = ["intensity", "color", "orientation", "curvature", "center_bias", "entropy_rate", "log_spectrum", "HOG"];
  // const COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628", "#f781bf", "#999999"];

  useEffect(() => {
    if ( patchDataList.length === 0)
      return;
    d3.select(svgRef.current).selectAll("*").remove();
    
    var margin = {top: 30, right: 30, bottom: 30, left: 30},
      drawWidth = width - margin.left - margin.right,
      drawHeight = height - margin.top - margin.bottom;
  
    var svg = d3.select(svgRef.current)
    .attr("width", width)
    .attr("height", height)
    .append("svg")
      .attr("width", width)
      .attr("height", height)
      .style("font", "12px sans-serif")
      .style("text-anchor", "middle")
      .append('g').attr("transform",
        "translate(" + margin.left + "," + margin.top + ")");

    let patchData = [];
    for(let i=4; i<FEATURE_ordered.length+4; i++){
      for(let j=0; j<patchDataList.length; j++){
        let patch={
          name: FEATURE_ordered[i-4],
          x: j,
          y: patchDataList[j][i],
          label: patchDataList[j][3]
        };
        patchData.push(patch);
      }
    }
    // console.log(patchData);

    var sumstat = d3.nest()
    .key(function(d){ return d.name; })
    .entries(patchData);
    
    // Add X axis --> it is a date format
    // var xMin = d3.min(patchData, (d=>d.x));
    // var xMax = d3.max(patchData, (d=>d.x));
    // var x = d3.scaleLinear()
    // .domain([xMin, xMax])
    // .range([ 0, drawWidth ]);
    var x = d3.scaleLinear()
    .domain(d3.extent(patchData, function(d){return +d.x}))
    .range([0, drawWidth]);
    svg.append("g")
    .attr("transform", "translate(0," + drawHeight + ")")
    .call(d3.axisBottom(x).ticks(1));

    var yMin = d3.min(patchData, (d=>parseFloat(d.y)));
    var yMax = d3.max(patchData, (d=>+d.y));
    var y = d3.scaleLinear()
    .domain([yMin, yMax])
    .range([ drawHeight, 0 ]);
    svg.append("g")
    .call(d3.axisLeft(y));



    var res = sumstat.map(function(d){return d.key});
    var color = d3.scaleOrdinal()
    .domain(res)
    .range(colorEncoding);
    
    // Draw lines
    svg.selectAll(".line")
    .data(sumstat)
    .enter()
      .append("path")
      .attr("fill", "none")
      .attr("stroke", function(d){return color(d.key)})
      .attr("stroke-width", 1.5)
      .attr("d", function(d){
        return d3.line()
        .x(function(d) { return x(d.x); })
        .y(function(d) { return y(d.y); })
        (d.values)
      });

    
    



  }, [,props.patchDataList, props.colorEncoding]);

  return (
    <>
      {patchDataList.length > 0 &&
        <svg ref={svgRef}>
        </svg>
      }
    </>
  );
}

export default LineChart;
