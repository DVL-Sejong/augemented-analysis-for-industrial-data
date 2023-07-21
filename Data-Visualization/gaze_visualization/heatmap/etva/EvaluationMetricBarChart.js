import React, { useEffect, useRef } from 'react';

function EvaluationMetricBarChart(props) {
  const { width, height, evaluationMetrics } = props;
  const svgRef = useRef();
  const d3 = window.d3v4;
  const evaluation_metrics = ["IG", "AUC", "sAUC", "NSS", "CC", "KLDiv", "SIM"];
  useEffect(() => {
    if ( evaluationMetrics.length === 0)
      return;
    d3.select(svgRef.current).selectAll("*").remove();
    
    let scores = [];
    for(let i=0; i<evaluationMetrics.length; i++){
      let s = {
        metric: evaluation_metrics[i],
        value: evaluationMetrics[i]
      };
      scores.push(s);
    }

    var margin = {top: 10, right: 15, bottom: 50, left: 30},
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


    var x = d3.scaleBand()
      .range([ 0, drawWidth ])
      .domain(scores.map(function(d) { return d.metric; }))
      .padding(0.2);
    svg.append("g")
      .attr("transform", "translate(0," + drawHeight + ")")
      .call(d3.axisBottom(x))
      .selectAll("text")
        .attr("transform", "translate(-10,0)rotate(-45)")
        .style("text-anchor", "end");

    var yMax = d3.max(scores, (d=>d.value));
    var y = d3.scaleLinear()
    .domain([0, yMax])
    .range([ drawHeight, 0]);
    svg.append("g")
    .call(d3.axisLeft(y));

    svg.selectAll("rect_em")
    .data(scores)
    .enter()
    .append("rect")
    .attr("x", function(d){ return x(d.metric)})
    .attr("y", function(d){ return y(d.value); })
    .attr("width", x.bandwidth())
    .attr("height", function(d) { return drawHeight-y(d.value) })
    .attr("fill", function(d){
      if(d.value == -999){
        return "none";
      }else{
        return "gray";
      }
    });

    svg.selectAll("text_score")
    .data(scores)
    .enter()
    .append("text")
    .attr("x", function(d){
      return x(d.metric)+x.bandwidth()/2;
    })
    .attr("y", function(d){
      return y(d.value)-2;
    })
    .text(function(d){
      return parseFloat(d.value).toFixed(4);
    })
    .attr("font-family", "Roboto, sans-serif")
    .style("font-size", "10px");
    
  }, [,props.evaluationMetrics]);

  return (
    <>
      {evaluationMetrics.length > 0 &&
        <svg ref={svgRef}>
        </svg>
      }
    </>
  );
}

export default EvaluationMetricBarChart;
