import React, { useEffect, useRef } from 'react';

function BarChart(props) {
  const { width, height, overviewSytle, countDataList } = props;
  const svgRef = useRef();
  const d3 = window.d3v4;

  useEffect(() => {
    if ( countDataList.length === 0)
      return;
    d3.select(svgRef.current).selectAll("*").remove();
    
    var margin = {top: 10, right: 30, bottom: 30, left: 30},
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

    let countData = [];
    let styleIdx = 1;
    if(overviewSytle == "scanpath"){
      styleIdx = 1;
    }else if(overviewSytle == "patch_total"){
      styleIdx = 2;
    }else if(overviewSytle == "patch_on"){
      styleIdx = 3;
    }else if(overviewSytle == "patch_out"){
      styleIdx = 4;
    }else{
      console.log("Unavailable overveiew option selected: "+overviewSytle);
    }
    for(let i=0; i<countDataList.length; i++){
      let _name = "s";
      if(i<10){
        _name = _name+"0"+String(i);
      }else{
        _name = _name+String(i);
      }
      let _d ={
        name: _name,
        value: countDataList[i][styleIdx]
      }
      countData.push(_d);
    }
    console.log(countData);

    // var xMin = d3.min(countData, (d=>d.value));
    var xMax = d3.max(countData, (d=>d.value));

    var x = d3.scaleLinear()
    .domain([0, xMax])
    .range([0, drawWidth]);
    svg.append("g")
    .attr("transform", "translate(0," + drawHeight + ")")
    .call(d3.axisBottom(x))
    .selectAll("text")
      .attr("transform", "translate(-10,0)rotate(-45)")
      .style("text-anchor", "end");

    var y = d3.scaleBand()
    .range([ 0, drawHeight ])
    .domain(countData.map(function(d) { return d.name; }))
    .padding(.1);
    svg.append("g")
    .call(d3.axisLeft(y))

    svg.selectAll("rect_"+overviewSytle)
    .data(countData)
    .enter()
    .append("rect")
    .attr("x", 0)
    .attr("y", function(d){ return y(d.name); })
    .attr("width", function(d){ return x(d.value) })
    .attr("height", y.bandwidth())
    .attr("fill", "gray")
    
    
  }, [,props.countDataList]);

  return (
    <>
      {countDataList.length > 0 &&
        <svg ref={svgRef}>
        </svg>
      }
    </>
  );
}

export default BarChart;
