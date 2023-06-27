import axios from 'axios';
import React, { useEffect, useRef } from 'react';

function HorizontalBarChart(props) {
  const { width, height, patchOutsideData, colorEncoding, selectedObserver, observerSelectFunction } = props;
  const svgRef = useRef();
  const d3 = window.d3v4;
  const colores_g = ["#109618", "#ff9900", "#990099", "#dd4477", "#0099c6", "#66aa00", "#b82e2e", "#316395", "#994499", "#22aa99", "#aaaa11", "#6633cc", "#e67300", "#8b0707", "#651067", "#329262", "#5574a6", "#3b3eac", "#3366cc", "#dc3912"];
  
  useEffect(() => {
    if ( patchOutsideData.length === 0)
      return;
    d3.select(svgRef.current).selectAll("*").remove();
    
    let allObserverList = [];
    for(let i=0; i<patchOutsideData.length; i++){
      allObserverList.push(patchOutsideData[i][0].split("/")[3].split(".")[0]);
    }
    let set = new Set(allObserverList);
    let observerList = [...set];

    let countObserver = [];
    for(let i=0; i<observerList.length; i++){
      let counting = 0;
      for(let j=0; j<patchOutsideData.length; j++){
        let ob = patchOutsideData[j][0].split("/")[3].split(".")[0];
        if(observerList[i] == ob){
          counting = counting+1;
        }
      }
      countObserver.push([observerList[i], counting]);
    }
    

    let countData = [];
    let patchCount = 0;
    for(let i=0; i<patchOutsideData.length; i++){
      let _id = patchOutsideData[i][0].split("/")[3].split(".")[0];
      let _idIdx = 0;
      for(let j=0; j<countObserver.length; j++){
        if(_id == countObserver[j][0]){
          _idIdx = j;
          break;
        }
      }
      let patch = {
        id: _idIdx,
        observer: _id,
        index: patchCount,
        value: countObserver[_idIdx][1]
      };
      countData.push(patch);
      patchCount = patchCount+1;
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
      .domain(countData.map(function(d) { return d.observer; }))
      .padding(0.2);
    svg.append("g")
      .attr("transform", "translate(0," + drawHeight + ")")
      .call(d3.axisBottom(x))
      .selectAll("text")
        .attr("transform", "translate(-10,0)rotate(-45)")
        .style("text-anchor", "end");

    var yMax = d3.max(countData, (d=>d.value));
    var y = d3.scaleLinear()
    .domain([0, yMax])
    .range([ drawHeight, 0]);
    svg.append("g")
    .call(d3.axisLeft(y));


    svg.selectAll("rect_count")
    .data(countData)
    .enter()
    .append("rect")
    .attr("x", function(d){ return x(d.observer)})
    .attr("y", function(d){ return y(d.value); })
    .attr("width", x.bandwidth())
    .attr("height", function(d) { return drawHeight-y(d.value) })
    .attr("fill", function(d){
      if(d.id < colores_g.length){
        return colores_g[d.id];
      }else{
        let mIdx = d.id - colores_g.length;
        return colorEncoding[mIdx];
      }
    })
    .attr("opacity", function(d){
      if(selectedObserver.length == 0){
        return 1;
      }else{
        if(d.observer == selectedObserver[1]){
          return 1;
        }else{
          return 0.1;
        }
      }
    })
    .on("click", function(d){
      let selectedObInfo_str = String(d.id)+"/"+d.observer+"/"+String(d.index);
      const data = new FormData();
      data.set('selectedObserver', selectedObInfo_str);
      axios.post(`http://${window.location.hostname}:5000/api/horizontalBarChart/selectObserverUpdate`, data)
      .then(response => {
        // console.log(response);
        observerSelectFunction();
      }).catch(error => {
        alert(`Error - ${error.message}`);
      });
    });
    
  }, [,props.patchOutsideData, props.colorEncoding, props.selectedObserver]);

  return (
    <>
      {patchOutsideData.length > 0 &&
        <svg ref={svgRef}>
        </svg>
      }
    </>
  );
}

export default HorizontalBarChart;
