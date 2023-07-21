import React, { useEffect, useRef } from 'react';

function HumanFixationMap(props) {
  const { width, height, humanFixationMapURL, patchDataList, colorEncoding, patchDrawFlag, selectedObserver, patchBoxOpacity } = props;
  const svgRef = useRef();
  const d3 = window.d3v4;
  const colores_g = ["#109618", "#ff9900", "#990099", "#dd4477", "#0099c6", "#66aa00", "#b82e2e", "#316395", "#994499", "#22aa99", "#aaaa11", "#6633cc", "#e67300", "#8b0707", "#651067", "#329262", "#5574a6", "#3b3eac", "#3366cc", "#dc3912"];
  const PATCH_SIZE = 20;
  const PATCH_DRAW_LENGTH = 8;
  
  useEffect(() => {
    if(humanFixationMapURL.length == 0)
      return;
    d3.select(svgRef.current).selectAll("*").remove();
    
    // set the dimensions and margins of the graph
    var margin = {top: 0, right: 0, bottom: 0, left: 0};
    var drawWidth = width - (margin.left + margin.right);
    var drawHeight = height - (margin.top + margin.bottom);

    // var imgWidth = stiSize.width;
    // var imgHeight = stiSize.height;
    var imgWidth = humanFixationMapURL[0].width;
    var imgHeight = humanFixationMapURL[0].height;
    var stiWidth = drawWidth;
    var stiHeight = (imgHeight/imgWidth)*stiWidth;
    var ratio = stiWidth/imgWidth;
    if(stiWidth > drawWidth || stiHeight > drawHeight){
      stiHeight = drawHeight;
      stiWidth = (imgWidth/imgHeight)*stiHeight;
      ratio = stiHeight/imgHeight;
    }
    // console.log("RAW DATA LIST");
    // console.log(patchDataList);
    let allObserverList = [];
    for(let i=0; i<patchDataList.length; i++){
      allObserverList.push(patchDataList[i][0].split("/")[3]);
    }
    let set = new Set(allObserverList);
    let observerList = [...set];

    let patchData = [];
    for(let i=0; i<patchDataList.length; i++){
      let _id = patchDataList[i][0].split("/")[3];
      let _idIdx = 0;
      for(let j=0; j<observerList.length; j++){
        if(_id == observerList[j]){
          _idIdx = j;
          break;
        }
      }
      let box = {
        id: _idIdx,
        x: parseFloat((patchDataList[i][1]-PATCH_SIZE/2)*ratio),
        y: parseFloat((patchDataList[i][2]-PATCH_SIZE/2)*ratio),
        label: patchDataList[i][3],
        observer: _id
      };
      patchData.push(box);
    }
    // console.log("patchData");
    // console.log(patchData);

    var url_rnd = humanFixationMapURL[0].url+"?"+Math.random();
    var svg = d3.select(svgRef.current)
    .attr("width", width)
    .attr("height", height)
      .append("svg")
      .attr("width", width)
      .attr("height", height)
      .append("g")
      .attr("transform", "translate(0, 0)");

    svg.append("image")
    .attr('x', margin.left)
    .attr('y', margin.top)
    .attr("width", stiWidth)
    .attr("height", stiHeight)
    .attr("xlink:href", url_rnd);

    var patchFrame = svg.selectAll(".opFrame")
    .data(patchData)
    // .data(scanpathData)
    .enter()
    .append("g")
    .attr("class", "opFrame")
    .attr("transform", function(d) {
      let _x = d.x + margin.left - (PATCH_DRAW_LENGTH/2);
      let _y = d.y + margin.top - (PATCH_DRAW_LENGTH/2);
      return "translate(" + _x + "," + _y + ")";
    });

    patchFrame.append('rect')
    .attr("width", PATCH_DRAW_LENGTH)
    .attr("height", PATCH_DRAW_LENGTH)
    .style("fill", function(d){
      if(patchDrawFlag == "box"){
        return colorEncoding[d.label];
      }else if(patchDrawFlag == "observer"){
        if(d.id < colores_g.length){
          return colores_g[d.id];
        }else{
          let mIdx = d.id - colores_g.length;
          return colorEncoding[mIdx];
        }
      }else{
        return "none";
      }
    })
    .attr("stroke", function(d){ return colorEncoding[d.label] })
    .attr("stroke-width", "2px")
    .attr("opacity", function(d){
      if(selectedObserver.length == 0){
        return patchBoxOpacity;
      }else{
        if(selectedObserver[1] == d.observer){
          return 1;
        }else{
          if(patchBoxOpacity == 1){
            return 0.3;
          }else{
            return patchBoxOpacity;
          }
        }
      }
    });

    
    
  }, [, props.humanFixationMapURL, props.patchDataList, props.patchDrawFlag, props.patchBoxOpacity, props.selectedObserver]);

  return (
    <>
    { humanFixationMapURL.length != 0 &&
      <svg ref={svgRef}>
      </svg>
    }
    </>
  );
}

export default HumanFixationMap;

