import React, { useEffect, useRef } from 'react';

function GTMapWithPoints(props) {
  const { width, height, mapURL, pointDataList, pointOpacity } = props;
  const svgRef = useRef();
  const d3 = window.d3v4;
  const PATCH_SIZE = 20;
  
  useEffect(() => {
    if(mapURL.length == 0)
      return;
    d3.select(svgRef.current).selectAll("*").remove();
    
    // set the dimensions and margins of the graph
    var margin = {top: 1, right: 1, bottom: 1, left: 1};
    var drawWidth = width - (margin.left + margin.right);
    var drawHeight = height - (margin.top + margin.bottom);

    // var imgWidth = stiSize.width;
    // var imgHeight = stiSize.height;
    var imgWidth = mapURL[0].width;
    var imgHeight = mapURL[0].height;
    var stiWidth = drawWidth;
    var stiHeight = (imgHeight/imgWidth)*stiWidth;
    var ratio = stiWidth/imgWidth;
    if(stiWidth > drawWidth || stiHeight > drawHeight){
      stiHeight = drawHeight;
      stiWidth = (imgWidth/imgHeight)*stiHeight;
      ratio = stiHeight/imgHeight;
    }
    let PATCH_DRAW_LENGTH = PATCH_SIZE*ratio;

    // console.log("ratio");
    // console.log(ratio);
    // console.log("RAW DATA LIST");
    // console.log(pointDataList);
    let allObserverList = [];
    for(let i=0; i<pointDataList.length; i++){
      allObserverList.push(pointDataList[i][0].split("/")[3].split(".")[0]);
    }
    let set = new Set(allObserverList);
    let observerList = [...set];

    let patchData = [];
    for(let i=0; i<pointDataList.length; i++){
      let _id = pointDataList[i][0].split("/")[3].split(".")[0];
      let _idIdx = 0;
      for(let j=0; j<observerList.length; j++){
        if(_id == observerList[j]){
          _idIdx = j;
          break;
        }
      }
      let p = {
        id: _idIdx,
        x: +pointDataList[i][1]*ratio,
        y: +pointDataList[i][2]*ratio,
        label: +pointDataList[i][4],
        observer: _id
      };
      patchData.push(p);
    }
    // console.log("patchData");
    // console.log(patchData);

    // var url_rnd = mapURL[0].url+"?"+Math.random();
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
    .attr("xlink:href", mapURL[0].url);

    var point = svg.selectAll(".gPoint")
    .data(patchData)
    .enter()
    .append("g")
    .attr("class", "gPoint")
    .attr("transform", function(d) {
      let _x = d.x + margin.left;
      let _y = d.y + margin.top;
      return "translate(" + _x + "," + _y + ")";
    });

    point.append("circle")
    .attr('cx', 0)
    .attr('cy', 0)
    .attr('r', PATCH_DRAW_LENGTH/2)
    .style('fill', function(d){
      if(d.label == 0){
        return "red";
      }else if(d.label == 1){
        return "blue";
      }else{
        return "green";
      }
    })
    .style("opacity", pointOpacity);
    
  }, [, props.mapURL, props.pointDataList, props.pointOpacity ]);

  return (
    <>
    { mapURL.length != 0 &&
      <svg ref={svgRef}>
      </svg>
    }
    </>
  );
}

export default GTMapWithPoints;

