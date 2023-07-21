// https://www.d3-graph-gallery.com/graph/parallel_custom.html
import axios from 'axios';
import React, { useEffect, useRef } from 'react';

function BrushParallelCoordinateChart(props) {
  const { width, height, patchDataFileURL, colorEncoding, selectedObserver, patchLineOpacity, lineOpacity_label0, lineOpacity_label1} = props;
  const svgRef = useRef();
  const d3 = window.d3v3;
  const colores_g = ["#109618", "#ff9900", "#990099", "#dd4477", "#0099c6", "#66aa00", "#b82e2e", "#316395", "#994499", "#22aa99", "#aaaa11", "#6633cc", "#e67300", "#8b0707", "#651067", "#329262", "#5574a6", "#3b3eac", "#3366cc", "#dc3912"];
  
  useEffect(() => {
    if (patchDataFileURL.length === 0)
      return;
    d3.select(svgRef.current).selectAll("*").remove();
    // console.log(patchDataFileURL);
    let dimensions = ["intensity", "color", "orientation", "curvature", "center_bias", "entropy_rate", "log_spectrum", "HOG"];

    let margin = {top: 30, right: 50, bottom: 30, left: 10};
    let drawWidth = width - (margin.left + margin.right);
    let drawHeight = height - (margin.top + margin.bottom);
    
    var x = d3.scale.ordinal().rangePoints([0, width], 1);
    var y = {};
    var dragging = {};

    var line = d3.svg.line();
    var axis = d3.svg.axis().orient("left");
    var background;
    var foreground;

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

    d3.csv(patchDataFileURL, function(data) {
      // console.log(data);
      let allObserverList = [];
      for(let i=0; i<data.length; i++){
        allObserverList.push(data[i].id.split(".")[0].split("/")[3]);
      }
      let set = new Set(allObserverList);
      let observerList = [...set];

      var lMax = d3.max(data, (d=>parseFloat(d.label)));
      var labelDomainRange = [];
      for(let i=0; i<lMax+1; i++){
        labelDomainRange.push(String(i));
      }
      var colorRange = [];
      for(let i=0; i<labelDomainRange.length; i++){
        colorRange.push(colorEncoding[i]);
      }

      var color = d3.scale.ordinal()
      .domain(labelDomainRange)
      .range(colorRange)

      
      // Extract the list of dimensions and create a scale for each.
      x.domain(dimensions = d3.keys(data[0]).filter(function(d) {
        return d != "id" && d != "x" && d != "y" && (y[d] = d3.scale.linear()
          .domain(d3.extent(data, function(p) { return +p[d]; }))
          .range([drawHeight, 0]));
      }));

      // Add grey background lines for context.
      background = svg.append("g")
      .attr("class", "background")
      .selectAll("path")
      .data(data)
      .enter().append("path")
      .attr("d", path)
      .style("stroke", function(d){ 
        if(selectedObserver.length != 0){
          let ob = d.id.split(".")[0].split("/")[3];
          let obIdx = 0;
          for(let i=0; i<observerList.length; i++){
            if(ob == observerList[i]){
              obIdx = i;
              break;
            }
          }
          if(selectedObserver[1] == ob){
            return colores_g[obIdx];
          }else{
            return(color(String(d.label)));
          }
        }else{
          return(color(String(d.label)));
        }
      })
      .style("opacity", function(d){
        // if(selectedObserver.length != 0){
        //   let ob = d.id.split(".")[0].split("/")[3];
        //   let obIdx = 0;
        //   for(let i=0; i<observerList.length; i++){
        //     if(ob == observerList[i]){
        //       obIdx = i;
        //       break;
        //     }
        //   }
        //   if(selectedObserver[1] == ob){
        //     return patchLineOpacity;
        //   }else{
        //     return lineOpacity_label0;
        //   }
        // }else{
        //   return lineOpacity_label0;
        // }

        return lineOpacity_label0;
      });

      // Add blue foreground lines for focus.
      foreground = svg.append("g")
      .attr("class", "foreground")
      .selectAll("path")
      .data(data)
      .enter().append("path")
      .attr("d", path)
      .style("stroke", function(d){ 
        if(selectedObserver.length != 0){
          let ob = d.id.split(".")[0].split("/")[3];
          let obIdx = 0;
          for(let i=0; i<observerList.length; i++){
            if(ob == observerList[i]){
              obIdx = i;
              break;
            }
          }
          if(selectedObserver[1] == ob){
            return colores_g[obIdx];
          }else{
            return(color(String(d.label)));
          }
        }else{
          return(color(String(d.label)));
        }
      })
      .style("opacity", function(d){
        if(selectedObserver.length != 0){
          let ob = d.id.split(".")[0].split("/")[3];
          let obIdx = 0;
          for(let i=0; i<observerList.length; i++){
            if(ob == observerList[i]){
              obIdx = i;
              break;
            }
          }
          if(selectedObserver[1] == ob){
            return patchLineOpacity;
          }else{
            return lineOpacity_label1;
          }
        }else{
          return lineOpacity_label1;
        }
      });

      // Add a group element for each dimension.
      var g = svg.selectAll(".dimension")
      .data(dimensions)
      .enter().append("g")
      .attr("class", "dimension")
      .attr("transform", function(d) { return "translate(" + x(d) + ")"; })
      .call(d3.behavior.drag()
        .origin(function(d) { return {x: x(d)}; })
        .on("dragstart", function(d) {
          dragging[d] = x(d);
          background.attr("visibility", "hidden");
        })
        .on("drag", function(d) {
          dragging[d] = Math.min(drawWidth, Math.max(0, d3.event.x));
          foreground.attr("d", path);
          dimensions.sort(function(a, b) { return position(a) - position(b); });
          x.domain(dimensions);
          g.attr("transform", function(d) { return "translate(" + position(d) + ")"; })
        })
        .on("dragend", function(d) {
          delete dragging[d];
          transition(d3.select(this)).attr("transform", "translate(" + x(d) + ")");
          transition(foreground).attr("d", path);
          background
              .attr("d", path)
            .transition()
              .delay(500)
              .duration(0)
              .attr("visibility", null);
        }));

      // Add an axis and title.
      g.append("g")
      .attr("class", "axis")
      .each(function(d) { d3.select(this).call(axis.scale(y[d])); })
      .append("text")
        .style("text-anchor", "middle")
        .attr("y", -9)
        .text(function(d) { return d; });

      // Add and store a brush for each axis.
      g.append("g")
      .attr("class", "brush")
      .each(function(d) {
        d3.select(this).call(y[d].brush = d3.svg.brush().y(y[d]).on("brushstart", brushstart).on("brush", brush));
      })
      .selectAll("rect")
        .attr("x", -8)
        .attr("width", 16);
    });

    function position(d) {
      var v = dragging[d];
      return v == null ? x(d) : v;
    }
    
    function transition(g) {
      return g.transition().duration(500);
    }
    
    // Returns the path for a given data point.
    function path(d) {
      return line(dimensions.map(function(p) { return [position(p), y[p](d[p])]; }));
    }
    
    function brushstart() {
      d3.event.sourceEvent.stopPropagation();
    }
    
    // Handles a brush event, toggling the display of foreground lines.
    function brush() {
      var actives = dimensions.filter(function(p) { return !y[p].brush.empty(); });
      var extents = actives.map(function(p) { return y[p].brush.extent(); });
      foreground.style("display", function(d) {
        return actives.every(function(p, i) {
          return extents[i][0] <= d[p] && d[p] <= extents[i][1];
        }) ? null : "none";
      });

      if(actives.length != 0 && extents.length != 0){
        let actExt_str = "";
        for(let i=0; i<actives.length; i++){
          if(i==0){
            actExt_str = actives[i]+"-"+String(extents[i][0])+"-"+String(extents[i][1]);
          }else{
            actExt_str = actExt_str +"/"+ actives[i]+"-"+String(extents[i][0])+"-"+String(extents[i][1]);
          }
        }
        if(actExt_str == ""){
          actExt_str = "0";
        }
        const postData = new FormData();
        postData.set('actives', actExt_str);
        axios.post(`http://${window.location.hostname}:5000/api/brushParallelCoordinateChart/updateActives`, postData);
        // .then(response => {
        //   // console.log(response);
          
        // }).catch(error => {
        //   alert(`Error - ${error.message}`);
        // });

      }
      
    }




  }, [props.patchDataFileURL, props.colorEncoding, props.selectedObserver, props.patchLineOpacity, props.lineOpacity_label0, props.lineOpacity_label1]);
  
  return (
    <>
      {patchDataFileURL.length !== 0 && 
        <svg ref={svgRef}>
        </svg>
      }
    </>
  );
}

export default BrushParallelCoordinateChart;