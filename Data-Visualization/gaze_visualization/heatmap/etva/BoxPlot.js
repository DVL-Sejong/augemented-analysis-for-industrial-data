// d3 code reference
// https://bl.ocks.org/rjurney/e04ceddae2e8f85cf3afe4681dac1d74
import React, { useEffect, useRef } from 'react';

function BoxPlot(props) {
  const { width, height, patchDataList, colorEncoding } = props;
  const svgRef = useRef();
  const d3 = window.d3v4;
  // const COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628", "#f781bf", "#999999"];
  const featureNumber = 8;

  useEffect(() => {
    if(patchDataList.length === 0)
      return;

    d3.select(svgRef.current).selectAll("*").remove();
    
    // set the dimensions and margins of the graph
    var margin = {top: 20, right: 50, bottom: 10, left: 5};
    // margin.right = getData.length*15;
    var drawWidth = width - margin.left - margin.right;
    var drawHeight = height - margin.top - margin.bottom;

    var barWidth = 30;

    var groupCounts = {};
    var globalCounts = [];
    for(let i=4; i<featureNumber+4; i++){
      var key = (i-4).toString();
      groupCounts[key] = [];
      for(let j=0; j<patchDataList.length; j++){
        groupCounts[key].push(patchDataList[j][i]);
        globalCounts.push(patchDataList[j][i]);
      }
    }
    // console.log(groupCounts);
    // console.log(globalCounts);

    // Sort group counts so quantile methods work
    for(var key in groupCounts) {
      var groupCount = groupCounts[key];
      groupCounts[key] = groupCount.sort(sortNumber);
    }

    // Setup a color scale for filling each box
    var colorScale = d3.scaleOrdinal(d3.schemeCategory20)
      .domain(Object.keys(groupCounts));

    // Prepare the data for the box plots
    var boxPlotData = [];
    for (var [key, groupCount] of Object.entries(groupCounts)) {
      var record = {};
      var localMin = d3.min(groupCount);
      var localMax = d3.max(groupCount);

      record["key"] = key;
      record["counts"] = groupCount;
      record["quartile"] = boxQuartiles(groupCount);
      record["whiskers"] = [localMin, localMax];
      // record["color"] = colorScale(key);
      record["color"] = colorEncoding[parseInt(key)];

      boxPlotData.push(record);
    }

    // Compute an ordinal xScale for the keys in boxPlotData
    var xScale = d3.scalePoint()
      .domain(Object.keys(groupCounts))
      .rangeRound([0, drawWidth])
      .padding([0.1]);

    // Compute a global y scale based on the global counts
    var min = d3.min(globalCounts);
    var max = d3.max(globalCounts);
    var yScale = d3.scaleLinear()
      .domain([min, max])
      .range([0, drawHeight]);

      
    // Setup the svg and group we will draw the box plot in
    var svg = d3.select(svgRef.current)
      .attr("width", drawWidth + margin.left + margin.right)
      .attr("height", drawHeight + margin.top + margin.bottom)
        .append("svg")
        .attr("width", drawWidth + margin.left + margin.right)
        .attr("height", drawHeight + margin.top + margin.bottom)
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    // Move the left axis over 25 pixels, and the top axis over 35 pixels
    var axisG = svg.append("g").attr("transform", "translate(20,0)");
    var axisTopG = svg.append("g").attr("transform", "translate(35,0)");

    // Setup the group the box plot elements will render in
    var g = svg.append("g")
      .attr("transform", "translate(20,5)");

    // Draw the box plot vertical lines
    var verticalLines = g.selectAll(".verticalLines")
      .data(boxPlotData)
      .enter()
      .append("line")
      .attr("x1", function(datum) {
          return xScale(datum.key) + barWidth/2;
        }
      )
      .attr("y1", function(datum) {
          var whisker = datum.whiskers[0];
          return yScale(whisker);
        }
      )
      .attr("x2", function(datum) {
          return xScale(datum.key) + barWidth/2;
        }
      )
      .attr("y2", function(datum) {
          var whisker = datum.whiskers[1];
          return yScale(whisker);
        }
      )
      .attr("stroke", "#000")
      .attr("stroke-width", 1)
      .attr("fill", "none");

    // Draw the boxes of the box plot, filled in white and on top of vertical lines
    var rects = g.selectAll("rect")
      .data(boxPlotData)
      .enter()
      .append("rect")
      .attr("width", barWidth)
      .attr("height", function(datum) {
          var quartiles = datum.quartile;
          var height = yScale(quartiles[2]) - yScale(quartiles[0]);
          return height;
        }
      )
      .attr("x", function(datum) {
          return xScale(datum.key);
        }
      )
      .attr("y", function(datum) {
          return yScale(datum.quartile[0]);
        }
      )
      .attr("fill", function(datum) {
        return datum.color;
        }
      )
      .attr("stroke", "#000")
      .attr("stroke-width", 1);

    // Now render all the horizontal lines at once - the whiskers and the median
    var horizontalLineConfigs = [
      // Top whisker
      {
        x1: function(datum) { return xScale(datum.key) },
        y1: function(datum) { return yScale(datum.whiskers[0]) },
        x2: function(datum) { return xScale(datum.key) + barWidth },
        y2: function(datum) { return yScale(datum.whiskers[0]) }
      },
      // Median line
      {
        x1: function(datum) { return xScale(datum.key) },
        y1: function(datum) { return yScale(datum.quartile[1]) },
        x2: function(datum) { return xScale(datum.key) + barWidth },
        y2: function(datum) { return yScale(datum.quartile[1]) }
      },
      // Bottom whisker
      {
        x1: function(datum) { return xScale(datum.key) },
        y1: function(datum) { return yScale(datum.whiskers[1]) },
        x2: function(datum) { return xScale(datum.key) + barWidth },
        y2: function(datum) { return yScale(datum.whiskers[1]) }
      }
    ];

    for(let i=0; i < horizontalLineConfigs.length; i++) {
      var lineConfig = horizontalLineConfigs[i];

      // Draw the whiskers at the min for this series
      var horizontalLine = g.selectAll(".whiskers")
        .data(boxPlotData)
        .enter()
        .append("line")
        .attr("x1", lineConfig.x1)
        .attr("y1", lineConfig.y1)
        .attr("x2", lineConfig.x2)
        .attr("y2", lineConfig.y2)
        .attr("stroke", "#000")
        .attr("stroke-width", 1)
        .attr("fill", "none");
    }

    // Setup a scale on the left
    var axisLeft = d3.axisLeft(yScale);
    axisG.append("g")
      .call(axisLeft);

    // Setup a series axis on the top
    var axisTop = d3.axisTop(xScale);
    axisTopG.append("g")
      .call(axisTop);

    function boxQuartiles(d) {
      return [
        d3.quantile(d, .25),
        d3.quantile(d, .5),
        d3.quantile(d, .75)
      ];
    }
      
    // Perform a numeric sort on an array
    function sortNumber(a,b) {
      return a - b;
    }
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

export default BoxPlot;
