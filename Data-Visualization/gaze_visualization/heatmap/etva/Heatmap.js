// reference code
// https://www.d3-graph-gallery.com/graph/heatmap_tooltip.html
import React, { useEffect, useRef } from 'react';

function Heatmap(props) {
  const {width, height, dataURL, FEATURE_DEFINE, STI_CLASS_DEFINE} = props;
  const svgRef = useRef();
  const d3 = window.d3v4;
  var selectedCols = [];
  var selectedRows = [];

  useEffect(() => {
    if (typeof dataURL !== 'string' && dataURL.length === 0)
      return;
    d3.select(svgRef.current).selectAll("*").remove();
    // console.log(svgRef.current)
    
    // set the dimensions and margins of the graph
    var margin = {top: 10, right: 10, bottom: 20, left: 30},
    drawWidth = width - margin.left - margin.right,
    drawHeight = height - margin.top - margin.bottom;

    // append the svg object to the body of the page
    // var svg = d3.select(svgRef.current)
    var svg = d3.select(svgRef.current)
      .attr("width", drawWidth + margin.left + margin.right)
      .attr("height", drawHeight + margin.top + margin.bottom)
      .append("svg")
        .attr("width", drawWidth + margin.left + margin.right)
        .attr("height", drawHeight + margin.top + margin.bottom)
      .append("g")
        .attr("transform",
              "translate(" + margin.left + "," + margin.top + ")");

    // console.log(FEATURE_DEFINE);
    // console.log(STI_CLASS_DEFINE);

    // Labels of row and columns
    var myGroups = ['intensity', 'color', 'orientation', 'curvature', 'center_b', 'entropy_r', 'log_s', 'HOG'];
    var myVars = [];
    // for(let i=0; i<FEATURE_DEFINE.length; i++){
    //   // myVars.push(FEATURE_DEFINE[i][2]);
    //   myGroups.push("f"+i.toString());
    // }
    for(let i=0; i<STI_CLASS_DEFINE.length; i++){
      let datasetName = STI_CLASS_DEFINE[i]['datasetName'].charAt(0);
      for(let j=0; j<STI_CLASS_DEFINE[i]['classList'].length; j++){
        let stiRename = datasetName;
        if(j<10){
          stiRename = stiRename + "0" + j.toString();
        }else{
          stiRename = stiRename + j.toString();
        }
        myVars.push(stiRename);
      }
    }
    
    

    // Build X scales and axis:
    var x = d3.scaleBand()
      .range([ 0, drawWidth ])
      .domain(myGroups)
      .padding(0.01);
    svg.append("g")
      .attr("class", "xaxis")
      .attr("transform", "translate(0," + drawHeight + ")")
      .attr("stroke", function(d){
        var _selected = false;
        for(var i=0; i<selectedCols.length; i++){
          if(d === selectedCols[i]){
              _selected = true;
              break;
            }
          }
          if(_selected){
            return "#737373";
          }else{
            return "black";
          }
      })
      .style("font", "8px sans-serif")
      .call(d3.axisBottom(x))

    // d3.selectAll(".xaxis .tick")
    //   .on('click', function(d){
    //     var dCheck = 0;
    //     for(let i=0; i<selectedCols.length; i++){
    //       if(selectedCols[i] === d){
    //         dCheck++;
    //         break;
    //       }
    //     }
    //     if(dCheck === 0){
    //       selectedCols.push(d);
    //       for(let i=0; i<selectedCols.length; i++){
    //         for(let j=0; j<myGroups.length; j++){
    //           if(selectedCols[i] === myGroups[j]){
    //             myGroups.splice(j, 1);
    //             break;
    //           }
    //         }
    //       }
    //     }else{
    //       myGroups.splice(0, 0, d);
    //       for(let i=0; i<selectedCols.length; i++){
    //         if(selectedCols[i] === d){
    //           selectedCols.splice(i, 1);
    //         }
    //       }
    //   }

    //   // console.log(selectedCols);
    //   // console.log(myGroups);
    //   d3.selectAll(".xaxis .tick").transition()
    //     .attr("stroke", function(d){
    //       var _selected = false;
    //       for(let i=0; i<selectedCols.length; i++){
    //         if(d === selectedCols[i]){
    //           _selected = true;
    //           break;
    //         }
    //       }
    //       if(_selected){
    //         return "#737373";
    //       }else{
    //         return "black";
    //       }
    //   });

    //   svg.selectAll("rect").transition()
    //   .attr("stroke", function(d){
    //     var _ccheck = false;
    //     for(let i=0; i<selectedCols.length; i++){
    //       if(d.group === selectedCols[i]){
    //         _ccheck = true;
    //         break;
    //       }
    //     }
    //     var _rcheck = false;
    //     for(let i=0; i<selectedRows.length; i++){
    //       if(d.variable === selectedRows[i]){
    //         _rcheck = true;
    //         break;
    //       }
    //     }

    //     if(_rcheck || _ccheck){
    //       return "#737373";
    //     }else{
    //       // if(d.value == -999)
    //       //   return "#fb9a99";
    //       // else if(d.value < 1)
    //       //   return "#fdbf6f";
    //       // else
    //       //   return myColor(d.value);
    //       return "#636363";
    //     }
    //   })
    //   .attr("fill", function(d){
    //     var _rcheck = false;
    //     for(let i=0; i<selectedRows.length; i++){
    //       if(d.variable === selectedRows[i]){
    //         _rcheck = true;
    //         break;
    //       }
    //     }
    //     var _ccheck = false;
    //     for(let i=0; i<selectedCols.length; i++){
    //       if(d.group === selectedCols[i]){
    //         _ccheck = true;
    //         break;
    //       }
    //     }
    //     if(d.value == -999)
    //       return "#fb9a99";
    //     else if(d.value < 1)
    //       return "#fdbf6f";
    //     else
    //       return myColor(d.value);
    //   })
    //   .attr("opacity", function(d){
    //     var _rcheck = false;
    //     for(let i=0; i<selectedRows.length; i++){
    //       if(d.variable === selectedRows[i]){
    //         _rcheck = true;
    //         break;
    //       }
    //     }
    //     var _ccheck = false;
    //     for(let i=0; i<selectedCols.length; i++){
    //       if(d.group === selectedCols[i]){
    //         _ccheck = true;
    //         break;
    //       }
    //     }
    //     if(_rcheck || _ccheck){
    //       return 0.3;
    //     }else{
    //       return 1;
    //     }
    //   });
    //   sendRemoveList(selectedCols, selectedRows);
    // });
    
    // Build X scales and axis:
    var y = d3.scaleBand()
      .range([ drawHeight, 0 ])
      .domain(myVars)
      .padding(0.01);
    svg.append("g")
      .attr("class", "yaxis")
      .attr("stroke", function(d){
        var _selected = false;
        for(let i=0; i<selectedRows.length; i++){
          if(d === selectedRows[i]){
            _selected = true;
            break;
          }
        }
        if(_selected){
          return "#737373";
        }else{
          return "black";
        }
      })
      .style("font", "8px sans-serif")
      .call(d3.axisLeft(y));

    // d3.selectAll(".yaxis .tick")
    //   .on('click', function(d){
    //     var dCheck = 0;
    //     for(let i=0; i<selectedRows.length; i++){
    //       if(selectedRows[i] === d){
    //         dCheck++;
    //         break;
    //       }
    //     }
    //     if(dCheck === 0){
    //       selectedRows.push(d);
    //       for(let i=0; i<selectedRows.length; i++){
    //         for(var j=0; j<myVars.length; j++){
    //           if(selectedRows[i] === myVars[j]){
    //             myVars.splice(j, 1);
    //             break;
    //           }
    //         }
    //       }
    //     }else{
    //         myVars.splice(0, 0, d);
    //         for(let i=0; i<selectedRows.length; i++){
    //           if(selectedRows[i] === d){
    //             selectedRows.splice(i, 1);
    //           }
    //         }
    //       }

    //       // console.log(selectedRows);
    //       // console.log(myVars);
    //       d3.selectAll(".yaxis .tick").transition()
    //       .attr("stroke", function(d){
    //         var _selected = false;
    //         for(let i=0; i<selectedRows.length; i++){
    //           if(d === selectedRows[i]){
    //             _selected = true;
    //             break;
    //           }
    //         }
    //         if(_selected){
    //           return "#737373";
    //         }else{
    //           return "black";
    //         }
    //       });
          
    //       svg.selectAll("rect").transition()
    //       .attr("stroke", function(d){
    //         var _rcheck = false;
    //         for(let i=0; i<selectedRows.length; i++){
    //           if(d.variable === selectedRows[i]){
    //             _rcheck = true;
    //             break;
    //           }
    //         }
    //         var _ccheck = false;
    //         for(let i=0; i<selectedCols.length; i++){
    //           if(d.group === selectedCols[i]){
    //             _ccheck = true;
    //             break;
    //           }
    //         }

    //         if(_rcheck || _ccheck){
    //           return "#737373";
    //         }else{
    //           return "#636363";
    //         }
    //       })
    //       .attr("fill", function(d){
    //         var _rcheck = false;
    //         for(let i=0; i<selectedRows.length; i++){
    //           if(d.variable === selectedRows[i]){
    //             _rcheck = true;
    //             break;
    //           }
    //         }
    //         var _ccheck = false;
    //         for(let i=0; i<selectedCols.length; i++){
    //           if(d.group === selectedCols[i]){
    //             _ccheck = true;
    //             break;
    //           }
    //         }
    //         if(d.value == -999)
    //           return "#fb9a99";
    //         else if(d.value < 1)
    //           return "#fdbf6f";
    //         else
    //           return myColor(d.value);
    //       })
    //       .attr("opacity", function(d){
    //         var _rcheck = false;
    //         for(let i=0; i<selectedRows.length; i++){
    //           if(d.variable === selectedRows[i]){
    //             _rcheck = true;
    //             break;
    //           }
    //         }
    //         var _ccheck = false;
    //         for(let i=0; i<selectedCols.length; i++){
    //           if(d.group === selectedCols[i]){
    //             _ccheck = true;
    //             break;
    //           }
    //         }
    //         if(_rcheck || _ccheck){
    //           return 0.3;
    //         }else{
    //           return 1;
    //         }
    //       });
    //       sendRemoveList(selectedCols, selectedRows);
    //   });


    // Build color scale
    var myColor = d3.scaleLinear()
      .range(["#e41a1c", "#377eb8"])
      .domain([0,100])

    //Read the data
    d3.csv(dataURL, function(data) {
      // console.log(data);
      // console.log('myGroups');
      // console.log(myGroups);
      // console.log('myVars');
      // console.log(myVars);
      
        // create a tooltip
      // var tooltip = d3.select("#my_dataviz")
      //   .append("div")
      //   .style("opacity", 0)
      //   .attr("class", "tooltip")
      //   .style("background-color", "white")
      //   .style("border", "solid")
      //   .style("border-width", "2px")
      //   .style("border-radius", "5px")
      //   .style("padding", "5px")
      //   .style("position", "absolute")

      //   // Three function that change the tooltip when user hover / move / leave a cell
      // var mouseover = function(d) {
      //   tooltip.style("opacity", 1)
      // }
      // var mousemove = function(d) {
      //   tooltip
      //     .html("The exact value of<br>this cell is: " + d.value)
      //     .style("left", (d3.mouse(this)[0]+40) + "px")
      //     .style("top", (d3.mouse(this)[1]+50) + "px")
      // }
      // var mouseleave = function(d) {
      //   tooltip.style("opacity", 0)
      // }

      //   // add the squares
      svg.selectAll()
      .data(data, function(d) {return d.group+':'+d.variable;})
      .enter()
      .append("rect")
        .attr("x", function(d) {return x(d.group) })
        .attr("y", function(d) {return y(d.variable) })
        .attr("rx", 4)
        .attr("ry", 4)
        .attr("width", x.bandwidth() )
        .attr("height", y.bandwidth() )
        .style("fill", function(d) { 
          if(d.value == -999)
            return "#fb9a99"
          else if(d.value < 0)
            return "#fdbf6f"
          else
            return myColor(parseFloat(d.value));
        })
        .attr("stroke", "#636363")
        .attr("stroke-width", 1);
      //   .on("mouseover", mouseover)
      //   .on("mousemove", mousemove)
      //   .on("mouseleave", mouseleave)
      });
      
  }, [props.dataURL, props.FEATURE_DEFINE, props.STI_CLASS_DEFINE]);

  return (
      <>
        {typeof dataURL === 'string' && dataURL.length > 0 &&
            <div id="my_dataviz">
              <svg ref={svgRef}>
              </svg>
            </div>
        }
      </>
  );
}

export default Heatmap;
