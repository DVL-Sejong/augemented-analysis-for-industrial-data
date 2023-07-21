import React, { useEffect, useRef } from 'react';

function ObserverTable(props) {
  const { width, height, patchDataList } = props;
  const svgRef = useRef();
  const d3 = window.d3v3;
  const columns = ["ID", "In", "Out"];
  
  useEffect(() => {
    if (patchDataList.length === 0)
      return;
    d3.select(svgRef.current).selectAll("*").remove();

    // console.log(data);
    let allObserverList = [];
    for(let i=0; i<patchDataList.length; i++){
      allObserverList.push(patchDataList[i][0].split(".")[0].split("/")[3]);
    }
    let set = new Set(allObserverList);
    let observerList = [...set];
    
    let tableData = [];

    for(let i=0; observerList.length; i++){
      let count_in = 0;
      let count_out = 0;
      for(let j=0; j<patchDataList.length; j++){
        let _id = patchDataList[j][0].split(".")[0].split("/")[3];
        if(observerList[i] == _id){
          if(patchDataList[j][4] == 0){
            count_out = count_out + 1;
          }else{
            count_in = count_in + 1;
          }
        }
      }
      tableData.push({
        id: _id,
        in: count_in,
        out: count_out
      });
    }
    console.log("tableData");
    console.log(tableData);
    

    
    let margin = {top: 10, right: 10, bottom: 30, left: 10};
    let drawWidth = width - (margin.left + margin.right);
    let drawHeight = height - (margin.top + margin.bottom);

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




  }, [props.patchDataList]);
  
  return (
    <>
      {patchDataList.length !== 0 && 
        <svg ref={svgRef}>
        </svg>
      }
    </>
  );
}

export default ObserverTable;