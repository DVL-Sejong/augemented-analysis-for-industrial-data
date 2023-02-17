import React, { useEffect } from 'react'

export default function ECDFChart(props) {
  const { data } = props
  const d3 = window.d3v4

  useEffect(() => {
    d3.select('.ecdf-wrapper').selectAll('*').remove()

    var margin = {top: 10, right: 10, bottom: 20, left: 30},
        width = 160 - margin.left - margin.right,
        height = 125 - margin.top - margin.bottom;

    var svg = d3.select('.ecdf-wrapper')
      .append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
      .append("g")
        .attr("transform",
              "translate(" + margin.left + "," + margin.top + ")");

    var sumstat = d3
      .nest()
      .key(function(d) { return d.index; })
      .entries(data.ECDFchartData);

    var x = d3.scaleLinear()
      .domain(d3.extent(data.ECDFchartData, function(d) { return +d.x; }))
      .range([ 0, width ]);
    svg.append("g")
      .attr("transform", "translate(0," + height + ")")
      .call(d3.axisBottom(x).ticks(5));

    var y = d3.scaleLinear()
      .domain([0, d3.max(data.ECDFchartData, function(d) { return +d.y; })])
      .range([ height, 10 ]);
    svg.append("g")
      .call(d3.axisLeft(y));

    var res = sumstat.map(function(d){ return d.key })
    var color = d3.scaleOrdinal()
      .domain(res)
      .range(["#CCCCCC", "#6C757D"])

    svg.selectAll(".line")
        .data(sumstat)
        .enter()
        .append("path")
          .attr("fill", "none")
          .attr("stroke", function(d){ return color(d.key) })
          .attr("stroke-width", 1.5)
          .attr("d", function(d){
            return d3
              .line()
              .x(function(d) { return x(d.x); })
              .y(function(d) { return y(+d.y); })
              (d.values)
          })

    svg.append("text").attr("x", 30).attr("y", 0).text(`K-S test: ${data.KStestValue}`).style("font-size", "12px").attr("alignment-baseline", "middle")

  }, [data, d3])

  return (
    <div className='ecdf-wrapper' style={{ position: 'relative', bottom: -20 }} />
  )
}