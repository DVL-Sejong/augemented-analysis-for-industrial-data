function draw_y_chart(dataset){
  console.log("draw_y_chart");

  // set the dimensions and margins of the graph
  var margin = {top: 10, right: 30, bottom: 30, left: 60},
      width = 460 - margin.left - margin.right,
      height = 400 - margin.top - margin.bottom;

  // append the svg object to the body of the page
  var svg = d3.select("#my_dataviz")
    .append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
    .append("g")
      .attr("transform",
            "translate(" + margin.left + "," + margin.top + ")");

  // Add X axis --> it is a date format
  var x = d3.scaleTime()
    .domain(d3.extent(dataset.y, function(d, i) { return i; }))
    .range([ 0, width ]);

  svg.append("g")
    .attr("transform", "translate(0," + height + ")")
    .call(d3.axisBottom(x));

  // Add Y axis
  var y = d3.scaleLinear()
    .domain([0, d3.max(dataset.y, function(d) { return d; })])
    .range([ height, 0 ]);
  svg.append("g")
    .call(d3.axisLeft(y));

/*
  var lineFunction = d3.line()
      .x(function(d, i){ return i; })
      .y(function(d){ return d; })
      .curve(d3.curveLinear);
*/
  // Add the line
  svg.append("path")
    .datum(dataset.y)
    .attr("fill", "none")
    .attr("stroke", "steelblue")
    .attr("stroke-width", 1.5)
    .attr("d", d3.line()
      .x(function(d, i) { return x(i) })
      .y(function(d) { return y(d) })
      )
}