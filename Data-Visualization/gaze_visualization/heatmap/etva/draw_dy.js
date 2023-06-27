function draw_dy_chart(dataset){
  console.log("draw_dy_chart");

  // set the dimensions and margins of the graph
  var margin = {top: 10, right: 30, bottom: 30, left: 60},
      width = w - margin.left - margin.right,
      height = h - margin.top - margin.bottom;

  var dy = [];
  var prev = 0;
  for(var i=0; i<dataset.y.length; i++){
    var cur = dataset.y[i];
    if(i==0){
      prev = cur;
      continue;
    }
    var _dy = cur-prev;
    dy.push(_dy);

    prev = cur;
  }

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
    .domain(d3.extent(dy, function(d, i) { return i; }))
    .range([ 0, width ]);

  svg.append("g")
    .attr("transform", "translate(0," + height + ")")
    .call(d3.axisBottom(x));

  // Add Y axis
  var y = d3.scaleLinear()
    .domain([d3.min(dy, function(d) { return d; }), d3.max(dy, function(d) { return d; })])
    .range([ height, 0 ]);
  svg.append("g")
    .call(d3.axisLeft(y));


  var lineFunction = d3.line()
      .x(function(d, i){ return i; })
      .y(function(d){ return d; })
      .curve(d3.curveLinear);


  // Add the line
  svg.append("path")
    .datum(dy)
    .attr("fill", "none")
    .attr("stroke", "steelblue")
    .attr("stroke-width", 1.5)
    .attr("d", d3.line()
      .x(function(d, i) { return x(i) })
      .y(function(d) { return y(d) })
      )
}
