function draw_heatmap(dataset){
  console.log("draw_heatmap");

  // set the dimensions and margins of the graph
  var margin = {top: 10, right: 30, bottom: 30, left: 40},
      width = w - margin.left - margin.right,
      height = h - margin.top - margin.bottom;

  var gazePoints = [];
  for(var i=0; i<dataset.x.length; i++){
    var _x = parseInt(Math.floor(dataset.x[i]));
    var _y = parseInt(Math.floor(dataset.y[i]));
    gazePoints.push([_x, _y]);
  }


  // append the svg object to the body of the page
  var svg = d3.select("#my_dataviz")
    .append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
    .append("g")
      .attr("transform",
            "translate(" + margin.left + "," + margin.top + ")");

  // Add X axis
  var x = d3.scaleLinear()
    .domain([0, w])
    .range([ margin.left, width - margin.right ]);
  svg.append("g")
    .attr("transform", "translate(0," + height + ")")
    .call(d3.axisBottom(x));

  // Add Y axis
  var y = d3.scaleLinear()
    .domain([0, h])
    .range([ height - margin.bottom, margin.top ]);
  svg.append("g")
    .call(d3.axisLeft(y));

  // Prepare a color palette
  var color = d3.scaleLinear()
      .domain([0, 1]) // Points per square pixel.
      .range(["white", "#69b3a2"])

  // compute the density data
  var densityData = d3.contourDensity()
    .x(function(d) { return x(d[0]); })
    .y(function(d) { return y(d[1]); })
    .size([width, height])
    .bandwidth(20)
    (gazePoints)

  // show the shape!
  svg.insert("g", "g")
    .selectAll("path")
    .data(densityData)
    .enter().append("path")
      .attr("d", d3.geoPath())
      .attr("fill", function(d) { return color(d.value); })


}
