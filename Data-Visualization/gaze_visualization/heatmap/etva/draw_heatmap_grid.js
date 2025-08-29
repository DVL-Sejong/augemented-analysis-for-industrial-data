function draw_heatmap_grid(dataset){
  console.log("draw_heatmap_grid");

  // Labels of row and columns
  var myGroups = [];
  var myVars = [];

  for(var i=0; i<h; i++){
    myGroups.push(String(i));
  }
  for(var i=0; i<w; i++){
    myVars.push(String(i));
  }
  alert("1");
  var gazePoints = [];
  for(var i=0; i<dataset.x.length; i++){
    var _x = parseInt(Math.floor(dataset.x[i]));
    var _y = parseInt(Math.floor(dataset.y[i]));
    gazePoints.push([_x, _y]);
  }
alert("2");
  var hgMatrix = [];
  for(var i=0; i<h; i++){
    var _row = [];
    for(var j=0; j<w; j++){
      _row.push(0);
    }
    hgMatrix.push(_row);
  }
alert("3");
  for(var i=0; i<gazePoints.length; i++){
    var _c = gazePoints[i][0];
    var _r = gazePoints[i][1];
    hgMatrix[_c][_r]++;
  }
alert("4");
  var hgData = [];
  for(var i=0; i<hgMatrix.length; i++){
    for(var j=0; j<hgMatrix[0].length; j++){
      hgData.push([String(i), String(j), hgMatrix[i][j]]);
    }
  }
alert("5");

  // set the dimensions and margins of the graph
  var margin = {top: 30, right: 30, bottom: 30, left: 30},
    width = w - margin.left - margin.right,
    height = h - margin.top - margin.bottom;

  // append the svg object to the body of the page
  var svg = d3.select("#my_dataviz")
  .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform",
          "translate(" + margin.left + "," + margin.top + ")");
alert("6");


  // Build X scales and axis:
  var x = d3.scaleBand()
    .range([ 0, width ])
    .domain(myGroups)
    .padding(0.01);
  svg.append("g")
    .attr("transform", "translate(0," + height + ")")
    .call(d3.axisBottom(x))

  // Build X scales and axis:
  var y = d3.scaleBand()
    .range([ height, 0 ])
    .domain(myVars)
    .padding(0.01);
  svg.append("g")
    .call(d3.axisLeft(y));

  // Build color scale
  var myColor = d3.scaleLinear()
    .range(["white", "#69b3a2"])
    .domain([1,100])

  svg.selectAll()
      .data(hgData, function(d) {return d[0]+':'+d[1];})
      .enter()
      .append("rect")
      .attr("x", function(d) { return x(d[0]) })
      .attr("y", function(d) { return y(d[1]) })
      .attr("width", x.bandwidth() )
      .attr("height", y.bandwidth() )
      .style("fill", function(d) { return myColor(d[2])} )
alert("7");
  
}
