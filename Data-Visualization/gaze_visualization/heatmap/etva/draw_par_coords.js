function draw_parallel_coordinates(){
	// set the dimensions and margins of the graph
	var margin = {top: 30, right: 50, bottom: 10, left: 50},
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

	// Parse the Data
	d3.csv("http://127.0.0.1:8000/static/data/eye_features/out_3.csv", function(data) {
		console.log(data);
	  // Color scale: give me a specie name, I return a color
	  var color = d3.scaleOrdinal()
	    .domain(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J" ])
	    .range([ "#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99", "#e31a1c", "#fdbf6f", "#ff7f00", "#cab2d6", "#6a3d9a" ])

	  // Here I set the list of dimension manually to control the order of axis:
	  dimensions = ["Center", "Contrast_Color", "Contrast_Ori", "Curvature", "Entropy", "Face_Color", "Horizontal_Line", "Log_Spectrum", "Color", "Intensity", "Orientation", "Skin_Color"]
	  var y = {}

	  // For each dimension, I build a linear scale. I store all in a y object
	  
	  y[dimensions[0]] = d3.scaleLinear()
	      .domain( [10,15] ) // --> Same axis range for each group
	      // --> different axis range for each group --> .domain( [d3.extent(data, function(d) { return +d[name]; })] )
	      .range([height, 0])
      y[dimensions[1]] = d3.scaleLinear()
	      .domain( [0.2,1] ) // --> Same axis range for each group
	      // --> different axis range for each group --> .domain( [d3.extent(data, function(d) { return +d[name]; })] )
	      .range([height, 0])
      y[dimensions[2]] = d3.scaleLinear()
	      .domain( [0,1] ) // --> Same axis range for each group
	      // --> different axis range for each group --> .domain( [d3.extent(data, function(d) { return +d[name]; })] )
	      .range([height, 0])
      y[dimensions[3]] = d3.scaleLinear()
	      .domain( [-50,50] ) // --> Same axis range for each group
	      // --> different axis range for each group --> .domain( [d3.extent(data, function(d) { return +d[name]; })] )
	      .range([height, 0])
      y[dimensions[4]] = d3.scaleLinear()
	      .domain( [0,10] ) // --> Same axis range for each group
	      // --> different axis range for each group --> .domain( [d3.extent(data, function(d) { return +d[name]; })] )
	      .range([height, 0])
      y[dimensions[5]] = d3.scaleLinear()
	      .domain( [0,765] ) // --> Same axis range for each group
	      // --> different axis range for each group --> .domain( [d3.extent(data, function(d) { return +d[name]; })] )
	      .range([height, 0])
      y[dimensions[6]] = d3.scaleLinear()
	      .domain( [0,765] ) // --> Same axis range for each group
	      // --> different axis range for each group --> .domain( [d3.extent(data, function(d) { return +d[name]; })] )
	      .range([height, 0])
      y[dimensions[7]] = d3.scaleLinear()
	      .domain( [0,15] ) // --> Same axis range for each group
	      // --> different axis range for each group --> .domain( [d3.extent(data, function(d) { return +d[name]; })] )
	      .range([height, 0])
      y[dimensions[8]] = d3.scaleLinear()
	      .domain( [0,1] ) // --> Same axis range for each group
	      // --> different axis range for each group --> .domain( [d3.extent(data, function(d) { return +d[name]; })] )
	      .range([height, 0])
      y[dimensions[9]] = d3.scaleLinear()
	      .domain( [0,1] ) // --> Same axis range for each group
	      // --> different axis range for each group --> .domain( [d3.extent(data, function(d) { return +d[name]; })] )
	      .range([height, 0])
      y[dimensions[10]] = d3.scaleLinear()
	      .domain( [0,1] ) // --> Same axis range for each group
	      // --> different axis range for each group --> .domain( [d3.extent(data, function(d) { return +d[name]; })] )
	      .range([height, 0])
      y[dimensions[11]] = d3.scaleLinear()
	      .domain( [0,765] ) // --> Same axis range for each group
	      // --> different axis range for each group --> .domain( [d3.extent(data, function(d) { return +d[name]; })] )
	      .range([height, 0])

/*
	  for (i in dimensions) {
	    name = dimensions[i]
	    y[name] = d3.scaleLinear()
	     .domain( [-100,100] ) // --> Same axis range for each group
	     //.domain( [d3.extent(data, function(d) { return +d[name]; })] )
	      // --> different axis range for each group --> .domain( [d3.extent(data, function(d) { return +d[name]; })] )
	      .range([height, 0])
	  }
*/
	  // Build the X scale -> it find the best position for each Y axis
	  x = d3.scalePoint()
	    .range([0, width])
	    .domain(dimensions);

	  // Highlight the specie that is hovered
	  var highlight = function(d){

	    selected_specie = d.Id

	    // first every group turns grey
	    d3.selectAll(".line")
	      .transition().duration(200)
	      .style("stroke", "lightgrey")
	      .style("opacity", "0.2")
	    // Second the hovered specie takes its color
	    d3.selectAll("." + selected_specie)
	      .transition().duration(200)
	      .style("stroke", color(selected_specie))
	      .style("opacity", "1")
	  }

	  // Unhighlight
	  var doNotHighlight = function(d){
	    d3.selectAll(".line")
	      .transition().duration(200).delay(1000)
	      .style("stroke", function(d){ return( color(d.Id))} )
	      .style("opacity", "1")
	  }

	  // The path function take a row of the csv as input, and return x and y coordinates of the line to draw for this raw.
	  function path(d) {
	      return d3.line()(dimensions.map(function(p) { return [x(p), y[p](d[p])]; }));
	  }
	  // Draw the lines
	  svg
	    .selectAll("myPath")
	    .data(data)
	    .enter()
	    .append("path")
	      .attr("class", function (d) { return "line " + d.Id } ) // 2 class for each line: 'line' and the group name
	      .attr("d",  path)
	      .style("fill", "none" )
	      .style("stroke", function(d){ return( color(d.Id))} )
	      .style("opacity", 0.5)
	      .on("mouseover", highlight)
	      .on("mouseleave", doNotHighlight )

	  // Draw the axis:
	  svg.selectAll("myAxis")
	    // For each dimension of the dataset I add a 'g' element:
	    .data(dimensions).enter()
	    .append("g")
	    .attr("class", "axis")
	    // I translate this element to its right position on the x axis
	    .attr("transform", function(d) { return "translate(" + x(d) + ")"; })
	    // And I build the axis with the call function
	    .each(function(d) { d3.select(this).call(d3.axisLeft().ticks(5).scale(y[d])); })
	    // Add axis title
	    .append("text")
	      .style("text-anchor", "middle")
	      .attr("y", -9)
	      .text(function(d) { return d; })
	      .style("fill", "black")

	})
}

/*

function draw_parallel_coordinates(){
	// set the dimensions and margins of the graph
	var margin = {top: 30, right: 50, bottom: 10, left: 50},
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

	// Parse the Data
	d3.csv("http://127.0.0.1:8000/static/data/eye_features/sti_t_z.csv", function(data) {
		console.log(data);
	  // Color scale: give me a specie name, I return a color
	  var color = d3.scaleOrdinal()
	    .domain(["U0121_1RTE", "U0122_1RTE", "U0131_1RTE" ])
	    .range([ "#440154ff", "#21908dff", "#fde725ff"])

	  // Here I set the list of dimension manually to control the order of axis:
	  dimensions = ["A", "B", "C", "D"]

	  // For each dimension, I build a linear scale. I store all in a y object
	  var y = {}
	  for (i in dimensions) {
	    name = dimensions[i]
	    y[name] = d3.scaleLinear()
	      .domain( [-8,8] ) // --> Same axis range for each group
	      // --> different axis range for each group --> .domain( [d3.extent(data, function(d) { return +d[name]; })] )
	      .range([height, 0])
	  }

	  // Build the X scale -> it find the best position for each Y axis
	  x = d3.scalePoint()
	    .range([0, width])
	    .domain(dimensions);

	  // Highlight the specie that is hovered
	  var highlight = function(d){

	    selected_specie = d.Species

	    // first every group turns grey
	    d3.selectAll(".line")
	      .transition().duration(200)
	      .style("stroke", "lightgrey")
	      .style("opacity", "0.2")
	    // Second the hovered specie takes its color
	    d3.selectAll("." + selected_specie)
	      .transition().duration(200)
	      .style("stroke", color(selected_specie))
	      .style("opacity", "1")
	  }

	  // Unhighlight
	  var doNotHighlight = function(d){
	    d3.selectAll(".line")
	      .transition().duration(200).delay(1000)
	      .style("stroke", function(d){ return( color(d.Species))} )
	      .style("opacity", "1")
	  }

	  // The path function take a row of the csv as input, and return x and y coordinates of the line to draw for this raw.
	  function path(d) {
	      return d3.line()(dimensions.map(function(p) { return [x(p), y[p](d[p])]; }));
	  }
	  console.log(data);
	  // Draw the lines
	  svg
	    .selectAll("myPath")
	    .data(data)
	    .enter()
	    .append("path")
	      .attr("class", function (d) { return "line " + d.Species } ) // 2 class for each line: 'line' and the group name
	      .attr("d",  path)
	      .style("fill", "none" )
	      .style("stroke", function(d){ return( color(d.Species))} )
	      .style("opacity", 0.5)
	      .on("mouseover", highlight)
	      .on("mouseleave", doNotHighlight )

	  // Draw the axis:
	  svg.selectAll("myAxis")
	    // For each dimension of the dataset I add a 'g' element:
	    .data(dimensions).enter()
	    .append("g")
	    .attr("class", "axis")
	    // I translate this element to its right position on the x axis
	    .attr("transform", function(d) { return "translate(" + x(d) + ")"; })
	    // And I build the axis with the call function
	    .each(function(d) { d3.select(this).call(d3.axisLeft().ticks(5).scale(y[d])); })
	    // Add axis title
	    .append("text")
	      .style("text-anchor", "middle")
	      .attr("y", -9)
	      .text(function(d) { return d; })
	      .style("fill", "black")

	})
}
*/