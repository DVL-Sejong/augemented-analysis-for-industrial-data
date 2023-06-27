function draw_distance_from_center_chart(dataset){
	console.log("draw_distance_from_center_chart");

	// set the dimensions and margins of the graph
	var margin = {top: 10, right: 30, bottom: 30, left: 60},
		width = w - margin.left - margin.right,
		height = h - margin.top - margin.bottom;

	var dists = [];
	for(var i=0; i<dataset.x.length; i++){
		var _x = dataset.x[i];
		var _y = dataset.y[i];

		var _dis = calcDistance(w/2, h/2, _x, _y);

		dists.push(_dis);
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
	.domain(d3.extent(dists, function(d, i) { return i; }))
	.range([ 0, width ]);

	svg.append("g")
	.attr("transform", "translate(0," + height + ")")
	.call(d3.axisBottom(x));

	// Add Y axis
	var y = d3.scaleLinear()
	.domain([d3.min(dists, function(d) { return d; }), d3.max(dists, function(d) { return d; })])
	.range([ height, 0 ]);
	svg.append("g")
	.call(d3.axisLeft(y));

	// Add the line
	svg.append("path")
	.datum(dists)
	.attr("fill", "none")
	.attr("stroke", "steelblue")
	.attr("stroke-width", 1.5)
	.attr("d", d3.line()
	  .x(function(d, i) { return x(i) })
	  .y(function(d) { return y(d) })
	)

}