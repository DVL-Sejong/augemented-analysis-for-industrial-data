function draw_gaze_points(dataset){
	console.log("draw_gaze_points");
	
	var gazePoints = [];
	for(var i=0; i<dataset.x.length; i++){
		var _x = dataset.x[i];
		var _y = dataset.y[i];
		gazePoints.push([_x, _y]);
	}
	// append the svg object to the body of the page
	var svg = d3.select("#my_dataviz")
	  .append("svg")
	  .attr("width", w)
	  .attr("height", h);

	var circles = svg.selectAll("circle")
	                .data(gazePoints)
	                .enter()
	                .append("circle")
	circles.attr("cx", function(d){
	  	return d[0];
	})
	.attr("cy", function(d){
		return d[1];
	})
	.attr("r", r);
}
