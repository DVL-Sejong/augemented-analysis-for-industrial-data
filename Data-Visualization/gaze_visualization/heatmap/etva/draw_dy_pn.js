function draw_dy_pn_chart(gpoints){
	console.log("draw_dy_pn_chart");
	
	var margin = {top: 20, right: 20, bottom: 50, left: 70};
	var width = 500 - margin.left - margin.right;
	var height = 200 - margin.top - margin.bottom;

	//add svg with margin !important
	//this is svg is actually group
	var svg = d3.select("#my_dataviz").append("svg")
				.attr("width",width+margin.left+margin.right)
				.attr("height",height+margin.top+margin.bottom)
				.append("g")  //add group to leave margin for axis
				.attr("transform","translate("+margin.left+","+margin.top+")");

	var _x_axis = [];
	var dataset = [];
	var prev = 0;
	for(var i=0; i<gpoints.length; i++){
		var cur = gpoints[i][1];
		if(i==0){
		prev = cur;
			continue;
		}
		var _dy = cur-prev;
		dataset.push(_dy);

		prev = cur;
		_x_axis.push(i-1);
	}

	var maxHeight=d3.max(dataset,function(d){return Math.abs(d)});
	var minHeight=d3.min(dataset,function(d){return Math.abs(d)});

	//set y scale
	var yScale = d3.scaleLinear().rangeRound([0,height]).domain([maxHeight,-maxHeight]);//show negative
	//add x axis
	var xScale = d3.scaleBand().rangeRound([0,width]).padding(0.1);//scaleBand is used for  bar chart
	xScale.domain(_x_axis);//value in this array must be unique
	/*if domain is specified, sets the domain to the specified array of values. The first element in domain will be mapped to the first band, the second domain value to the second band, and so on. Domain values are stored internally in a map from stringified value to index; the resulting index is then used to determine the band. Thus, a band scale’s values must be coercible to a string, and the stringified version of the domain value uniquely identifies the corresponding band. If domain is not specified, this method returns the current domain.*/

	var barpadding = 2;
	var bars = svg.selectAll("rect").data(dataset).enter().append("rect");
	bars.attr("x",function(d,i){
			  return xScale(i);//i*(width/dataset.length);
			  })
	.attr("y",function(d){
		if(d<0){
			return height/2;
		}
		else{
			return yScale(d);	
		}
		
	})//for bottom to top
	.attr("width", xScale.bandwidth()/*width/dataset.length-barpadding*/)
	.attr("height", function(d){
		return height/2 -yScale(Math.abs(d));
	});
	bars.attr("fill",function(d){
		if(d>=0){
			return "green";
		}
		else{
			return "orange";
		}
	});

	//add tag to every bar
	var tags = svg.selectAll("text").data(dataset).enter().append("text").text(function(d){
		return d;
	});
	tags.attr("x",function(d,i){
			  return xScale(i)+8;
			  })
	.attr("y",function(d){
		if(d>=0){
			return yScale(d)+12;
		}
		else{
			return height-yScale(Math.abs(d))-2;
		}
	})//for bottom to top
	.attr("fill","white");

	//add x and y axis
	var yAxis = d3.axisLeft(yScale);
	svg.append("g").call(yAxis);


	var xAxis = d3.axisBottom(xScale);/*.tickFormat("");remove tick label*/
	svg.append("g").call(xAxis).attr("transform", "translate(0,"+height/2+")");

	//add label for x axis and y axis
	svg.append("text").text("Delta Y")
		.attr("x",0-height/2)
		.attr("y",0-margin.left)
		.attr("dy","1em")
	  	.style("text-anchor", "middle")
		.attr("transform","rotate(-90)");
	svg.append("text").text("X Label")
		.attr("x",width/2)
		.attr("y",height+margin.bottom)
	  	.style("text-anchor", "middle");
}