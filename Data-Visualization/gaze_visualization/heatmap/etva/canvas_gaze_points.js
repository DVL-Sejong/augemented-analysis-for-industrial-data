function canvas_draw_gazePoints(_c, dataset){
	console.log("canvas_draw_gazePoints");
	
	fixations = [];
	for(var i=0; i<dataset.x.length; i++){
		var _p = {
			"x": +(dataset.x[i]*(_c.width/(w))),
			"y": +(dataset.y[i]*(_c.height/(h))),
			"c": 0
		};
		fixations.push(_p);
	}

	canvas_draw_feature_areas(c);

	var _ctx = _c.getContext("2d");
	var r = 5;
	_ctx.beginPath();
	for(var i=0; i<fixations.length; i++){
		_ctx.fillStyle = "white";
		_ctx.moveTo(fixations[i].x, fixations[i].y);
		_ctx.arc(fixations[i].x, fixations[i].y, r+1, 0, 2*Math.PI, true);
	}
	_ctx.fill();
	_ctx.closePath();

	
	for(var i=0; i<fixations.length; i++){
		_ctx.beginPath();
		if(+fixations[i].c == 1){
			// in-count
			_ctx.fillStyle = "green";
		}else if(+fixations[i].c == 2){
			// out-count
			_ctx.fillStyle = "red";
		}else{
			// none
			_ctx.fillStyle = "black";
		}
		_ctx.moveTo(fixations[i].x, fixations[i].y);
		_ctx.arc(fixations[i].x, fixations[i].y, r, 0, 2*Math.PI, true);
		_ctx.fill();
		_ctx.closePath();
	}
	
}

function canvas_draw_feature_areas(_c){
	console.log("canvas_draw_feature_areas");
	var auc_vals = [];

	if(feat_flag[0] == true){
		var _auc = histogram_color(_c);
		var _v = {
			"Feature": "Color",
			"AUC": _auc
		};
		auc_vals.push(_v);
	}
	if(feat_flag[1] == true){
		var _auc = histogram_intensity(_c);
		var _v = {
			"Feature": "Intensity",
			"AUC": _auc
		};
		auc_vals.push(_v);
	}
	if(feat_flag[2] == true){
		var _auc = histogram_orientation(_c);
		var _v = {
			"Feature": "Orientation",
			"AUC": _auc
		};
		auc_vals.push(_v);
	}
	if(feat_flag[3] == true){
		var _auc = histogram_hog(_c);
		var _v = {
			"Feature": "HOG",
			"AUC": _auc
		};
		auc_vals.push(_v);
	}
	
	console.log(auc_vals);
	darw_auc_barchart(auc_vals);
}

function darw_auc_barchart(_aucs){
	var data = _aucs;

	// set the dimensions and margins of the graph
	var margin = {top: 20, right: 20, bottom: 30, left: 40},
	    width = 960 - margin.left - margin.right,
	    height = 500 - margin.top - margin.bottom;

	// set the ranges
	var x = d3.scaleBand()
	          .range([0, width])
	          .padding(0.1);
	var y = d3.scaleLinear()
	          .range([height, 0]);
	          
	// append the svg object to the body of the page
	// append a 'group' element to 'svg'
	// moves the 'group' element to the top left margin
	var svg = d3.select("#AUC_field").append("svg")
	    .attr("width", width + margin.left + margin.right)
	    .attr("height", height + margin.top + margin.bottom)
	  .append("g")
	    .attr("transform", 
	          "translate(" + margin.left + "," + margin.top + ")");

	
	// format the data
	data.forEach(function(d) {
		d.AUC = +d.AUC;
	});

	// Scale the range of the data in the domains
	x.domain(data.map(function(d) { return d.Feature; }));
	y.domain([0, d3.max(data, function(d) { return d.AUC; })]);

	// append the rectangles for the bar chart
	svg.selectAll(".bar")
	  .data(data)
	.enter().append("rect")
	  .attr("class", "bar")
	  .attr("x", function(d) { return x(d.Feature); })
	  .attr("width", x.bandwidth())
	  .attr("y", function(d) { return y(d.AUC); })
	  .attr("height", function(d) { return height - y(d.AUC); });

	// add the x Axis
	svg.append("g")
	  .attr("transform", "translate(0," + height + ")")
	  .call(d3.axisBottom(x));

	// add the y Axis
	svg.append("g")
	  .call(d3.axisLeft(y));

	
}

function histogram_color(_c){
	var threshold = +document.getElementById("thr_color").value;
	var colorArea = [];
	var auc = 0;
	
	for(var i=0; i<df_color.length; i++){
		if(df_color[i].Val < threshold){
			continue;
		}
		var _a = {
			"r": +df_color[i].Row,
			"c": +df_color[i].Col
		};
		colorArea.push(_a);
	}

	draw_area(_c, colorArea);

	for(var i=0; i<fixations.length; i++){
		var _cv = fixations[i].c;
		var _fx = fixations[i].x;
		var _fy = fixations[i].y;
		var _fc = parseInt(Math.floor(_fx));
		var _fr = parseInt(Math.floor(_fy));
		var _idx = _fr*800+_fc;
		
		if(df_color[_idx].Val >= threshold){
			fixations[i].c = 1;
			auc++;
		}else{
			fixations[i].c = 2;
		}
	}
	auc /= fixations.length;

	return auc;
}

function histogram_intensity(_c){
	var threshold = +document.getElementById("thr_intensity").value;
	var intensityArea = [];
	var auc = 0;
	
	for(var i=0; i<df_intensity.length; i++){
		if(df_intensity[i].Val < threshold){
			continue;
		}
		var _a = {
			"r": +df_intensity[i].Row,
			"c": +df_intensity[i].Col
		};
		intensityArea.push(_a);
	}

	draw_area(_c, intensityArea);

	for(var i=0; i<fixations.length; i++){
		var _cv = fixations[i].c;
		
		var _fx = fixations[i].x;
		var _fy = fixations[i].y;

		var _fc = parseInt(Math.floor(_fx));
		var _fr = parseInt(Math.floor(_fy));
		var _idx = _fr*800+_fc;
		
		if(df_intensity[_idx].Val >= threshold){
			auc++;
			fixations[i].c = 1;
		}else{
			fixations[i].c = 2;
		}
		
	}
	auc /= fixations.length;

	return auc;
}

function histogram_orientation(_c){
	var threshold = +document.getElementById("thr_orientation").value;
	var orientationArea = [];
	var auc = 0;
	
	for(var i=0; i<df_orientation.length; i++){
		if(df_orientation[i].Val < threshold){
			continue;
		}
		var _a = {
			"r": +df_orientation[i].Row,
			"c": +df_orientation[i].Col
		};
		orientationArea.push(_a);
	}

	draw_area(_c, orientationArea);

	for(var i=0; i<fixations.length; i++){
		var _cv = fixations[i].c;
		
		var _fx = fixations[i].x;
		var _fy = fixations[i].y;

		var _fc = parseInt(Math.floor(_fx));
		var _fr = parseInt(Math.floor(_fy));
		var _idx = _fr*800+_fc;
		
		if(df_orientation[_idx].Val >= threshold){
			fixations[i].c = 1;
			auc++;
		}else{
			fixations[i].c = 2;
		}
		
	}
	auc /= fixations.length;

	return auc;
}

function histogram_hog(_c){
	var threshold = +document.getElementById("thr_hog").value;
	var hogArea = [];
	var auc = 0;
	
	for(var i=0; i<df_hog.length; i++){
		if(df_hog[i].Val < threshold){
			continue;
		}
		var _a = {
			"r": +df_hog[i].Row,
			"c": +df_hog[i].Col
		};
		hogArea.push(_a);
	}

	draw_area(_c, hogArea);

	for(var i=0; i<fixations.length; i++){
		var _cv = fixations[i].c;
		
		var _fx = fixations[i].x;
		var _fy = fixations[i].y;

		var _fc = parseInt(Math.floor(_fx));
		var _fr = parseInt(Math.floor(_fy));
		var _idx = _fr*800+_fc;
		
		if(df_hog[_idx].Val >= threshold){
			fixations[i].c = 1;
			auc++;
		}else{
			fixations[i].c = 2;
		}
		
	}
	auc /= fixations.length;

	return auc;
}

function draw_area(_c, _area){
	var _ctx = _c.getContext("2d");

	_ctx.beginPath();
	_ctx.fillStyle = "white";
	_ctx.globalAlpha = 0.5;
	for(var i=0; i<_area.length; i++){
		var _x = _area[i].c;
		var _y = _area[i].r;
		var _w = 1;
		var _h = 1;
		_ctx.rect(_x, _y, _w, _h);
	}
	_ctx.fill();
	_ctx.closePath();

	_ctx.globalAlpha = 1;
}