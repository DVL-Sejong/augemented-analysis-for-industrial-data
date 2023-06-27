function draw_duration_chart(dataset){
  console.log("draw_duration_chart");
  // set the dimensions and margins of the graph
  var margin = {top: 20, right: 20, bottom: 30, left: 40},
      width = w - margin.left - margin.right,
      height = h - margin.top - margin.bottom;

  // set the ranges
  var x = d3.scaleBand()
            .range([0, width])
            .padding(0.1);
  var y = d3.scaleLinear()
            .range([height, 0]);
            
  // append the svg object to the body of the page
  // append a 'group' element to 'svg'
  // moves the 'group' element to the top left margin
  var svg = d3.select("#my_dataviz").append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
    .append("g")
      .attr("transform", 
            "translate(" + margin.left + "," + margin.top + ")");

  // Scale the range of the data in the domains
  x.domain(dataset.map(function(d, i) { return i; }));
  y.domain([0, d3.max(dataset, function(d) { return d; })]);

  // append the rectangles for the bar chart
  svg.selectAll(".bar")
      .data(dataset)
    .enter().append("rect")
      .attr("class", "bar")
      .attr("x", function(d, i) { return x(i); })
      .attr("width", x.bandwidth())
      .attr("y", function(d) { return y(d); })
      .attr("height", function(d) { return height - y(d); });

  // add the x Axis
  svg.append("g")
      .attr("transform", "translate(0," + height + ")")
      .call(d3.axisBottom(x));

  // add the y Axis
  svg.append("g")
      .call(d3.axisLeft(y));
}