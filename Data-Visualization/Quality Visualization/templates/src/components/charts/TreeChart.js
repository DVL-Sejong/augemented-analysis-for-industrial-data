import React, { useEffect, useRef } from 'react'

export default function TreeChart(props) {
  const { treeData, setDataIndex, onNodeClick, treeChartNode } = props;
  const svgRef = useRef()
  const d3 = window.d3v3
  const nodeGap = 90

  const [data, setData] = React.useState({
    "index": "0",
    "state": "none",
    "name": "start",
    "children": [
      {
        "index": "1",
        "state": "none",
        "name": "mm",
        "children": [
          {
            "index": "2",
            "state": "none",
            "name": "mod",
            "children": []
          }
        ]
      }
    ]
  });

  useEffect(() => {
    if (treeData) {
      const temp = {
        "index": 0,
        "state": "none",
        "name": "start",
        "children": []
      }

      var root = temp;

      for (var i = 1; i <= treeData.length; i++) {
        root.children.push({
          "index": i,
          "state": "state",
          "name": "name",
          "children": [],
        })
        root = root.children[0];
      }
      setData(temp);
    }
    onNodeClick(0)
  }, [treeData])

  useEffect(() => {
    if (!treeData) return;
    
    d3.select('.treeChart-wrapper').selectAll('*').remove()

    var margin = { top: 10, right: 0, bottom: 0, left: 0 },
    width = 50 - margin.left - margin.right,
    height = (treeData.length * nodeGap) + margin.top + margin.bottom

    var i = 0
    var svg = d3
      .select('.treeChart-wrapper')
      .append('svg')
      .attr('width', width + margin.right + margin.left)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')')

    var tree = d3.layout.tree().size([height, width])
    var diagonal = d3.svg.diagonal().projection(function (d) {
      return [d.x, d.y]
    })

    var root = data
    update(root)

    function update() {
      var nodes = tree.nodes(root).reverse()
      var links = tree.links(nodes)

      nodes.forEach(function (d) {
        d.x = 20
        d.y = d.depth * nodeGap
      })

      var node = svg.selectAll('g.node').data(nodes, function (d) {
        return d.id || (d.id = ++i)
      })

      var nodeEnter = node
        .enter()
        .append('g')
        .attr('class', 'node')
        .style('cursor', 'pointer')
        .on("mouseover", function (d) { setDataIndex({
          index: d.index,
          top: this.getBoundingClientRect().top
        }) })
        .on("mouseout", function () { setDataIndex() })
        .on('click', (d) => {
          onNodeClick(d.index)
        })
        .attr('transform', function (d) {
          return 'translate(' + d.x + ',' + d.y + ')'
        })

      nodeEnter
        .append('circle')
        .attr('r', 10)
        .style('fill', function (d) {
          if (d.index === treeChartNode) {
            return '#999999'
          } else {
            return '#cccccc';
          }
        })

      nodeEnter
        .append('text')
        .attr('text-anchor', 'middle')
        .attr('alignment-baseline', 'middle')
        .text(function (d) {
          return d.index
        })

      svg
        .selectAll('path.link')
        .data(links)
        .enter()
        .insert('path', 'g')
        .attr('class', 'link')
        .attr('d', diagonal)
    }
  }, [d3, data, onNodeClick, setDataIndex, svgRef, treeChartNode, treeData])

  return (
    <>
      <div className="treeChart-wrapper" />
    </>
  )
}