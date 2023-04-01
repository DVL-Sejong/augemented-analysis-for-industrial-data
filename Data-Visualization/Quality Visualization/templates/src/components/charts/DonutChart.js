import React, { useEffect, useRef } from 'react'

export default function DonutChart(props) {
  const { data, color, label } = props.donutData
  const svgRef = useRef()
  const d3 = window.d3v4

  useEffect(() => {
    d3.select(`.donut-wrapper-${label}`).selectAll('*').remove()
    
    const width = 48, height = 48, margin = 17
    const radius = Math.min(width, height) / 2 - margin

    const svg = d3
      .select(`.donut-wrapper-${label}`)
      .append('svg')
      .attr('width', width)
      .attr('height', height)
      .append('g')
      .attr('transform', `translate(${(width / 2)}, ${height / 2})`)
      .attr('z-index', 100)

    const colorScale = d3
      .scaleOrdinal()
      .domain(data)
      .range([color, 'lightgray'])

    const pie = d3
      .pie()
      .value(d => d.value)
      .sort(null)

    const calculatedData = pie(d3.entries(data))

    svg
      .selectAll()
      .data(calculatedData)
      .enter()
      .append('path')
      .attr('d', d3
        .arc()
        .innerRadius(width / 2)
        .outerRadius(radius * 1.5)
      )
      .attr('fill', d => colorScale(d.data.key))
  }, [data, color, label, svgRef, d3])

  return (
      <div className={`donut-wrapper-${label}`} />
  )
}
