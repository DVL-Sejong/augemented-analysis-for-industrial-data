import React, { useEffect } from 'react'

export default function Legend(props) {
  const { dataColorInfo, onLegendClick } = props
  const d3 = window.d3v4

  useEffect(() => {
    d3.select('.legend-wrapper').selectAll('*').remove()

    const width = 100
    const height = 90

    const svg = d3
      .select('.legend-wrapper')
      .append('svg')
      .attr('width', width)
      .attr('height', height)
      .append('g')

    Object.entries(dataColorInfo).forEach(([idx, data], i) => {
      svg
        .append('circle')
        .attr('cx', 5)
        .attr('cy', 5  + i * 15)
        .attr('r', 5)
        .style('fill', data.color)
      svg
        .append('text')
        .attr('x', 12)
        .attr('y', 5 + i * 15)
        .text(data.text)
        .style('cursor', 'pointer')
        .style('font-size', '13px')
        .attr('alignment-baseline', 'middle')
        .on('click', () => {
          onLegendClick(i)
        })
    })
  }, [d3, dataColorInfo, onLegendClick])

  return <div className="legend-wrapper" />
}
