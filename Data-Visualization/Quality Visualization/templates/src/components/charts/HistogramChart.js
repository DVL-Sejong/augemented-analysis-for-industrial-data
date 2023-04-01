import React, { useEffect } from 'react'
import ApexCharts from 'apexcharts'

export default function HistogramChart(props) {
  const { chartName, data, method } = props
  const d3 = window.d3v4

  useEffect(() => {
    if (!data || data.visualization != chartName)
      return;
    
    d3.select('.histogram-wrapper').selectAll('*').remove()

    if (method === 'z-score') {
      data.lower = -Infinity
      data.upper = data.threshold
    }

    var options = {
      series: data.seriesData,
      chart: {
        toolbar: {
          show: false
        },
        animations: {
          enabled: false
        },
        width: 260,
        height: 175,
        type: 'bar',
      },
      dataLabels: {
        enabled: false,
      },
      xaxis: {
        categories: data.categoryData,
        tickAmount: 10
      },
      colors: [({ value, seriesIndex, dataPointIndex }) => {
        if (data.categoryData[dataPointIndex] <= data.lower || data.categoryData[dataPointIndex] >= data.upper)
          return '#D91212'
        return '#6C757D'
      }]
    };

    var chart = new ApexCharts(document.querySelector(".histogram-wrapper"), options);
    chart.render();

  }, [data])

  return (
    <div className="histogram-wrapper" />
  )
}
