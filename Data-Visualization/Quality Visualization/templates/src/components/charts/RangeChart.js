import React, { useEffect } from 'react'
import ApexCharts from 'apexcharts'

export default function RangeChart(props) {
  const { data } = props
  const d3 = window.d3v4

  useEffect(() => {
    if (!data)
      return

    d3.select('.range-wrapper').selectAll('*').remove()

    var options = {
      series: [{
        name: 'effect',
        data: data.seriesData
      }],
      chart: {
        toolbar: {
          show: false
        },
        animations: {
          enabled: false
        },
        type: 'bar',
        height: 135
      },
      dataLabels: {
        enabled: false
      },
      stroke: {
        show: true,
        width: 1,
        colors: ['#fff']
      },
      tooltip: {
        shared: true,
        intersect: false
      },
      xaxis: {
        categories: ['mis', 'out', 'inc', 'dup', 'hig', 'low']
      },
      yaxis: {
        labels: {
          show: false
        }
      },
      legend: {
        show: false
      },
      plotOptions: {
        bar: {
          columnWidth: '40%'
        }
      },
      colors: ["#6C757D"]
    };

    var chart = new ApexCharts(document.querySelector(".range-wrapper"), options);
    chart.render();
  
    }, [data])

  return (
    <div className="range-wrapper" style={{ position: 'relative', bottom: 15 }} />
  )
}