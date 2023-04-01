import React, { useEffect } from 'react'
import ApexCharts from 'apexcharts'

export default function BarChart(props) {
  const { data } = props
  const d3 = window.d3v4

  useEffect(() => {
    d3.select('.bar-wrapper').selectAll('*').remove()

    var options = {
      series: data.seriesData,
      chart: {
        toolbar: {
          show: false
        },
        animations: {
          enabled: false
        },
        type: 'bar',
        height: 170
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
        categories: ['record', 'feature', 'instance']
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

    var chart = new ApexCharts(document.querySelector(".bar-wrapper"), options);
    chart.render();
  
    }, [data])

  return (
    <div className="bar-wrapper" />
  )
}