import React, { useEffect } from 'react'
import ApexCharts from 'apexcharts'

export default function HorizontalBarChart(props) {
  const { data } = props
  const d3 = window.d3v4

  useEffect(() => {
    d3.select('.horizontalBar-wrapper').selectAll('*').remove()

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
        height: 180
      },
      plotOptions: {
        bar: {
          horizontal: true,
          dataLabels: {
            position: 'top',
          },
        }
      },
      dataLabels: {
        enabled: false,
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
        categories: ['MAE', 'RMSE', 'R2', 'RMSLE', 'MAPE'],
        tickAmount: 4
      },
      legend: {
        show: false
      },
      colors: ["#CCCCCC", "#6C757D"]
    };

    var chart = new ApexCharts(document.querySelector(".horizontalBar-wrapper"), options);
    chart.render();
  
    }, [data])

  return (
    <div className="horizontalBar-wrapper" style={{ position: 'relative' }} />
  )
}