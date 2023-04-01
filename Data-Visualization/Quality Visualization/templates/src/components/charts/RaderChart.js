import React, { useEffect } from 'react'
import ApexCharts from 'apexcharts'

export default function RaderChart(props) {
  const { data } = props
  const d3 = window.d3v4

  useEffect(() => {
    d3.select('.radar-wrapper').selectAll('*').remove()

    var options = {
      series: data,
      
      chart: {
        toolbar: {
          show: false
        },
        animations: {
          enabled: false
        },
        width: 200,
        height: 250,
        type: 'radar',
      },
      xaxis: {
        categories: ['MAE', 'RMSE', 'R2', 'RMSLE', 'MAPE']
      },
      yaxis: {
        show: false
      },
      legend: {
        showForSingleSeries: true,
        width: 195,
        height: 35
      },
      colors: ['#B22222', '#87CEFA', '#C71585', '#2E8B57', '#B8860B'],
    };

    var chart = new ApexCharts(document.querySelector(".radar-wrapper"), options);
    chart.render();
  
    }, [d3, data])

  return (
    <div className="radar-wrapper" />
  )
}