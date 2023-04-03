import React, { useEffect } from 'react'
import ApexCharts from 'apexcharts'

export default function PNBarChart(props) {
  const { chartName, data } = props
  const d3 = window.d3v4

  useEffect(() => {
    if (!data || data.visualization != chartName)
      return;

    d3.select('.PNbar-wrapper').selectAll('*').remove()

    var options = {
      series: [{
        name: 'correlation',
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
        width: 230,
        height: 215,
      },
      plotOptions: {
        bar: {
          colors: {
            ranges: [{
              from: -1.0,
              to: -0.8,
              color: '#6C757D'
            }, {
              from: -0.8,
              to: 0.8,
              color: '#D91212'
            }, {
              from: 0.8,
              to: 1.0,
              color: '#6C757D'
            }]
          },
        }
      },
      dataLabels: {
        enabled: false,
      },
      yaxis: {
        type: 'category',
        categories: data.categoryData,
        labels: {
          formatter: function (y) {
            return y.toFixed(1) + "%";
          }
        },
        min: -1,
        max: 1
      },
      xaxis: {
        type: 'category',
        categories: data.categoryData,
        labels: {
          show: false
        },
        axisBorder: {
          show: false
        },
        axisTicks: {
          show: false
        }
      }
    };

    var chart = new ApexCharts(document.querySelector(".PNbar-wrapper"), options);
    chart.render();
  
    }, [data])

  return (
    <div className="PNbar-wrapper" />
  )
}