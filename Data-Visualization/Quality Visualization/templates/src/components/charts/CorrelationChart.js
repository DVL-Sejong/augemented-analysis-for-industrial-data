import React, { useEffect } from 'react'
import ApexCharts from 'apexcharts'

export default function HeatmapChart(props) {
  const { chartName, data } = props
  const d3 = window.d3v4

  useEffect(() => {
    if (!data || data.visualization != chartName)
      return;

    d3.select('.correlation-wrapper').selectAll('*').remove()

    var options = {
      series: data.seriesData,
      chart: {
        toolbar: {
          show: false
        },
        animations: {
          enabled: false
        },
        width: 230,
        height: 215,
        type: 'heatmap'
      },
      dataLabels: {
        enabled: false
      },
      xaxis: {
        categories: data.categoryData,
        tooltip: {
          enabled: false
        }
      },
      tooltip: {
        x: {
          show: true
        }
      },
      legend: {
        show: false
      },
      plotOptions: {
        heatmap: {
          colorScale: {
            ranges: [
              {
                from: -1.0,
                to: -0.8,
                name: 'negative high',
                color: '#D91212'
              }, {
                from: -0.8,
                to: 0.8,
                name: 'moderate',
                color: '#6C757D'
              }, {
                from: 0.8,
                to: 1.0,
                name: 'positive high',
                color: '#D91212'
              }
            ]
          }
        }
      }
    };

    var chart = new ApexCharts(document.querySelector(".correlation-wrapper"), options);
    chart.render();

  }, [data])

  return (
    <div className="correlation-wrapper" />
  )
}
