import React, { useEffect } from 'react'
import ApexCharts from 'apexcharts'

export default function ColumnSummaryChart(props) {
  const { data } = props
  const d3 = window.d3v4

  useEffect(() => {
    if (!data)
      return

    d3.select('.column-wrapper').selectAll('*').remove()

    var options = {
      series: data.seriesData,
      chart: {
        toolbar: {
          show: false
        },
        animations: {
          enabled: false
        },
        width: 723,
        height: 80,
        type: 'heatmap',
      },
      dataLabels: {
        enabled: false
      },
      colors: ['#9370DB', '#2F4F4F'],
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
      },
      yaxis: {
        labels: {
          show: false
        }
      },
      tooltip: {
        enabled: false
      }
    };

    var chart = new ApexCharts(document.querySelector(".column-wrapper"), options);
    chart.render();

    }, [data])

  return (
    <div className="column-wrapper" />
  )
}
