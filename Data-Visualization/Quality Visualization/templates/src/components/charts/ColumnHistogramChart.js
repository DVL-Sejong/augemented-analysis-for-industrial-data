import React, { useEffect } from 'react'
import ApexCharts from 'apexcharts'

export default function ColumnHistogramChart(props) {
  const { data } = props
  const d3 = window.d3v4

  useEffect(() => {
    if (!data || !data.histogramSeriesData || !data.histogramCategoryData)
      return

    d3.select('.columnHistogram-wrapper').selectAll('*').remove()

    var options = {
      series: data.histogramSeriesData,
      chart: {
        toolbar: {
          show: false
        },
        animations: {
          enabled: false
        },
        height: 210,
        type: 'bar',
      },
      dataLabels: {
        enabled: false,
      },
      xaxis: {
        categories: data.histogramCategoryData,
      },
      colors: ["#6c757d"]
    };

    var chart = new ApexCharts(document.querySelector(".columnHistogram-wrapper"), options);
    chart.render();

  }, [data])

  return (
    <div className="columnHistogram-wrapper" style={{ position: 'relative', bottom: 50 }} />
  )
}
