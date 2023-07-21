import React, { useState, useRef } from 'react';
import ReactHighcharts from 'react-highcharts';
import Modal from 'components/Modal';

function Aside() {
  const [modalVisible, setModalVisible] = useState(false);
  const [modalPosition, setModalPosition] = useState([0, 0]);
  const modal = useRef();

  const onModalClose = () => {
    setModalVisible(false);
  }

  const chartOptions = {
    chart: {
      type: 'bar',
      backgroundColor: 'transparent',
      height: 150
    },
    title: {
      text: ''
    },
    legend: {
      enabled: false
    },
    credits: {
      enabled: false
    },
    plotOptions: {
      series: {
        stacking: 'normal',
        point: {
          events: {
            click: e => {
              setModalVisible(true);
              setModalPosition([e.clientX, e.clientY]);
            }
          }
        }
      }
    },
    tooltip: {
      enabled: false
    },
    xAxis: {
      categories: ['A', 'B', 'C', 'D', 'E'],
      tickLength: 0,
      lineColor: 'transparent',
      labels: {
        style: {
          color: '#777'
        }
      }
    },
    yAxis: {
      min: 0,
      gridLineColor: 'transparent',
      title: {
        enabled: false
      },
      labels: {
        enabled: false
      }
    },
    series: [{
      pointWidth: 20,
      borderRadius: 5,
      borderColor: 'transparent',
      data: [7, 5, 4, 3, 2],
      zoneAxis: 'x',
      zones: [
        { value: 1, color: '#46b985' },
        { value: 2, color: '#f96706' },
        { value: 3, color: '#4e71b1' },
        { value: 4, color: '#d52a93' },
        { value: 5, color: '#9dd728' }
      ]
    }]
  };

  return (
    <>
      <div className="header">Feature type</div>
      <ReactHighcharts config={chartOptions} />
      {modalVisible &&
        <Modal
          ref={modal}
          rightTopX={modalPosition[0]}
          rightTopY={modalPosition[1]}
          onClose={onModalClose}
        />
      }
    </>
  );
}

export default Aside;
