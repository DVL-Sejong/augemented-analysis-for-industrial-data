import React from 'react'

export default function checkTable(props) {

  const { checkTableData, setCheckTableData, data, renderChartData, setRenderChartData } = props
  const colors = ['#B22222', '#87CEFA', '#C71585', '#2E8B57', '#B8860B'];

  const handleTableClick = (idx) => {
    if(checkTableData.includes(idx)) {
      if(checkTableData.length === 1) {
        return;
      }
      setCheckTableData(checkTableData.filter((i) => i !== idx))
      for(let i=0;i<renderChartData.length;i++) {
        if(renderChartData[i].key === idx) {
          setRenderChartData([...renderChartData.slice(0, i), ...renderChartData.slice(i+1)]);
          break;
        }
      }
    } else {
      if(checkTableData.length === 5) {
        return;
      }
      setCheckTableData([...checkTableData, idx]);
      setRenderChartData([...renderChartData, {
        key: idx,
        name: data[idx][0],
        data: data[idx].slice(1)
      }]);
    }
  }

  return data.length > 0 && (
    <div style={{
      overflowY: 'auto',
      marginTop: '5px',
      width: '370px',
      marginLeft: -10,
    }}>
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'auto auto auto auto auto auto'
      }}>
        {data.map((columnData, rowIdx) => {
          return (
            <React.Fragment key={`col${rowIdx}`}>
              {columnData.map((data, idx) => {
                const onClick = () => handleTableClick(rowIdx)
                return (
                  <React.Fragment key={idx}>
                    {rowIdx === 0
                      ? <div
                        className="grid-th"
                        key={idx}
                        style={{
                          cursor: 'default',
                          background: undefined,
                          textAlign: 'center',
                          fontWeight: 'bold',
                          borderRight: 'none',
                          height: '20px',
                        }}
                      >
                        {data.slice(0, 5)}
                      </div>
                      : <div
                        className="grid-td"
                        onClick={onClick}
                        style={{
                          borderRight: 'none',
                          height: '20px',
                          cursor: 'pointer',
                          backgroundColor: checkTableData.includes(rowIdx) ? '#eee' : undefined,
                        }}
                        key={idx}
                      >
                        {idx === 0
                          ? <div
                            style={{
                              backgroundColor: checkTableData.includes(rowIdx) ? colors[checkTableData.indexOf(rowIdx)] : 'darkgray',
                              padding: '2px 8px',
                              borderRadius: '3px',
                              color: 'white'
                            }}>{data.slice(0, 5)}</div>
                          : data.toString().slice(0, 5)
                        }
                      </div>
                    }

                  </React.Fragment>
                )
              })}
            </React.Fragment>
          )
        })}
      </div>
    </div>
  )
}
