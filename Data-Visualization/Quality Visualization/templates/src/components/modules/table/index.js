import React from 'react'
import { Box } from '../../Box'
import { useFileData, postData } from '../../../contexts/FileDataContext'
import ColumnSummaryChart from '../../charts/ColumnSummaryChart'

export default function Table() {
  const {
    isEmptyData,
    settingValues,
    tablePointData,
    actionRadioValue,
    checkTableData,
    setCheckTableData,
    tableData,
    setCustomValues,
    customValues,
    columnSummary,
    rowSummary
  } = useFileData()

  const [rColorData, setrColorData] = React.useState({});
  const [gColorData, setgColorData] = React.useState({});
  const [bColorData, setbColorData] = React.useState({});
  const [columnDatas, setcolumnDatas] = React.useState([]);
  const [gridData, setGridData] = React.useState('');
  const [pointData, setPointData] = React.useState();

  const handleTableClick = (key, idx) => {
    if (actionRadioValue === 'customization') {
      setCheckTableData({
        key: key,
        data: idx
      });
      setCustomValues({
        ...customValues,
        'select': key === 'row' ? 'row' : 'column',
        'selectDetail': key === 'row' ? idx-1 : columnDatas[0][idx],
      })
    }
  }

  React.useEffect(() => {
    if (columnDatas.length === 0) {
      return;
    }
    handleTableClick('row', 1)
  }, [columnDatas])

  React.useEffect(() => {
    if (!tablePointData) {
      return;
    }
    setPointData({
      com: {
        rColor: 255,
        gColor: 140,
        bColor: 0,
        points: tablePointData.comPointList,
      },
      acc: {
        rColor: 70,
        gColor: 130,
        bColor: 180,
        points: tablePointData.accPointList,
      },
      con: {
        rColor: 154,
        gColor: 205,
        bColor: 50,
        points: tablePointData.conPointList,
      }
    })
  }, [tablePointData]);

  React.useEffect(() => {
    if (!pointData) {
      return;
    }
    const rColorData = {};
    const gColorData = {};
    const bColorData = {};

    for (let key in pointData) {
      const rColor = pointData[key]['rColor'];
      const gColor = pointData[key]['gColor'];
      const bColor = pointData[key]['bColor'];

      for (let point of pointData[key]['points']) {
        if (rColorData[point.x] === undefined)
          rColorData[point.x] = {};
        rColorData[point.x][point.y + 1] = rColor;

        if (gColorData[point.x] === undefined)
          gColorData[point.x] = {};
        gColorData[point.x][point.y + 1] = gColor;

        if (bColorData[point.x] === undefined)
          bColorData[point.x] = {};
        bColorData[point.x][point.y + 1] = bColor;
      }
    }

    setrColorData(rColorData);
    setgColorData(gColorData);
    setbColorData(bColorData);
  }, [pointData]);

  React.useEffect(() => {
    if (tableData) {
      const codeData = [...tableData.tableData];
      if (codeData[0][0] === 'idx') {
        return;
      }
      codeData[0].unshift('idx');
      for (let i = 1; i < codeData.length; i++) {
        codeData[i].unshift(`${i}`);
      }
      const columnWidth = 740 / (codeData[0].length);
      setGridData(Array.from({ length: codeData[0].length }, () => `${columnWidth}px`).join(" "));
      setcolumnDatas(codeData);
    }
  }, [tableData]);

  const getBgColor = (rowNumber, columnNumber) => {
    if (rowSummary && rowSummary.rowIndex && columnNumber == 0 && rowSummary.rowIndex.includes(rowNumber-1) && rowNumber > 0) {
      return 'rgba(240, 128, 128, 1)'
    }
    if (actionRadioValue === 'customization') {
      if ((checkTableData.key === 'row' && checkTableData.data === rowNumber) || (checkTableData.key === 'col' && checkTableData.data === columnNumber)) {
        if (columnNumber > 0 && rColorData[rowNumber] && rColorData[rowNumber][columnNumber]) {
          return `rgba(${rColorData[rowNumber][columnNumber]}, ${gColorData[rowNumber][columnNumber]}, ${bColorData[rowNumber][columnNumber]}, 1)`;
        }
        return '#eee';
      }
    }
    if (columnNumber > 0 && rColorData[rowNumber] && rColorData[rowNumber][columnNumber]) {
      return `rgba(${rColorData[rowNumber][columnNumber]}, ${gColorData[rowNumber][columnNumber]}, ${bColorData[rowNumber][columnNumber]}, 1)`;
    }
    return undefined;
  }

  return (
    <>
      {!isEmptyData({
        settingValues
      }) && settingValues.model && columnDatas.length > 0 && pointData ? (
        <div style={{ display: 'flex', marginTop: -30, marginLeft: -25 }}>
          <div>
            <div style={{ marginLeft: 57 }}>
              <ColumnSummaryChart data={columnSummary} />
            </div>
            <div style={{
              display: 'flex',
              overflowY: 'scroll',
              marginTop: -30,
              paddingRight: 10,
              width: '792px',
              height: '470px',
            }}>
              <div style={{
                display: 'grid',
                gridTemplateColumns: gridData,
                marginLeft: 30,
              }}>
                {columnDatas && columnDatas.map((columnData, rowIdx) => {
                  const onClickRow = () => handleTableClick('row', rowIdx)
                  return (
                    <React.Fragment key={`col${rowIdx}`}>
                      {columnData.map((data, idx) => {
                        const rowNumber = rowIdx;
                        const columnNumber = idx;
                        const onClickCol = () => handleTableClick('col', idx)
                        return (
                          <React.Fragment key={idx}>
                            {rowNumber === 0
                              ? <div
                                className="grid-th"
                                key={idx}
                                onClick={idx > 0 ? onClickCol : undefined}
                                style={{
                                  cursor: idx > 0 ? 'pointer' : 'default',
                                  background: getBgColor(rowNumber, columnNumber),
                                  textAlign: 'center',
                                  fontWeight: 'bold',
                                  borderRight: columnNumber === columnData.length - 1 && 'none',
                                }}
                              >
                                {data.toString().slice(0, 5)}
                              </div>
                              : <div
                                className="grid-td"
                                onClick={idx === 0 ? onClickRow : undefined}
                                style={{
                                  backgroundColor: getBgColor(rowNumber, columnNumber),
                                  cursor: idx === 0 ? 'pointer' : 'default',
                                  borderRight: columnNumber === columnData.length - 1 && 'none',
                                }}
                                key={idx}
                              >
                                {data.toString().slice(0, 5)}
                              </div>
                            }
                          </React.Fragment>
                        )
                      })}
                    </React.Fragment>
                  )
                }
                )
                }
              </div>
            </div>
          </div>
        </div>
      ) : <Box title="table" />
      }
    </>
  )
}