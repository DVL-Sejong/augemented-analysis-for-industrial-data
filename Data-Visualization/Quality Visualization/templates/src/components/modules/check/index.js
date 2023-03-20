import React from 'react'
import Select from 'react-select'
import Title from '../../Title'
import { Box } from '../../Box'
import { useFileData } from '../../../contexts/FileDataContext'
import Legend from '../../charts/Legend'
import DonutChart from '../../charts/DonutChart'
import HeatmapChart from '../../charts/HeatmapChart'
import HistogramChart from '../../charts/HistogramChart'
import CorrelationChart from '../../charts/CorrelationChart'
import RaderChart from '../../charts/RaderChart'
import TreeChart from '../../charts/TreeChart'
import PNBarChart from '../../charts/PNBarChart'
import CheckTable from './CheckTable'

const legendData = [
  { label: 0, text: 'completeness', color: 'darkorange' },
  { label: 1, text: 'outlier', color: 'steelblue' },
  { label: 2, text: 'homogeneity', color: 'yellowgreen' },
  { label: 3, text: 'duplicate', color: 'lightcoral' },
  { label: 4, text: 'correlation', color: 'darkslategray' },
  { label: 5, text: 'relevance', color: 'mediumpurple' },
]

const metricList = [
  { label: 'completeness', visualChart: 'heatmapChart', value: 0 },
  { label: 'outlier', visualChart: 'histogramChart', value: 1 },
  { label: 'homogeneity', visualChart: 'heatmapChart', value: 2 },
  { label: 'duplicate', visualChart: 'duplicate', value: 3 },
  { label: 'correlation', visualChart: 'correlationChart', value: 4 },
  { label: 'relevance', visualChart: 'relevanceChart', value: 5 },
]

const outlierList = [
  { label: 'iqr', value: 0 },
  { label: 'z-score', value: 1 }
];

const correlationList = [
  { label: 'pearson', value: 0 },
  { label: 'kendall', value: 1 },
  { label: 'spearman', value: 2 }
];

export default function Check() {
  const {
    isEmptyData,
    settingValues,
    selectedLegendIdx,
    setSelectedLegendIdx,
    modelSettingData: { columnList },
    donutChartData,
    treeChartData,
    visualizationData,
    updateVisualizationData,
    modelTableData,
    actionRadioValue,
    setTreeChartNode,
    treeChartNode,
    newVisualizationChartData
  } = useFileData()

  const [metricValues, setMetricValues] = React.useState({
    label: "completeness",
    visualChart: "heatmapChart",
    value: 0
  });
  const [visualizationList, setVisualizationList] = React.useState([]);
  const [completenessCell, setCompletenessCell] = React.useState([0, 0]);
  const [completenessQualityIssueCnt, setCompletenessQualityIssueCnt] = React.useState('');
  const [dataList, setDataList] = React.useState();
  const [dataIndex, setDataIndex] = React.useState({
    index: 0,
    top: 0
  });
  const [checkTableData, setCheckTableData] = React.useState([1]);
  const [renderChartData, setRenderChartData] = React.useState();
  const [columnData, setColumnData] = React.useState();
  const [outlierData, setOutlierData] = React.useState();
  const [corrData, setCorrData] = React.useState();

  React.useEffect(() => {
    if (modelTableData) {
      setRenderChartData([{
        key: 1,
        name: modelTableData[1][0],
        data: modelTableData[1].slice(1)
      }])
    }
  }, [modelTableData])

  React.useEffect(() => {
    if (columnList) {
      setColumnData(columnList[0].label)
      setOutlierData(outlierList[0].label)
      setCorrData(correlationList[0].label)
    }
  }, [columnList])

  React.useEffect(() => {
    if (treeChartData) {
      setDataList(treeChartData.map(d => (
        <div style={{ display: 'flex' }}>
          {d.map(imgName => (
            <img
              src={require(`../../icons/${imgName}.png`)}
              alt={''}
              style={{ height: 20, width: 20 }}
            />
          ))}
        </div>
      )));
    }
  }, [treeChartData])

  React.useEffect(() => {
    setMetricValues(metricList[selectedLegendIdx])
  }, [selectedLegendIdx])

  React.useEffect(() => {
    setCheckTableData([1]);
  }, [newVisualizationChartData])

  React.useEffect(() => {
    if (metricValues?.label) {
      setVisualizationList([metricValues.visualChart]);

      let params = {}
      switch (metricValues.visualChart) {
        case "heatmapChart":
          params = {
            rowIdx: completenessCell[0],
            columnIdx: completenessCell[1],
          }
          break

        case "histogramChart":
          params = {
            column: columnData,
            outlier: outlierData,
          }
          break

        case "correlationChart":
          params = {
            method: corrData,
          }
          break

        case "relevanceChart":
          params = {
            column: columnData,
            method: corrData,
          }
          break
      }
      updateVisualizationData(treeChartNode, metricValues.visualChart, metricValues.label, params)
    }
  }, [
    metricValues,
    completenessCell,
    columnData, outlierData,
    corrData,
  ])

  const chartData = (value) => {
    switch (value) {
      case "heatmapChart":
        return <div style={{ display: 'flex', marginLeft: -5 }}>
          <HeatmapChart
            chartName='heatmapChart'
            label={metricValues?.label}
            setCell={setCompletenessCell}
            setQualityIssueCnt={setCompletenessQualityIssueCnt}
            data={visualizationData}
          />
          <div style={{ position: 'relative', right: 10 }}>
            <div style={{ width: 193, height: 85, border: '1px solid #999999', marginTop: 30 }}>
              <div style={{ position: 'absolute', top: 20, left: 10, fontSize: 13, backgroundColor: '#fff', paddingLeft: 5, paddingRight: 5 }}>information</div>
              <div style={{ marginTop: 10 }}>
                <p><strong>record index</strong> {visualizationData.rowIndex}</p>
                <p><strong>feature name</strong> {visualizationData.columnName}</p>
                <p><strong>quality issue cnt</strong> {completenessQualityIssueCnt}</p>
              </div>
            </div>
            <div style={{ width: 193, height: 70, border: '1px solid #999999', marginTop: 15, overflowY: 'auto' }}>
              <div style={{ position: 'absolute', top: 120, left: 10, fontSize: 13, backgroundColor: '#fff', paddingLeft: 5, paddingRight: 5 }}>quality issue</div>
              <div style={{ marginTop: 10 }}>
                {visualizationData.issueList.map(issue =>
                  <p>
                    <strong>record idx&nbsp;{issue[0]}</strong>
                    &nbsp;{issue[1]}
                  </p>
                )}
              </div>
            </div>
          </div>
        </div>

      case "histogramChart":
        return <div style={{ display: 'flex' }}>
          <div>
            <div style={{ display: 'flex', marginTop: 20, marginRight: 10 }}>
              <div style={{
                width: '45%',
                margin: '0 5%'
              }}>
                <Title title="feature" />
                <Select className="select"
                  options={columnList}
                  placeholder={<div>{columnData}</div>}
                  defaultValue={columnData}
                  onChange={v => {
                    setColumnData(v.label)
                  }}
                />
              </div>
              <div style={{ width: '45%' }}>
                <Title title="method" />
                <Select className="select"
                  options={outlierList}
                  placeholder={<div>{outlierData}</div>}
                  defaultValue={outlierData}
                  onChange={v => {
                    setOutlierData(v.label)
                  }}
                />
              </div>
            </div>
            <div style={{ position: 'relative', bottom: 15 }}>
              <HistogramChart
                chartName='histogramChart'
                data={visualizationData}
                method={outlierData}
              />
            </div>
          </div>
          <div>
            <div style={{ width: 168, height: 85, border: '1px solid #999999', marginTop: 30, overflowY: 'auto' }}>
              <div style={{ position: 'absolute', top: 20, left: 270, fontSize: 13, backgroundColor: '#fff', paddingLeft: 5, paddingRight: 5 }}>information</div>
              <div style={{ marginTop: 10 }}>
                <p><strong>outlier</strong> {visualizationData.standard}</p>
                <p><strong>quality issue cnt</strong> {visualizationData.cnt}</p>
              </div>
            </div>
            <div style={{ width: 168, height: 70, border: '1px solid #999999', marginTop: 15, overflowY: 'auto' }}>
              <div style={{ position: 'absolute', top: 120, left: 270, fontSize: 13, backgroundColor: '#fff', paddingLeft: 5, paddingRight: 5 }}>quality issue</div>
              <div style={{ marginTop: 10 }}>
                {visualizationData.issueList.map(issue =>
                  <p>
                    <strong>record idx&nbsp;{issue[0]}</strong>
                    &nbsp;{issue[1]}
                  </p>
                )}
              </div>
            </div>
          </div>
        </div>

      case "duplicate":
        return <>
          <div style={{ position: 'relative', height: 172, left: 10, marginTop: 30, border: '1px solid #999999', width: 418 }}>
            <div style={{ position: 'absolute', top: -10, left: 10, fontSize: 13, backgroundColor: '#fff', paddingLeft: 5, paddingRight: 5 }}>information &amp; quality issue</div>
            <div style={{ marginTop: 10 }}>
              <p><strong>duplicate record cnt</strong> {visualizationData.cnt}</p>
              {visualizationData.issueList &&
                <p><strong>record idx</strong> {visualizationData.issueList}</p>
              }
            </div>
          </div>
        </>

      case "correlationChart":
        return <>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gridTemplateRows: '60px auto', marginTop: 20, marginRight: 10 }}>
            <div style={{ gridRow: '1 / 3', marginTop: -20, marginLeft: -10 }}>
              <CorrelationChart
                chartName='correlationChart'
                data={visualizationData}
              />
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr' }}>
              <div>
                <Title title="method" />
                <Select className="select"
                  options={correlationList}
                  placeholder={<div>{corrData}</div>}
                  defaultValue={corrData}
                  onChange={v => {
                    setCorrData(v.label)
                  }}
                />
              </div>
            </div>
            <div style={{ gridColumn: '2 / 3' }}>
              <div style={{ position: 'relative', height: 112, marginTop: 10, border: '1px solid #999999' }}>
                <div style={{ position: 'absolute', top: -10, left: 2, fontSize: 13, backgroundColor: '#fff', paddingLeft: 5, paddingRight: 5 }}>information &amp; quality issue</div>
                <div style={{ marginTop: 10 }}>
                  <p><strong>high correlation feature cnt</strong> {visualizationData.cnt}</p>
                  {visualizationData.issueList.length > 0 &&
                    <p><strong>high correlation feature name</strong> {visualizationData.issueList[0].join(', ')}</p>
                  }
                </div>
              </div>
            </div>
          </div>
        </>

      case "relevanceChart":
        return <>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gridTemplateRows: '60px auto', marginTop: 20, marginRight: 10 }}>
            <div style={{ gridRow: '1 / 3', marginTop: -10, marginLeft: -10 }}>
              <PNBarChart
                chartName='relevanceChart'
                data={visualizationData}
              />
            </div>
            <div style={{ display: 'flex', marginLeft: -10 }}>
              <div style={{
                width: '45%',
                margin: '0 5%'
              }}>
                <Title title="feature" />
                <Select className="select"
                  options={columnList}
                  placeholder={<div>{columnData}</div>}
                  defaultValue={columnData}
                  onChange={v => {
                    setColumnData(v.label)
                  }}
                />
              </div>
              <div style={{ width: '45%' }}>
                <Title title="method" />
                <Select className="select"
                  options={correlationList}
                  placeholder={<div>{corrData}</div>}
                  defaultValue={corrData}
                  onChange={v => {
                    setCorrData(v.label)
                  }}
                />
              </div>
            </div>
            <div style={{ gridColumn: '2 / 3' }}>
              <div style={{ position: 'relative', height: 112, marginTop: 10, border: '1px solid #999999' }}>
                <div style={{ position: 'absolute', top: -10, left: 2, fontSize: 13, backgroundColor: '#fff', paddingLeft: 5, paddingRight: 5 }}>information &amp; quality issue</div>
                <div style={{ marginTop: 10 }}>
                  <p><strong>low correlation feature cnt</strong> {visualizationData.cnt}</p>
                  {visualizationData.issueList.length > 0 &&
                    <p><strong>low correlation feature name</strong> {visualizationData.issueList.join(', ')}</p>
                  }
                </div>
              </div>
            </div>
          </div>
        </>

      default:
        return
    }
  }

  return (
    <Box title="check">
      {!isEmptyData({
        settingValues
      }) && settingValues.model && <div style={{
        display: 'flex',
        width: '440px'
      }}>
          <div style={{ display: 'flex', height: '275px' }}>
            <div style={{ width: '440px' }}>
              <div style={{
                position: 'absolute',
                top: 75,
                left: 0,
              }}>
                {visualizationList.map((chart, idx) => {
                  return (
                    <div key={idx} className='chart' style={visualizationList.length >= 2 ? { width: 220, height: 230 } : { width: 440, height: 230 }} >
                      {chartData(visualizationData.visualization)}
                    </div>
                  )
                })}
              </div>
            </div>
            <div style={{
              width: '200px',
              display: 'flex',
              position: 'absolute',
              top: 10,
              left: 10,
            }}>
              <Legend
                onLegendClick={setSelectedLegendIdx}
                dataColorInfo={legendData}
              />
              {donutChartData && donutChartData.donutChartData.map((donutData, idx) => (
                <div style={{ margin: '17px 3px 0' }} key={idx}>
                  <DonutChart
                    donutData={donutData}
                  />
                </div>
              ))}
            </div>

            {modelTableData && <CheckTable
              checkTableData={checkTableData}
              setCheckTableData={setCheckTableData}
              data={modelTableData}
              renderChartData={renderChartData}
              setRenderChartData={setRenderChartData} />}

            <div style={{ display: 'flex' }}>
              <div style={{ position: 'relative', top: 10, left: 10 }}>
                {renderChartData && <RaderChart data={renderChartData} />}
              </div>
              <div style={{ overflowY: 'auto', zIndex: 100, marginLeft: 10 }}>
                <TreeChart
                  treeData={dataList}
                  setDataIndex={setDataIndex}
                  actionRadioValue={actionRadioValue}
                  onNodeClick={setTreeChartNode}
                  treeChartNode={treeChartNode} />
              </div>
            </div>
          </div>
          {dataList && dataList.length > 0 && dataIndex &&
            <div style={{
              position: 'relative',
              border: '1px solid #eee',
              borderRadius: '10px',
              top: dataIndex.top - 37,
              right: '15px',
              minWidth: '120px',
              height: '80px',
              backgroundColor: '#fff',
              zIndex: 100,
              fontSize: 12
            }}>
              <div style={{
                backgroundColor: '#eee',
                height: '20px',
                padding: '2px',
                borderRadius: '10px 10px 0px 0px',
              }}>
                step {dataIndex?.index}
              </div>
              <div style={{
                display: 'flex',
                padding: '2px',
                fontSize: 12
              }}>
                <div style={{
                  margin: 2
                }}>method: {dataIndex?.index === 0 && 'none'} </div> {dataIndex?.index !== 0 && dataList[Number.parseInt(dataIndex?.index) - 1].props?.children[0]}</div>
              <div style={{
                display: 'flex',
                padding: '2px',
                fontSize: 12
              }}><div style={{
                margin: 2
              }}>detail method: {dataIndex?.index === 0 && 'none'} </div>{dataIndex?.index !== 0 && dataList[Number.parseInt(dataIndex?.index) - 1].props?.children[1]}</div>
            </div>}
        </div>}
    </Box>
  )
}