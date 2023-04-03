import React from 'react'
import Select from 'react-select'
import Title from '../../Title'
import { useFileData } from '../../../contexts/FileDataContext'
import ColumnHistogramChart from '../../charts/ColumnHistogramChart'
import ColumnBoxplotChart from '../../charts/ColumnBoxplotChart'
import RowScatterChart from '../../charts/RowScatterChart'

export default function Action() {
  const {
    checkTableData,
    customValues,
    treeChartData,
    setCustomValues,
    treeChartNode,
    updateCustomData,
    newVisualizationChartData,
    updateNewVisualizationChartData
  } = useFileData()

  const [actionList, setActionList] = React.useState([]);
  const [actionValues, setActionValues] = React.useState();
  const [buttonActive, setButtonActive] = React.useState(false);

  React.useEffect(() => {
    updateNewVisualizationChartData(treeChartNode, customValues.select, customValues.selectDetail)

    if (treeChartNode === 0) {
      setActionValues();
      setActionList([]);
      return;
    }
    if (customValues.select === 'column') {
      setActionValues();
      const node = treeChartData[treeChartNode - 1][0];
      if (node === 'm') setActionList(['none', 'remove', 'min', 'max', 'mean', 'median'].map((item, idx) => {
        return {
          label: item,
          value: idx
        }
      }));
      else if (node === 'o') setActionList(['none', 'iqr', 'zscore'].map((item, idx) => {
        return {
          label: item,
          value: idx
        }
      }));
      else if (node === 'i') setActionList(['deletion'].map((item, idx) => {
        return {
          label: item,
          value: idx
        }
      }));
      else if (node === 'c' || node === 'r') setActionList(['deletion'].map((item, idx) => {
        return {
          label: item,
          value: idx
        }
      }));
    } else if (customValues.select === 'row') {
      setActionValues();
      setActionList(['deletion'].map((item, idx) => {
        return {
          label: item,
          value: idx
        }
      }));
    }
  }, [customValues, treeChartData, treeChartNode])

  React.useEffect(() => {
    if (
      customValues?.action) {
      setButtonActive(true);
    } else {
      setButtonActive(false);
    }
  }, [customValues])

  const handleChange = (key, value) => {
    if (key === 'action') {
      setActionValues(value);
      setCustomValues({
        ...customValues,
        action: value?.label
      })
    }
  }

  const submitSetting = () => {
    if(treeChartNode) {
      updateCustomData(treeChartNode);
    }
  }

  return (
    <React.Fragment>
      {checkTableData.key === 'col' ?
        <React.Fragment>
          <ColumnBoxplotChart data={newVisualizationChartData} />
          <ColumnHistogramChart data={newVisualizationChartData} />
        </React.Fragment>
        :
        <RowScatterChart data={newVisualizationChartData} />
      }
      <div style={{ display: 'flex', position: 'relative', bottom: (checkTableData.key === 'col' ? 80 : 20) }}>
        <div style={{ width: '47.5%', margin: '0 5%', marginTop: 10 }}>
          <Title title="method" />
          <Select className="select"
            options={actionList}
            placeholder={<div>select</div>}
            defaultValue={actionValues}
            onChange={v => {
              handleChange('action', v)
            }}
          />
        </div>
        <div style={{ width: '47.5%' }} >
          <React.Fragment>
            <button
              disabled={buttonActive ? false : true}
              style={{ width: '50%', height: 30, marginLeft: '25%', marginTop: 30 }}
              onClick={submitSetting}>submit</button>
          </React.Fragment>
        </div>
      </div>
    </React.Fragment>
  )
}