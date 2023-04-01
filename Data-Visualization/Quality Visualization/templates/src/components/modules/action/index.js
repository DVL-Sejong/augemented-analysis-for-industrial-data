import React, { useMemo } from 'react'
import { Box } from "../../Box"
import { useFileData } from '../../../contexts/FileDataContext'
import Recommend from './Recommend'
import New from './New'
import RangeChart from '../../charts/RangeChart'

export default function Action() {
  const {
    isEmptyData,
    combinationTableData,
    selectedCombinationTableRow,
    setTreeChartData,
    actionRadioValue,
    setActionRadioValue,
    settingValues,
    qualityImpact
  } = useFileData();

  const { combinationData } = combinationTableData
  const handleChangeRadio = (e) => {
    setActionRadioValue(e.target.value);
  }

  React.useEffect(() => {
    let imgData;
    if (!selectedCombinationTableRow) {
      return;
    }
    imgData = [selectedCombinationTableRow];
    const len = Math.max(imgData[0].combination.length, imgData[0].combinationDetail.length);
      const treeData = Array.from({length: len}, () => []);
      for(let i=0;i<len;i++) {
        const [comb, combDetail] = [imgData[0].combination[i], imgData[0].combinationDetail[i]];
        if(!comb) {
          treeData[i].push(combDetail, combDetail);
        } else if(!combDetail) {
          treeData[i].push(comb, comb);
        } else {
          treeData[i].push(comb, combDetail);
        }
      }
      setTreeChartData(treeData);
  }, [selectedCombinationTableRow, setTreeChartData])

  return (
    <Box title="action">
      {!isEmptyData({
        settingValues
      }) && settingValues.model && <>
        <RangeChart data={qualityImpact} />
      </>}
      {!isEmptyData({ combinationData }) && (
        <React.Fragment>
          <div style={{
            display: 'flex',
            height: '20px',
            marginBottom: '5px',
            marginTop: -30,
          }}>
            {['selection', 'customization'].map((item) => (
              <div key={item} style={{ display: 'flex', fontSize: 13, alignItems: 'center', width: '50%' }}>
                <input
                  type='radio'
                  name='radio'
                  value={item}
                  style={{ marginRight: '10px' }}
                  onClick={handleChangeRadio}
                  checked={actionRadioValue === item}
                />
                {item}
              </div>
            ))}
          </div>
          {actionRadioValue === 'selection' && <Recommend />}
          {actionRadioValue === 'customization' && <New />}
        </React.Fragment>
      )}
    </Box>
  )
}