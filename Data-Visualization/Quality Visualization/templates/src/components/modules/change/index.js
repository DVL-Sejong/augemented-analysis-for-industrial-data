import React from 'react'
import { Box } from '../../Box'
import { useFileData } from '../../../contexts/FileDataContext'
import BarChart from '../../charts/BarChart'
import ECDFChart from '../../charts/ECDFChart'
import HorizontalBarChart from '../../charts/HorizontalBarChart'
import Title from '../../Title'

export default function Change() {
  const {
    isEmptyData,
    settingValues,
    changeCntData,
    changeDistortData,
    changePerformanceData
  } = useFileData()
  
  return (
    <Box title="change">
      {!isEmptyData({
        settingValues
      }) && settingValues.model && <>
        <Title title="count" />
        <div style={{ position: 'relative', bottom: '15px', height: '125px' }}>
          { changeCntData && <BarChart data={changeCntData} /> }
        </div>
        <Title title="model performance" />
        <div style={{ position: 'relative', bottom: '15px' }}>
          { changePerformanceData && <HorizontalBarChart data={changePerformanceData} /> }
        </div>
        <div style={{ position: 'relative', bottom: '35px' }}>
          <Title title="distort" />
          <div style={{ position: 'relative', bottom: '15px', height: '115px', left: -3 }}>
            { changeDistortData && <ECDFChart data={changeDistortData} /> }
          </div>
        </div>
      </>}
    </Box>
  )
}
