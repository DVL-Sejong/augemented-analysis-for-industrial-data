import axios from 'axios'
import React, { useCallback, useContext, useEffect, useState } from 'react'
import { PORT } from '../const'

export const fetchData = async route => {
  try {
    const res = await axios.get(
      `http://${window.location.hostname}:${PORT}${route}?${Math.random()}`
    )
    return res.data
  } catch (e) {
    console.log(`ERROR - ${e.message}`)
    return undefined
  }
}

export const postData = async (route, params, config) => {
  try {
    const res = await axios.post(
      `http://${window.location.hostname}:${PORT}${route}?${Math.random()}`,
      params,
      config
    )
    return res.data
  } catch (e) {
    console.log(`ERROR - ${e.message}`)
    return undefined
  }
}

const FileDataContext = React.createContext()

export const FileDataProvider = ({ children }) => {
  const [file, setFile] = useState()  
  const [settingValues, setSettingValues] = useState({
    column: undefined,
    model: undefined,
    eval: undefined
  })
  const [modelSettingData, setModelSettingData] = useState({})
  const [combinationTableData, setCombinationTableData] = useState({})
  const [combinationTableSortingInfo, setCombinationTableSortingInfo] = useState({})
  const [selectedCombinationTableRow, setSelectedCombinationTableRow] = useState()
  const [myCombinationRadioValue, setMyCombinationRadioValue] = React.useState('');
  const [selectedLegendIdx, setSelectedLegendIdx] = useState(0);
  const [treeChartData, setTreeChartData] = useState(0);
  const [donutChartData, setDonutChartData] = useState();
  const [tablePointData, setTablePointData] = useState();
  const [actionRadioValue, setActionRadioValue] = useState('selection');
  const [visualizationData, setVisualizationData] = useState();
  const [modelTableData, setModelTableData] = useState();
  const [changeCntData, setChangeCntData] = useState();
  const [changeDistortData, setChangeDistortData] = useState();
  const [changePerformanceData, setChangePerformanceData] = useState();
  const [checkTableData, setCheckTableData] = useState({
    key: 'row',
    data: 1
  });
  const [treeChartNode, setTreeChartNode] = useState(0);
  const [tableData, setTableData] = useState();
  const [customValues, setCustomValues] = useState({
    select: 'row',
    selectDetail: 1,
    action: undefined
  })
  const [selectedCombinationTableData, setSelectedCombinationTableData] = useState();
  const [columnSummary, setColumnSummary] = useState();
  const [rowSummary, setRowSummary] = useState();
  const [qualityImpact, setQualityImpact] = useState();
  const [newVisualizationChartData, setNewVisualizationChartData] = useState();

  const isEmptyData = data => {
    return Object.values(data).some(value => value === undefined)
  }

  const handleSettingValuesChange = useCallback(async () => {
    if (Object.values(settingValues).some(value => value === undefined)) {
      return
    }
    await postData('/setting', settingValues)
  }, [settingValues])

  useEffect(() => {
    handleSettingValuesChange()
  }, [handleSettingValuesChange])

  const updateSettingList = useCallback(async () => {
    const { columnList, modelList, evalList, dimensionList } = await fetchData('/setting')
    setModelSettingData({
      columnList,
      modelList,
      evalList,
      dimensionList,
    })
  }, [])

  const handleDrop = useCallback(
    async files => {
      setFile(files[0])
      var formData = new FormData()
      const config = {
        header: { 'content-type': 'multipart/form-data' },
      }
      formData.append('file', files[0])
      await postData('/fileUpload', formData, config)
      await updateSettingList()
    },
    [updateSettingList]
  )

  const updateCombinationTable = useCallback(async () => {
    const combinationData = await fetchData('/combination')
    setCombinationTableSortingInfo(prev => ({
      ...prev,
      column: combinationData.inputEvalList[0],
    }))
    setCombinationTableData({ combinationData })
  }, [])
  
  useEffect(() => {
    if (!file || isEmptyData(settingValues)) {
      return
    }
    updateCombinationTable()
  }, [file, updateCombinationTable, settingValues, treeChartNode])

  useEffect(() => {
    if (!file || !settingValues) {
      return
    }
    updateDonutChartData(treeChartNode)
    updateTablePointData(treeChartNode)
    updateModelTableData(treeChartNode)
    updateVisualizationData(treeChartNode, 'heatmapChart', 'completeness', {rowIdx: 0, columnIdx: 0})
    updateChangeCntData(treeChartNode)
    updateChangeDistortData(treeChartNode)
    updateChangePerformanceData(treeChartNode, 'lr')
    updateTableData(treeChartNode)
    updateColumnSummaryData(treeChartNode)
    updateRowSummaryData(treeChartNode)
    updateNewVisualizationChartData(treeChartNode, customValues.select, customValues.selectDetail)
  }, [file, treeChartNode, settingValues])

  useEffect(() => {
    if (!selectedCombinationTableData) {
      return
    }
    updateRecommendData()
  }, [selectedCombinationTableData])

  useEffect(() => {
    updateQualityImpactData()
  }, [])

  const updateRecommendData = async () => {
    const option = {
      ...selectedCombinationTableData,
    }
    const tableData = await postData('/recommend', option);
  }

  const updateDonutChartData = async (fileName) => {
    const option = {
      fileName: fileName 
    }
    const donutData = await postData('/donutChart', option);
    setDonutChartData(donutData);
  }

  const updateTablePointData = async (fileName) => {
    const option = {
      fileName: fileName 
    }
    const tableData = await postData('/tablePoint', option);
    setTablePointData(tableData);
  }

  const updateVisualizationData = async (fileName, visualization, metricValue, params) => {
    const option = {
      fileName: fileName,
      visualization: visualization,
      metricValue: metricValue,
      ...params,
    }
    const visualizationData = await postData('/checkVisualization', option);
    setVisualizationData(visualizationData);
  }

  const updateModelTableData = async (fileName) => {
    const option = {
      fileName: fileName 
    }
    const modelTableData = await postData('/modelTable', option);
    setModelTableData(modelTableData.modelResultData);
  }

  const updateChangeCntData = async (fileName) => {
    const option = {
      fileName: fileName 
    }
    const changeCntData = await postData('/changeCnt', option);
    setChangeCntData(changeCntData);
  }

  const updateChangeDistortData = async (fileName) => {
    const option = {
      fileName: fileName 
    }
    const changeDistortData = await postData('/changeDistort', option);
    setChangeDistortData(changeDistortData);
  }

  const updateChangePerformanceData = async (fileName, modelName) => {
    const option = {
      fileName: fileName,
      modelName: modelName
    }
    const changePerformanceData = await postData('/changePerformance', option);
    setChangePerformanceData(changePerformanceData);
  }

  const updateTableData = async (fileName) => {
    const option = {
      fileName: fileName 
    }
    const tableData = await postData('/tableData', option);
    setTableData(tableData);
  }

  const updateCustomData = async (fileName) => {
    const option = {
      ...customValues,
      fileName: fileName 
    }
    const tableData = await postData('/new', option);
  }

  const updateColumnSummaryData = async (fileName) => {
    const option = {
      fileName: fileName
    }
    const columnSummaryData = await postData('/columnSummary', option)
    setColumnSummary(columnSummaryData)
  }

  const updateRowSummaryData = async (fileName) => {
    const option = {
      fileName: fileName
    }
    const rowSummaryData = await postData('/rowSummary', option);
    setRowSummary(rowSummaryData);
  }

  const updateQualityImpactData = async () => {
    const qualityImpactData = await postData('/impact');
    setQualityImpact(qualityImpactData);
  }

  const updateNewVisualizationChartData = async (fileName, select, selectDetail) => {
    const option = {
      fileName: fileName,
      select: select,
      selectDetail: selectDetail
    }
    const newVisualizationChartData = await postData('/newVisualization', option)
    setNewVisualizationChartData(newVisualizationChartData)
  }

  return (
    <FileDataContext.Provider
      value={{
        file,
        isEmptyData,
        handleDrop,
        modelSettingData,
        combinationTableData,
        combinationTableSortingInfo,
        setCombinationTableSortingInfo,
        selectedCombinationTableRow,
        setSelectedCombinationTableRow,
        setSettingValues,
        settingValues,
        setMyCombinationRadioValue,
        myCombinationRadioValue,
        selectedLegendIdx,
        setSelectedLegendIdx,
        treeChartData,
        setTreeChartData,
        donutChartData,
        tablePointData,
        actionRadioValue,
        setActionRadioValue,
        visualizationData,
        updateVisualizationData,
        modelTableData,
        changeCntData,
        changeDistortData,
        changePerformanceData,
        checkTableData,
        setCheckTableData,
        treeChartNode,
        setTreeChartNode,
        tableData,
        setCustomValues,
        customValues,
        updateCustomData,
        setSelectedCombinationTableData,
        selectedCombinationTableData,
        columnSummary,
        setColumnSummary,
        rowSummary,
        setRowSummary,
        qualityImpact,
        setQualityImpact,
        newVisualizationChartData,
        setNewVisualizationChartData,
        updateNewVisualizationChartData
      }}
    >
      {children}
    </FileDataContext.Provider>
  )
}

export const useFileData = () => useContext(FileDataContext)