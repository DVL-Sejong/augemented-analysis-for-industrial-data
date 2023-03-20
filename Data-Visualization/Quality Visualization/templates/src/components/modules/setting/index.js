import React from 'react'
import Select from 'react-select'
import Title from '../../Title'
import { Box } from '../../Box'
import { useFileData } from '../../../contexts/FileDataContext'

export default function Setting() {
  const {
    isEmptyData,
    file,
    modelSettingData: { columnList, modelList, evalList },
    settingValues,
    setSettingValues
  } = useFileData()

  const [values, setValues] = React.useState(settingValues);
  const [buttonActive, setButtonActive] = React.useState(false);

  React.useEffect(() => {
    setValues(settingValues);
  }, [setSettingValues, settingValues])

  React.useEffect(() => {
    if (
      values?.column &&
      values?.eval && values.eval.length > 0 &&
      values?.model && values.model.length > 0) {
      setButtonActive(true);
    } else {
      setButtonActive(false);
    }
  }, [values])

  const handleChange = (key, value) => {
    if (key === 'purpose') {
      setSettingValues({
        column: undefined,
        model: undefined,
        eval: undefined,
      });
    } else {
      setValues(prev => ({
        ...prev,
        [key]: value,
      }))
    }
  }

  const submitSetting = () => {
    setSettingValues(values);
  }

  return (
    <Box title="setting" style={{ overflow: 'auto' }}>
      {!isEmptyData({
        file,
      }) && (
          <React.Fragment>
            <div
              style={{
                display: 'grid',
                overflow: 'visible',
                gridGap: '5px',
              }}
            >
                <Title title="target feature" />
                <Select className="select"
                  options={columnList}
                  value={settingValues.column}
                  placeholder={<div>select</div>}
                  defaultValue={values?.column ? values.column : undefined}
                  onChange={v => {
                    handleChange('column', v)
                  }}
                />
                <Title title="machine learning model" />
                <Select className="select"
                  isMulti
                  options={modelList}
                  placeholder={<div>select</div>}
                  defaultValue={values?.model ? values.model : undefined}
                  onChange={v => handleChange('model', v)}
                />
                <Title title="evaluation metric" />
                <Select className="select"
                  isMulti
                  options={evalList}
                  placeholder={<div>select</div>}
                  defaultValue={values?.eval ? values.eval : undefined}
                  onChange={v => handleChange('eval', v)}
                />
            </div>
            <button
              disabled={buttonActive ? false : true}
              style={{ width: '40%', margin: '5px 0 0 60%' }}
              onClick={submitSetting}>submit</button>
          </React.Fragment>
        )}
    </Box>
  )
}
