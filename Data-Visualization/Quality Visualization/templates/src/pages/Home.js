import React from 'react'
import { mainLayoutStyle } from '../const'
import Dataset from '../components/modules/dataset'
import Setting from '../components/modules/setting'
import Check from '../components/modules/check'
import Action from '../components/modules/action'
import Change from '../components/modules/change'
import Table from '../components/modules/table'
import { FileDataProvider } from '../contexts/FileDataContext'

const Home = () => {
  return (
    <FileDataProvider>
      <div className="main" style={mainLayoutStyle}>
        <Dataset />
        <Setting />
        <Check />
        <Table />
        <Action />
        <Change />
      </div>
    </FileDataProvider>
  )
}

export default Home
