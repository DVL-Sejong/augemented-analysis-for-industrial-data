import React, { useState } from 'react'
import Button from './Button'

const selectedButtonStyle = {
  borderRadius: '4px 4px 0 0',
  backgroundColor: 'white',
  borderWidth: '1px 1px 0 1px',
}

const unSelectedButtonStyle = {
  borderRadius: '4px 4px 0 0',
  backgroundColor: 'transparent',
  borderWidth: '0 0 1px 0',
}

export default function IndexingButtonBox({ componentInfo, style }) {
  const buttonList = Object.keys(componentInfo)
  const [selectedButton, setSelectedButton] = useState(buttonList[0])
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateRows: '40px 1fr',
        ...style,
      }}
    >
      <div style={{ display: 'flex' }}>
        {buttonList.map(key => (
          <Button
            key={key}
            onClick={() => setSelectedButton(key)}
            style={
              selectedButton === key
                ? selectedButtonStyle
                : unSelectedButtonStyle
            }
          >
            {key}
          </Button>
        ))}
      </div>
      <div
        style={{
          border: '1px solid var(--grey-100)',
          borderTopWidth: 0,
          background: 'white',
          overflow: 'auto',
          padding: '2px',
          boxSizing: 'border-box',
        }}
      >
        {componentInfo[selectedButton]}
      </div>
    </div>
  )
}
