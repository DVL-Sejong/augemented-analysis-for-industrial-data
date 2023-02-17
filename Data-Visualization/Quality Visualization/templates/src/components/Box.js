import React from 'react'
import { boxTitles } from '../const'

export function Box({ title, style, children }) {
  return (
    <div
      style={{
        gridArea: title,
        display: 'grid',
        gridTemplateRows: 'auto 1fr',
      }}
    >
      <div
        style={{
          lineHeight: '20px',
          color: 'rgb(123, 123, 123)',
          backgroundColor: 'rgb(238, 238, 238)',
          fontWeight: 'bold',
          padding: '1px 10px 0',
          borderRadius: '4px',
        }}
      >
        {boxTitles[title]}
      </div>
      <div className="box" style={style}>
        {children}
      </div>
    </div>
  )
}
