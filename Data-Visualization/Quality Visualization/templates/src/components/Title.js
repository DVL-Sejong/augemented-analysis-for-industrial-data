import React from "react"

export default function Title({ title }) {
  return (
    <div
      className="title"
      style={{
        display: "grid",
        alignItems: "center",
        gridTemplateColumns: "1fr auto 1fr",
        gridAutoFlow: "column",
        gridGap: "5px",
      }}
    >
      <div className="divider" />
      {title}
      <div className="divider" />
    </div>
  )
}