import React from "react"

export default function Input({ style, ...otherProps }) {
  return (
    <input
      style={{
        border: "solid 1px rgb(204, 204, 204)",
        borderRadius: "4px",
        minHeight: "38px",
        outline: 0,
        boxSizing: "border-box",
        width: "100%",
        padding: "8px",
        ...style,
      }}
      {...otherProps}
    />
  )
}
