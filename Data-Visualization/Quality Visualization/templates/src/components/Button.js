import React from "react"

export default function Button({ style, children, ...otherProps }) {
  return (
    <button
      style={{
        border: "solid 1px var(--grey-100)",
        borderRadius: "4px",
        minHeight: "38px",
        outline: 0,
        boxSizing: "border-box",
        width: "100%",
        cursor: "pointer",
        ...style,
      }}
      {...otherProps}
    >
      {children}
    </button>
  )
}
