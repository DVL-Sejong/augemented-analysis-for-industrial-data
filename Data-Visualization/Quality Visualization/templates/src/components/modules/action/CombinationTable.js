import React from 'react'
import { useFileData } from '../../../contexts/FileDataContext'

export default function CombinationTable({
  data = [],
  onTableRowClick,
  onTableHeadClick,
  canSortColumns,
  selectedColumn,
  selectedKey,
  lengthValues,
  checkedList,
  filterList
}) {
  const { combinationTableSortingInfo } = useFileData()
  const columnKeys = ['idx', 'model', 'DQ issue(s)', 'DQI method(s)'];
  
  const isVisibleLength = (idx) => {
    if(!lengthValues) {
      return true;
    }
    if(lengthValues?.label >= filterList[idx].combination.length) {
      return true;
    }
    return false;
  }
  
  const isVisibleImg = (idx) => {
    const len = filterList[idx].combination.filter((item) => checkedList.includes(item)).length;
    if(len === filterList[idx].combination.length) {
      return true;
    }
    return false;
  }

  return data.length > 0 && checkedList ? (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: `auto auto auto auto`
      }}
    >
      {columnKeys.map((key, i) => {
        const isSortButton = canSortColumns.includes(key)
        const selected = selectedColumn === key
        return (
          <div
            className="grid-th"
            key={key}
            style={{
              cursor: isSortButton ? 'pointer' : 'default',
              background: selected ? '#e1e1e1' : undefined,
              textAlign: 'center',
              fontWeight: 'bold',
              borderRight: i === columnKeys.length - 1 ? 'none' : undefined,
            }}
            onClick={() => isSortButton && onTableHeadClick(key)}
          >
            {key}
            {selected && (
              <>
                &nbsp;
                {combinationTableSortingInfo.isAscending ? (
                  <>&uarr;</>
                ) : (
                  <>&darr;</>
                )}
              </>
            )}
          </div>
        )
      })}
      {data.slice(0, 50).map(({ key, ...others }, rowIdx) => {
        const isLastRow = rowIdx === data.length - 1
        const onClick = () =>
          onTableRowClick(filterList[rowIdx])
        return (
          <React.Fragment key={key}>
            {isVisibleLength(rowIdx) && isVisibleImg(rowIdx) && <>
              <div
                className="grid-td"
                style={{
                  padding: '6px 3px',
                  borderBottom: isLastRow ? 'none' : undefined,
                  backgroundColor: selectedKey === key ? '#eee' : undefined,
                  cursor: 'pointer'
                }}
                onClick={onClick}
              >
                {key}
              </div>
              {Object.values(others).map((chart, colIdx) => (
                <div
                  className="grid-td"
                  key={`${key}${colIdx}`}
                  onClick={onClick}
                  style={{
                    padding: '6px 3px',
                    borderRight: colIdx === columnKeys.length - 2 && 'none',
                    borderBottom: isLastRow ? 'none' : undefined,
                    backgroundColor: selectedKey === key ? '#eee' : undefined,
                    cursor: 'pointer'
                  }}
                >
                  {chart}
                </div>
              ))}
            </>
            }

          </React.Fragment>
        )
      })}
    </div>
  ) : (
    <></>
  )
}
