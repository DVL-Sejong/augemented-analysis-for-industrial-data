export const mainLayoutStyle = {
  gridGap: '10px',
  gridTemplateColumns: '220px 190px 350px 350px 180px',
  gridTemplateRows: '80px 35px 30px 135px 155px 110px 110px 100px',
  gridTemplateAreas: `
    'dataset check check check check'
    'setting check check check check'
    'setting check check check check'
    'setting check check check check'
    'table table table action change'
    'table table table action change'
    'table table table action change'
    'table table table action change'
  `,
}

export const boxTitles = {
  'dataset': 'data upload',
  'setting': 'setting panel',
  'check': 'data quality assessment',
  'table': 'table',
  'action': 'data quality improvement',
  'change': 'data change'
}

export const PORT = 5000