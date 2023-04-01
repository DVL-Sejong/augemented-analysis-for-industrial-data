import React from 'react'
import ReactDOM from 'react-dom'
import { Home } from './pages'
import './index.scss'

class Application extends React.Component {
  constructor(props) {
    super(props)
    this.state = {
      page: '/',
    }
  }

  render() {
    return (
      <>
        <div className="container">
          <header></header>
          <section>
            <Home />
          </section>
          <footer></footer>
        </div>
      </>
    )
  }
}

ReactDOM.render(<Application />, document.getElementById('app'))
