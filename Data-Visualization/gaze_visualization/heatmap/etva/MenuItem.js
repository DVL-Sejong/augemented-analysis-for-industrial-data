import React from 'react';

class MenuItem extends React.Component {
  
  onMenuClick = () => {
    const { href, onClick } = this.props;
    onClick(href);
  }

  render() {
    const { href, title, children, currentPage } = this.props;
    const active = (currentPage == href);
    const className = (active) ? "menu-button active" : "menu-button";

    return (
      <div className="menu-item">
        <button className={className} href={href} title={title} onClick={this.onMenuClick}>{children}</button>
      </div>
    );
  }
}

export default MenuItem;
