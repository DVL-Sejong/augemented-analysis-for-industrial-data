import React from 'react';
import MenuItem from 'components/MenuItem';


class Menu extends React.Component {

  onMenuClick = (page) => {
    this.props.onClick(page);
  }

  render() {
    const { page } = this.props;

    return (
      <ul className="menu-list">
        <li className="menu-item-wrapper">
          <MenuItem href="/data" title="Data" currentPage={page} onClick={this.onMenuClick}>+</MenuItem>
        </li>
        <li className="menu-item-wrapper">
          <MenuItem href="/spatial-variance" title="Spatial Variance" currentPage={page} onClick={this.onMenuClick}>S</MenuItem>
        </li>
        <li className="menu-item-wrapper">
          <MenuItem href="/analysis" title="Analysis view" currentPage={page} onClick={this.onMenuClick}>A</MenuItem>
        </li>
        <li className="menu-item-wrapper">
          <MenuItem href="/visualization" title="Visualization view" currentPage={page} onClick={this.onMenuClick}>V</MenuItem>
        </li>
      </ul>
    );
  }
}

export default Menu;
