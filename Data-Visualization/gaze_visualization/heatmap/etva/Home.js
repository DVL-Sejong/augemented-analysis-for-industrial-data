import React, { useState } from 'react';
import axios from 'axios';

import Select from 'react-select';

import Scanpath from '../components/Scanpath';

const FLASK_PORT = "5000";
const CLIENT_PORT = "3000";

const SELECT_OPTION_DATASET = [
  { value:'cat2000', label: 'CAT2000' }
];



const Home =()=>{
  const [select_dataset, setSelectDataset] = useState({ value:'default', label: 'SELECT DATASET' })
  const [SELECT_OPTION_DATAFILE, setSelectData] = useState([])
  const [select_datafile, setSelectDatafile] = useState("")
  
  const select_onChanged_dataset=(selectedValue)=>{
    setSelectDataset(selectedValue);

    // load data
    const postData = new FormData();
    postData.set('DATASET', selectedValue.value);
    axios.post(`http://${window.location.hostname}:${FLASK_PORT}/api/loadData`, postData)
      .then(response => {
        console.log(response.data);
      })
      .catch(error =>{
        alert(`ERROR - ${error.message}`);
      });
  }

  
  return (
    <div className='content'>
      {/* col 1*/}
      <div className="inputBoxWrap">
        <div className="page-header">
          <div id="logo"></div><div><h3>Meaning-Salience</h3></div>
        </div>
        <div className="page-selection">
          <Select 
            value={select_dataset}
            options={SELECT_OPTION_DATASET}
            onChange={select_onChanged_dataset}
          />
        </div>
      </div>

      {/* col 2*/}
      <div className="dataVisualizationWrap">
        <div className="section-header">
          <h4> Visualization </h4>
        </div>
        {/* <Scanpath /> */}
      </div>
    </div>
  );
}

export default Home;