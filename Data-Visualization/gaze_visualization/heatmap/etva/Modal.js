import React, { useEffect, useState, useRef } from 'react';
import axios from 'axios';

function Modal(props) {
  const { onClose, rightTopX, rightTopY } = props;
  const [style, setStyle] = useState({top: 0, left: 0});
  const [stimulusTypes, setStimulusTypes] = useState([]);
  const [eyeMovementDataList, setEyeMovementDataList] = useState({});
  const [selectedStimulusType, setSelectedStimulusType] = useState('');
  const wrapper = useRef();

  const loadStimulusTypes = dataName => {
    axios.get(`http://${window.location.hostname}:5000/static/data/${dataName}/stimulus_types.csv`)
      .then(response => {
        const result = [];
        for (let value of response.data.split('\n')) {
          if (value.length > 0)
           result.push(value.replace('\r', ''));
        }
        setStimulusTypes(result);
      });
  };

  const loadEyeMovementDataList = (dataName) => {
    axios.get(`http://${window.location.hostname}:5000/static/data/${dataName}/eye_movement_data_list.csv`)
      .then(response => {
        const result = {};
        console.log(response.data);
        for (let row of response.data.split('\n')) {
          const [stimulus, eyeMovement] = row.replace('\r', '').split(',');
          if (!result[stimulus])
            result[stimulus] = [];
          result[stimulus].push(eyeMovement);
        }
        console.log(result);
        setEyeMovementDataList(result);
      });
  };

  const onStimulusTypeSelected = e => {
    setSelectedStimulusType(e.currentTarget.value);
  };

  const onSubmit = e => {

  };

  useEffect(() => {
    setStyle({
      top: rightTopY,
      left: rightTopX - wrapper.current.offsetWidth
    });
  }, [rightTopX, rightTopY]);

  useEffect(() => {
    loadStimulusTypes('mit300');
    loadEyeMovementDataList('mit300');
  }, []);

  return (
    <div ref={wrapper} className="modal-wrapper" style={style}>
      <div className="">
        <button className="" onClick={onClose}>x</button>
      </div>
      <div className="">
        <div className="">
          <p className="">Stimulus</p>
          <ul className="">
            {stimulusTypes.map((value, index) =>
              <li key={index}>
                <input
                  type="radio"
                  id={`stimulus-type-${index}`}
                  name="stimulus-type"
                  value={value}
                  onClick={onStimulusTypeSelected}
                />
                <label htmlFor={`stimulus-type-${index}`}>
                  <span>{value}</span>
                </label>
              </li>
            )}
          </ul>
        </div>
        <div className="">
          <p className="">Eye movement</p>
          <ul className="">
            {selectedStimulusType &&
            eyeMovementDataList[selectedStimulusType].map((value, index) =>
              <li key={index}>
                <input
                  type="radio"
                  id={`eye-movement-data-${index}`}
                  name="eye-movement-data"
                  value={value}
                />
                <label htmlFor={`eye-movement-data-${index}`}>
                  <span>{value}</span>
                </label>
              </li>
            )}
          </ul>
        </div>
        <div className="">
          <button className="" onClick={onSubmit}>Apply</button>
        </div>
      </div>
    </div>
  );
}

export default Modal;
