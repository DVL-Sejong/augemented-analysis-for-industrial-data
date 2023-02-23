function drawCalendar() {
    var chart = undefined,
          titleAnnotation = undefined,
          dateRange = [start, end], //format [new Date('5/17/2008'), new Date('6/19/2008')],
          curDate = dateRange[0],
          picking,
          chartConfig = {
            type: 'calendar month solid',
            margin_bottom: 10,
            width: 300,
            height: 350,
            title_label_verticalAlign: 'middle',
            animation_duration: 150,
            defaultTooltip_enabled: false,
            defaultCultureName: 'en-US',
            box_fill: 'none',
            defaultBox_boxVisible: false,
            calendar: {
              defaultEmptyPoint_legendEntry_visible: false
            },
            annotations: [
              {
                width: 200,
                label: {
                  align: 'center',
                  text: formatTitle(dateRange[0]),
                  style_fontSize: 15
                },
                position: 'top'
              }
            ],
            yAxis_visible: false,
            palette_colorBar_axis_scale_interval: 50,
            palette: {
              colorBar: {
                visible: false
              },
              ranges: {
                min: 750,
                max: 1250,
                interval: 100
              },
              colors: [
                '#d3d3d3',
                '#bdbdbd',
                '#9e9e9e',
                '#7d7d7d',
                '#696969'
              ]
            },
            legend: {
              offset: '0,14',
              position: 'bottom',
              boxVisible: false
            },
            highlights: [
              {
                pattern: { weekday: [0, 6] },
                fill: '#ffedc4',
                outline_color: '#bcbcbc',
                outline_width: 0
              }
            ],
            defaultSeries: {
              shape_innerPadding: 0,
              legendEntry_visible: false,
              opacity: 0.9,
              defaultPoint: {
                legendEntry_visible: false,
                events: {
                  click: pointClick,
                  mouseOver: pointOver
                }
              }
            },
  
            toolbar_items: {
              backward: {
                position: 'top left',
                fill: 'white',
                icon_name: 'material/hardware/keyboard-arrow-left',
                events_click: function() {
                  var d = new Date(curDate);
                  zoomTo(d.setMonth(d.getMonth() - 1));
                }
              },
              forward: {
                position: 'top right',
                fill: 'white',
                icon_name: 'material/hardware/keyboard-arrow-right',
                events_click: function() {
                  var d = new Date(curDate);
                  zoomTo(d.setMonth(d.getMonth() + 1));
                }
              },
              reset: {
                label: {
                  text: 'Clear Dates',
                  color: '#e34e2f',
                  offset: '0,-4'
                },
                margin: 5,
                boxVisible: false,
                position: 'bottom right',
                visible: false,
                events_click: function() {
                  reset();
                }
              }
            }
          };
  
        //Note: HTML elements must be rendered before making references..
        var elementRefs = {
            startDate: document.getElementById('startDate'),
            endDate: document.getElementById('endDate'),
            pickerContainer: document.getElementById('pickerContainer'),
            picker: document.getElementById('picker')
          },
          datesPicked = { start: undefined, end: undefined },
          selectorColor = '#5264cb',
          //Blackout dates loaded from csv file.
          tempData = undefined;
  
        initPicker();
  
        function initPicker() {
          JSC.fetch('filename').then(function(response) {
            if (response.ok) {
              response
                .text()
                .then(function(text) {
                  var parsedData = JSC.parseCsv(text);
  
                  return parsedData.data;
                })
                .then(function(data) {
                  tempData = data;
                  chartConfig.data = tempData.slice(0);
                  chart = JSC.chart('chartDiv', chartConfig);
                });
            } else {
              console.error('Problem loading csv data.');
            }
          });
  
          elementRefs.pickerContainer.addEventListener('click', function(ev) {
            ev.stopPropagation();
          });
          window.addEventListener('click', function() {
            highlightInput();
            closePicker();
          });
        }
  
        /**
         * Date Utils
         */
        function toDate(v) {
          return new Date(v);
        }
        function formatTitle(date) {
          return (
            new Date(date).toLocaleDateString('en', { month: 'long', year: 'numeric' })
          );
        }
        function addOneDay(date) {
          var d = toDate(date);
          return new Date(d.getFullYear(), d.getMonth(), d.getDate() + 1).getTime();
        }
        function formatDate(d) {
          return JSC.formatDate(d, 'd');
        }
  
        /**
         * Picked dates utils
         */
        function reset() {
          datesPicked = { start: undefined, end: undefined };
          picking = 'start';
          highlightInput(elementRefs.startDate);
          updateInputs();
          highlightDates();
          updateAvailability(chart);
          updateReset();
        }
        function pickedDates() {
          var dates = [];
          datesPicked.start && dates.push(datesPicked.start);
          datesPicked.end && dates.push(datesPicked.end);
          return dates;
        }
        function pickedCount() {
          var count = 0;
          datesPicked.end && count++;
          datesPicked.start && count++;
          return count;
        }
  
        /**
         * Chart event handlers
         */
  
        function pointClick() {
          var point = this;
          var clickedDate = point.tokenValue('%date');
          if (picking === 'end') {
            datesPicked.end = clickedDate;
            if (pickedCount() < 2) {
              highlightInput(elementRefs.startDate);
              picking = 'start';
            } else {
              picking = 'none';
              highlightDates();
            }
          } else {
            datesPicked.start = clickedDate;
            highlightInput(elementRefs.endDate);
            highlightDates();
            picking = 'end';
          }
          updateInputs();
          if (picking == 'none') {
            //Clear highligh
  
            highlightInput();
            closePicker();
          }
          updateAvailability(chart);
          updateReset();
  
          updateLineChart();
          //Disable any default point click behavior.
          return false;
        }
        function pointOver() {
          var point = this;
          var pointDate = point.tokenValue('%date');
          //If another point is hovered while only one date is selected, update the highlight
          if (pickedCount() === 1) {
            highlightDates(pointDate);
          }
        }
  
        /**
         * Update chart functions
         */
        //Update possible checkout dates
        function updateAvailability(chartRef) {
          /*Nothing to update*/
        }
  
        function getHighlightCfg(hoverDate) {
          var cfg,
            dateVal,
            dates = pickedDates(),
            isFinalRange = dates.length === 2;
  
          //Include hoverDate in highlight range
          hoverDate && dates.push(hoverDate) && dates.sort();
          cfg = {
            id: 'selection',
            fill: isFinalRange ? selectorColor : '#c3d2ff',
            outline: { color: selectorColor, dashStyle: isFinalRange ? 'solid' : 'dash' }
          };
  
          //Highlight date or range
          dateVal = dates.length === 1 ? dates[0] : { range: dates };
          if (dates.length) {
            cfg.pattern = { date: dateVal };
          }
          return cfg;
        }
        function highlightDates(hoverDate) {
          var hl, //dates = selectedDates,
            updateOptions = hoverDate ? { animation: { duration: 0 } } : {};
          hl = chart.highlights('selection');
          if (pickedCount() || hoverDate) {
            let config = getHighlightCfg(hoverDate);
            if (hl) {
              hl.options(config, updateOptions);
            } else {
              chart.highlights.add(config, updateOptions);
            }
          } else if (hl) {
            hl.remove();
          }
        }
        function zoomTo(d) {
          var d = new Date(d);
          if (d >= dateRange[0] && d <= dateRange[1] && d.getMonth() != curDate.getMonth()) {
            titleAnnotation = titleAnnotation || chart.annotations(0);
            titleAnnotation.options({ label_text: formatTitle(d) }, { animation: { duration: 0 } });
            chart.zoom(d);
            curDate = d;
          }
        }
        function updateReset() {
          chart.uiItems('reset').options({ visible: !!pickedCount() });
        }
  
        /**
         * Update HTML Elements
         */
        function updateInputs() {
          elementRefs.startDate.value = datesPicked.start ? formatDate(datesPicked.start) : 'Start Date';
          elementRefs.endDate.value = datesPicked.end ? formatDate(datesPicked.end) : 'End Date';
        }
        function updateDatePicker(el) {
          if (el === startDate) {
            datesPicked.start = new Date(elementRefs.startDate.value).getTime();
          } else {
            datesPicked.end = new Date(elementRefs.endDate.value).getTime();
          }
          highlightDates();
        }
        function inputClicked(el) {
          picking = el === elementRefs.startDate ? 'start' : 'end';
          highlightInput(el);
          openPicker();
          //Zoom chart to date in clicked text input.
          zoomTo(el.value);
          updateAvailability(chart);
        }
        function highlightInput(el) {
          if (el !== elementRefs.startDate) {
            elementRefs.startDate.classList.remove('activeEl');
          }
          if (el !== elementRefs.endDate) {
            elementRefs.endDate.classList.remove('activeEl');
          }
          if (el) {
            el.classList.add('activeEl');
          }
        }
        function openPicker() {
          elementRefs.picker.style.display = 'block';
          elementRefs.pickerContainer.classList.add('opened');
        }
        function closePicker() {
          elementRefs.picker && (elementRefs.picker.style.display = 'none');
          elementRefs.pickerContainer.classList.remove('opened');
        }
  
        function attachEvents() {
          var startDateElement = document.getElementById('startDate');
          var endDateElement = document.getElementById('endDate');
  
          startDateElement.addEventListener('keyup', function() {
            return updateDatePicker(startDateElement);
          });
          startDateElement.addEventListener('click', function() {
            return inputClicked(startDateElement);
          });
  
          endDateElement.addEventListener('keyup', function() {
            return updateDatePicker(endDateElement);
          });
          endDateElement.addEventListener('click', function() {
            return inputClicked(endDateElement);
          });
        }
  
        attachEvents();
  }