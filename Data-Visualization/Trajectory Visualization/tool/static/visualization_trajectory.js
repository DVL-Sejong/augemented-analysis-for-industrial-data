var smarker = [];
var dmarker = [];
var lines = [];
var dataPath = 'static/data/trajectory.csv'
var visName = 'graph_visualization'

function plot_trajectory() {
    d3.csv(dataPath, function(error, data) {
        if (error) throw error;
        for (var i = 0; i < data.length; i++) {
            var polylineGeometrys = []
            var geometry = data[i].geometry.slice(11,-1).split(',')
            var point = data[i].point.slice(11,-1).split(',')
            for (var j = 0; j < geometry.length; j++) {
                var temp = geometry[j].split(' ')
                polylineGeometrys.push([temp[1], temp[0]])
            }
            var m_polyline = L.polyline(polylineGeometrys, {
                color: 'red',
                weight: 3,
            })
            lines[i] = m_polyline;
            for (var j = 0; j < point.length; j++) {
                var temp = point[j].split(' ')
                if (j == 0) {
                    var markers = L.marker([temp[1], temp[0]], {
                        icon: greenIcon
                    }).bindPopup('Index ' + i + ' trajectory starting point')
                    smarker[i] = markers;
                }
                else if (j == point.length - 1) {
                    var markers = L.marker([temp[1], temp[0]], {
                        icon: redIcon
                    }).bindPopup('Index ' + i + ' trajectory destination point')
                    dmarker[i] = markers;
                }
            }
        }
        try {
            for (var i = 0; i < data.length; i++) {
                map.addLayer(smarker[i])
                map.addLayer(dmarker[i])
                map.addLayer(lines[i])
            }
            console.log('[' + dataPath + ']-[' + visName + ']-[Success]')
        } catch(e) {
            console.log('[' + dataPath + ']-[' + visName + ']-[Failure]')
        }
    })
}