var map = L.mapbox.map('map', null, { renderer: L.canvas() })
    .setView([37.7938262, -122.41103158], 15.5)
    .addLayer(L.mapbox.styleLayer('Input Your Layer'));

var featureGroup = L.featureGroup().addTo(map);
var drawControl = new L.Control.Draw({
    edit: {
        featureGroup: featureGroup
    },
    draw: {
        polygon: {
            shapeOptions: {
                fill: false
            }
        },
        polyline: false,
        rectangle: {
            shapeOptions: {
                fill: false
            }
        },
        circle: {
            shapeOptions: {
                fill: false
            }
        },
        marker: false
    }
}).addTo(map);

map.on('draw:created', showPolygonArea);
map.on('draw:edited', showPolygonAreaEdited);

function showPolygonAreaEdited(e) {
    e.layers.eachLayer(function(layer) {
        showPolygonArea({ layer: layer });
    });
}

function showPolygonArea(e) {
    var layer = e.layer;
    
    featureGroup.clearLayers();
    featureGroup.addLayer(layer);
}