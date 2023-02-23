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
    if ("_latlngs" in layer) { // Case for Rectangle and Polygon
        var bbox = [-122.4259, 37.8117, -122.3813, 37.7680] // bounding box of draw grid cells in [minX, minY, maxX, maxY] order
        var cellSide = 0.2;
        var options = {units: 'kilometers'};
        var hexgrid = turf.squareGrid(bbox, cellSide, options);

        var polyList = [];
        for (var i = 0; i < layer._latlngs[0].length; i++) {
            polyList.push([layer._latlngs[0][i].lng, layer._latlngs[0][i].lat])
        }
        polyList.push([layer._latlngs[0][0].lng, layer._latlngs[0][0].lat])
        
        let poly2 = turf.polygon(test2);


        _.each(hexgrid.features, function(hex){
            var intersection = turf.intersect(poly2, hex.geometry);
            if(intersection)
            {
                hex.geometry = intersection.geometry;
            }else
            {
                hex.geometry ={type: "Polygon", coordinates: []}
            }
        })

        var geoHexgrid = L.geoJSON(hexgrid, {
                style: function (feature) {
                    return {
                        weight: 1,
                        fillColor: false
                    };
                },
            });
        featureGroup.addLayer(L.geoJSON(poly2));
        featureGroup.addLayer(geoHexgrid);
    }
    else { // Case for Circle
        var bbox = [-122.4259, 37.8117, -122.3813, 37.7680] // bounding box of draw grid cells in [minX, minY, maxX, maxY] order
        var center = [layer._latlng.lng, layer._latlng.lat];
        var radius = layer._mRadius;
        var coptions = {steps: 80, units: 'meters'};
        var circle = turf.circle(center, radius, coptions);
        var cellSide = 0.2;
        var options = {units: 'kilometers'};
        var hexgrid = turf.squareGrid(bbox, cellSide, options);

        _.each(hexgrid.features, function(hex){
            var intersection = turf.intersect(circle, hex.geometry);
            if(intersection)
            {
                hex.geometry = intersection.geometry;
            }else
            {
                hex.geometry ={type: "Polygon", coordinates: []}
            }
        })
        var geoHexgrid = L.geoJSON(hexgrid, {
            style: function (feature) {
                return {
                    weight: 1,
                    fillColor: false
                };
            },
        });
    featureGroup.addLayer(L.geoJSON(circle));
    featureGroup.addLayer(geoHexgrid);
    }
}