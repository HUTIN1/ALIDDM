import json

def WriteLandmark(dic_landmark,path):
    true = True
    false = False

    cp_list = []
    model={
                    "id": "1",
                    "label": '',
                    "description": "",
                    "associatedNodeID": "",
                    "position": [],
                    "orientation": [-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 1.0],
                    "selected": true,
                    "locked": false,
                    "visibility": true,
                    "positionStatus": "defined"
                }
    for idx , (landmark, pos) in enumerate(dic_landmark.items()):
        dic = model.copy()
        dic['id'] = f'{idx+1}'
        dic['label'] = f'{landmark}'
        dic['position'] = pos
        cp_list.append(dic)

    true = True
    false = False
    file = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.0.json#",
        "markups": [
        {
            "type": "Fiducial",
            "coordinateSystem": "LPS",
            "locked": false,
            "labelFormat": "%N-%d",
            "controlPoints": cp_list,
            "measurements": [],
            "display": {
                "visibility": false,
                "opacity": 1.0,
                "color": [0.4, 1.0, 0.0],
                "selectedColor": [1.0, 0.5000076295109484, 0.5000076295109484],
                "activeColor": [0.4, 1.0, 0.0],
                "propertiesLabelVisibility": false,
                "pointLabelsVisibility": true,
                "textScale": 3.0,
                "glyphType": "Sphere3D",
                "glyphScale": 1.0,
                "glyphSize": 5.0,
                "useGlyphScale": true,
                "sliceProjection": false,
                "sliceProjectionUseFiducialColor": true,
                "sliceProjectionOutlinedBehindSlicePlane": false,
                "sliceProjectionColor": [1.0, 1.0, 1.0],
                "sliceProjectionOpacity": 0.6,
                "lineThickness": 0.2,
                "lineColorFadingStart": 1.0,
                "lineColorFadingEnd": 10.0,
                "lineColorFadingSaturation": 1.0,
                "lineColorFadingHueOffset": 0.0,
                "handlesInteractive": false,
                "snapMode": "toVisibleSurface"
            }
            }
            ]
            }

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(file, f, ensure_ascii=False, indent=4)
    f.close
