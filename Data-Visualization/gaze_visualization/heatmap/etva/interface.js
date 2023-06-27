let stiGrid = [];
let AOIarray = [];
let AOIduration = [];
let selectedAppendCell = [];
let AOIcolorBrewer_12class_set3 = ["#e31a1c", "#a6cee3", "#33a02c", "#ff7f00", "#b15928", "#6a3d9a", "#fdbf6f", "#b2df8a", "#1f78b4", "#fb9a99", "#cab2d6", "#ffff99"];
let SELECTED_AOI = -999;
let selectedDeleteCell = [];
let featureCounting = 0;

let DIV_WIDTH = 350;
let DIV_HEIGHT = 350;
let STIMULUS_WIDTH = 800;
let STIMULUS_HEIGHT = 600;
let STIMULUS_OPACITY = 0.5;

let NAME_FEATURES = ["Color", "Intensity", "Orientation"];

// for interaction
let FLAG_DRAW_GAZE_ON_STIMULUS = 0;


let temp = [];
let __t = [];
let _t = {
	'x': 412.4542124669535, 'y': 306.7882471890983, 'duration': 47, 'clss': 0
};
__t.push(_t);
_t = {
	'x': 410.989011, 'y': 311.8541033, 'duration': 1, 'clss': 3
};
__t.push(_t);
_t = {
	'x': 537.7335649980503, 'y': 301.41225397938206, 'duration': 16, 'clss': 1
};
__t.push(_t);
_t = {
	'x': 692.3076923, 'y': 284.4984802, 'duration': 1, 'clss': 2
};
__t.push(_t);
_t = {
	'x': 722.469400633399, 'y': 268.30750977413163, 'duration': 49, 'clss': 0
};
__t.push(_t);
_t = {
	'x': 740.6593407, 'y': 260.7902736, 'duration': 1, 'clss': 3
};
__t.push(_t);
_t = {
	'x': 750.7455507393778, 'y': 245.07076847017288, 'duration': 9, 'clss': 1
};
__t.push(_t);
_t = {
	'x': 756.043956, 'y': 213.3738602, 'duration': 1, 'clss': 2
};
__t.push(_t);
temp.push(__t);

let rawGazeData = [
	[410.9890109890109, 306.3829787234042],
	[410.9890109890109, 306.3829787234042],
	[413.18681318681325, 308.2066869300912],
	[413.18681318681325, 308.2066869300912],
	[413.18681318681325, 308.2066869300912],
	[413.18681318681325, 310.0303951367781],
	[413.18681318681325, 311.85410334346506],
	[413.18681318681325, 310.0303951367781],
	[413.18681318681325, 310.0303951367781],
	[413.18681318681325, 310.0303951367781],
	[410.9890109890109, 310.0303951367781],
	[410.9890109890109, 308.2066869300912],
	[410.9890109890109, 308.2066869300912],
	[410.9890109890109, 308.2066869300912],
	[408.79120879120876, 310.0303951367781],
	[408.79120879120876, 310.0303951367781],
	[408.79120879120876, 308.2066869300912],
	[408.79120879120876, 308.2066869300912],
	[408.79120879120876, 308.2066869300912],
	[408.79120879120876, 310.0303951367781],
	[408.79120879120876, 310.0303951367781],
	[410.9890109890109, 310.0303951367781],
	[410.9890109890109, 310.0303951367781],
	[410.9890109890109, 310.0303951367781],
	[413.18681318681325, 311.85410334346506],
	[413.18681318681325, 311.85410334346506],
	[410.9890109890109, 311.85410334346506],
	[413.18681318681325, 311.85410334346506],
	[413.18681318681325, 310.0303951367781],
	[410.9890109890109, 310.0303951367781],
	[410.9890109890109, 308.2066869300912],
	[410.9890109890109, 308.2066869300912],
	[410.9890109890109, 306.3829787234042],
	[410.9890109890109, 306.3829787234042],
	[413.18681318681325, 306.3829787234042],
	[410.9890109890109, 304.55927051671733],
	[410.9890109890109, 304.55927051671733],
	[410.9890109890109, 304.55927051671733],
	[413.18681318681325, 306.3829787234042],
	[413.18681318681325, 306.3829787234042],
	[413.18681318681325, 308.2066869300912],
	[410.9890109890109, 308.2066869300912],
	[410.9890109890109, 308.2066869300912],
	[410.9890109890109, 310.0303951367781],
	[410.9890109890109, 310.0303951367781],
	[410.9890109890109, 311.85410334346506],
	[410.9890109890109, 311.85410334346506],
	[410.9890109890109, 311.85410334346506],
	[410.9890109890109, 311.85410334346506],
	[410.9890109890109, 311.85410334346506],
	[408.79120879120876, 311.85410334346506],
	[413.18681318681325, 313.67781155015194],
	[424.1758241758242, 313.67781155015194],
	[441.7582417582418, 311.85410334346506],
	[470.32967032967036, 310.0303951367781],
	[503.2967032967033, 306.3829787234042],
	[536.2637362637362, 304.55927051671733],
	[571.4285714285714, 300.9118541033435],
	[606.5934065934066, 297.2644376899696],
	[639.5604395604396, 291.79331306990883],
	[661.5384615384615, 288.14589665653494],
	[679.1208791208791, 286.322188449848],
	[685.7142857142857, 284.4984802431611],
	[690.1098901098901, 284.4984802431611],
	[692.3076923076924, 284.4984802431611],
	[694.5054945054945, 284.4984802431611],
	[696.7032967032967, 284.4984802431611],
	[698.9010989010989, 286.322188449848],
	[698.9010989010989, 284.4984802431611],
	[698.9010989010989, 282.67477203647417],
	[698.9010989010989, 280.8510638297872],
	[698.9010989010989, 279.0273556231003],
	[698.9010989010989, 275.37993920972644],
	[698.9010989010989, 273.5562310030395],
	[696.7032967032967, 271.73252279635255],
	[696.7032967032967, 271.73252279635255],
	[694.5054945054945, 271.73252279635255],
	[696.7032967032967, 273.5562310030395],
	[696.7032967032967, 275.37993920972644],
	[696.7032967032967, 277.2036474164134],
	[696.7032967032967, 277.2036474164134],
	[696.7032967032967, 277.2036474164134],
	[694.5054945054945, 279.0273556231003],
	[696.7032967032967, 279.0273556231003],
	[696.7032967032967, 279.0273556231003],
	[696.7032967032967, 279.0273556231003],
	[696.7032967032967, 277.2036474164134],
	[696.7032967032967, 275.37993920972644],
	[696.7032967032967, 275.37993920972644],
	[696.7032967032967, 275.37993920972644],
	[701.0989010989011, 275.37993920972644],
	[709.8901098901099, 273.5562310030395],
	[716.4835164835165, 269.90881458966567],
	[723.0769230769231, 266.2613981762918],
	[725.2747252747253, 264.43768996960483],
	[729.6703296703297, 260.790273556231],
	[731.8681318681319, 258.96656534954406],
	[738.4615384615385, 255.31914893617022],
	[738.4615384615385, 253.49544072948328],
	[736.2637362637362, 253.49544072948328],
	[734.065934065934, 255.31914893617022],
	[734.065934065934, 255.31914893617022],
	[736.2637362637362, 257.1428571428571],
	[738.4615384615385, 257.1428571428571],
	[740.6593406593407, 258.96656534954406],
	[742.8571428571429, 257.1428571428571],
	[742.8571428571429, 258.96656534954406],
	[742.8571428571429, 258.96656534954406],
	[742.8571428571429, 258.96656534954406],
	[742.8571428571429, 260.790273556231],
	[742.8571428571429, 260.790273556231],
	[740.6593406593407, 258.96656534954406],
	[740.6593406593407, 258.96656534954406],
	[740.6593406593407, 260.790273556231],
	[740.6593406593407, 260.790273556231],
	[740.6593406593407, 260.790273556231],
	[742.8571428571429, 258.96656534954406],
	[747.2527472527472, 257.1428571428571],
	[753.8461538461538, 251.67173252279636],
	[756.043956043956, 244.37689969604864],
	[756.043956043956, 235.258358662614],
	[756.043956043956, 229.7872340425532],
	[756.043956043956, 224.31610942249242],
	[758.2417582417582, 218.8449848024316],
	[756.043956043956, 213.37386018237083],
	[756.043956043956, 211.5501519756839],
	[753.8461538461538, 211.5501519756839],
	[753.8461538461538, 211.5501519756839],
	[756.043956043956, 213.37386018237083],
	[758.2417582417582, 213.37386018237083],
	[760.4395604395604, 213.37386018237083],
	[760.4395604395604, 213.37386018237083],
	[762.6373626373627, 215.19756838905775],
	[762.6373626373627, 215.19756838905775],
	[760.4395604395604, 217.0212765957447],
	[760.4395604395604, 220.66869300911856],
	[762.6373626373627, 220.66869300911856],
	[762.6373626373627, 222.49240121580547],
	[762.6373626373627, 222.49240121580547],
	[762.6373626373627, 222.49240121580547],
	[760.4395604395604, 220.66869300911856],
	[760.4395604395604, 220.66869300911856],
	[760.4395604395604, 220.66869300911856],
	[758.2417582417582, 220.66869300911856]
	];

let saliencyFeatures =[
	[0.16052112,0.043206494,0.1323152347029887],
	[0.16052112,0.043206494,0.1323152347029887],
	[0.16282108,0.075270265,0.12930980487364843],
	[0.16282108,0.075270265,0.12930980487364843],
	[0.16282108,0.075270265,0.12930980487364843],
	[0.16290806,0.07658491,0.1267733772562356],
	[0.16180123,0.051869046,0.12447649375210154],
	[0.16290806,0.07658491,0.1267733772562356],
	[0.16290806,0.07658491,0.1267733772562356],
	[0.16290806,0.07658491,0.1267733772562356],
	[0.15678275,0.06431136,0.12853578679458807],
	[0.15746929,0.05784232,0.12906675318675775],
	[0.15746929,0.05784232,0.12906675318675775],
	[0.15746929,0.05784232,0.12906675318675775],
	[0.15129656,0.055002317,0.13048385849532101],
	[0.15129656,0.055002317,0.13048385849532101],
	[0.1522229,0.04385593,0.12883006348650494],
	[0.1522229,0.04385593,0.12883006348650494],
	[0.1522229,0.04385593,0.12883006348650494],
	[0.15129656,0.055002317,0.13048385849532101],
	[0.15129656,0.055002317,0.13048385849532101],
	[0.15678275,0.06431136,0.12853578679458807],
	[0.15678275,0.06431136,0.12853578679458807],
	[0.15678275,0.06431136,0.12853578679458807],
	[0.16180123,0.051869046,0.12447649375210154],
	[0.16180123,0.051869046,0.12447649375210154],
	[0.15571661,0.040180184,0.12806799470188582],
	[0.16180123,0.051869046,0.12447649375210154],
	[0.16290806,0.07658491,0.1267733772562356],
	[0.15678275,0.06431136,0.12853578679458807],
	[0.15746929,0.05784232,0.12906675318675775],
	[0.15746929,0.05784232,0.12906675318675775],
	[0.16052112,0.043206494,0.1323152347029887],
	[0.16052112,0.043206494,0.1323152347029887],
	[0.16553298,0.066077046,0.134898388618601],
	[0.16613753,0.039621525,0.1353297961223818],
	[0.16613753,0.039621525,0.1353297961223818],
	[0.16613753,0.039621525,0.1353297961223818],
	[0.16553298,0.066077046,0.134898388618601],
	[0.16553298,0.066077046,0.134898388618601],
	[0.16282108,0.075270265,0.12930980487364843],
	[0.15746929,0.05784232,0.12906675318675775],
	[0.15746929,0.05784232,0.12906675318675775],
	[0.15678275,0.06431136,0.12853578679458807],
	[0.15678275,0.06431136,0.12853578679458807],
	[0.15571661,0.040180184,0.12806799470188582],
	[0.15571661,0.040180184,0.12806799470188582],
	[0.15571661,0.040180184,0.12806799470188582],
	[0.15571661,0.040180184,0.12806799470188582],
	[0.15571661,0.040180184,0.12806799470188582],
	[0.15025252,0.03500809,0.131473541086004],
	[0.16065125,0.038762543,0.12267556672640245],
	[0.1639236,0.08823463,0.12298910912397565],
	[0.14563946,0.08131657,0.09702650620771064],
	[0.13930583,0.027225655,0.10219947982119454],
	[0.16939095,0.051507078,0.14496766410991133],
	[0.15462878,0.18776779,0.26750068117883435],
	[0.16546786,0.3357673,0.33643744181511653],
	[0.2077326,0.51447546,0.5281387371001204],
	[0.22464232,0.19187912,0.3739646668277283],
	[0.24406457,0.07603589,0.17465821271604623],
	[0.1909773,0.30653575,0.23763981968408954],
	[0.22845909,0.17031114,0.13918199295558392],
	[0.2325291,0.14128101,0.047506111704165244],
	[0.2312893,0.07477626,0.055758167863764516],
	[0.22751166,0.098526575,0.10752841324599291],
	[0.22615628,0.18045746,0.16055837945058732],
	[0.23543021,0.24858856,0.22770210288662088],
	[0.22448786,0.21403486,0.2081381205627295],
	[0.21832664,0.20797352,0.19835390662948735],
	[0.20587884,0.20894553,0.18017627537075417],
	[0.19256005,0.22474267,0.17313456628306254],
	[0.17262988,0.26370603,0.18310929551373684],
	[0.17065287,0.27603704,0.1890902591516427],
	[0.16776314,0.286195,0.16711278010853278],
	[0.16776314,0.286195,0.16711278010853278],
	[0.16634843,0.22458224,0.12256846611171265],
	[0.16913813,0.24329439,0.15102634119939817],
	[0.1720732,0.23068713,0.14322031017622877],
	[0.18122327,0.20919533,0.12912883888524854],
	[0.18122327,0.20919533,0.12912883888524854],
	[0.18122327,0.20919533,0.12912883888524854],
	[0.1958371,0.114393935,0.07912049608242941],
	[0.19394998,0.1919716,0.1292968884964791],
	[0.19394998,0.1919716,0.1292968884964791],
	[0.19394998,0.1919716,0.1292968884964791],
	[0.18122327,0.20919533,0.12912883888524854],
	[0.1720732,0.23068713,0.14322031017622877],
	[0.1720732,0.23068713,0.14322031017622877],
	[0.1720732,0.23068713,0.14322031017622877],
	[0.1770246,0.26939642,0.2157679070877814],
	[0.2092587,0.26758668,0.1612646485744482],
	[0.23403767,0.1255887,0.1075642130283432],
	[0.22926106,0.3084171,0.36181764194306576],
	[0.20749572,0.3531313,0.4486519595914641],
	[0.1456736,0.41368622,0.552998170773888],
	[0.10460751,0.387622,0.5749354449981375],
	[0.073378965,0.20639738,0.6240341135846112],
	[0.07097964,0.20841646,0.6032153236996223],
	[0.07529202,0.18649895,0.5805180439693488],
	[0.08252928,0.2405681,0.5718652358157968],
	[0.08252928,0.2405681,0.5718652358157968],
	[0.08073819,0.235622,0.6129123887956092],
	[0.07657708,0.2042624,0.6464695214892491],
	[0.068917364,0.41838616,0.72573448768821],
	[0.06557594,0.49422058,0.753324618684377],
	[0.06846899,0.5159404,0.7683798790525773],
	[0.06846899,0.5159404,0.7683798790525773],
	[0.06846899,0.5159404,0.7683798790525773],
	[0.070435196,0.5392858,0.7757823859467489],
	[0.070435196,0.5392858,0.7757823859467489],
	[0.068917364,0.41838616,0.72573448768821],
	[0.068917364,0.41838616,0.72573448768821],
	[0.06961268,0.45472962,0.7383286187153885],
	[0.06961268,0.45472962,0.7383286187153885],
	[0.06961268,0.45472962,0.7383286187153885],
	[0.06846899,0.5159404,0.7683798790525773],
	[0.068104625,0.6261708,0.8512230209658794],
	[0.053764362,0.71150005,0.8884100353886206],
	[0.059789266,0.61112446,0.7932324874513844],
	[0.054802362,0.56712645,0.6808641579519287],
	[0.051411103,0.5682251,0.6272425497746831],
	[0.063966796,0.5127094,0.5031700180148192],
	[0.07624698,0.37759665,0.3013591690023694],
	[0.07794705,0.12248223,0.11471320753972766],
	[0.08041997,0.10379384,0.09997342373455667],
	[0.09119491,0.12370708,0.10367395819867184],
	[0.09119491,0.12370708,0.10367395819867184],
	[0.07794705,0.12248223,0.11471320753972766],
	[0.06992833,0.11539672,0.10247849204895182],
	[0.06369382,0.09590633,0.0806614664607227],
	[0.06369382,0.09590633,0.0806614664607227],
	[0.06046052,0.1640697,0.09888714932605368],
	[0.06046052,0.1640697,0.09888714932605368],
	[0.08094026,0.28212368,0.18579316780435967],
	[0.05672029,0.4458254,0.32873616205383155],
	[0.053624976,0.39667973,0.23217873896143307],
	[0.05037461,0.43628597,0.26532322606254294],
	[0.05037461,0.43628597,0.26532322606254294],
	[0.05037461,0.43628597,0.26532322606254294],
	[0.05672029,0.4458254,0.32873616205383155],
	[0.05672029,0.4458254,0.32873616205383155],
	[0.05672029,0.4458254,0.32873616205383155],
	[0.06234395,0.4487122,0.37994428617000947]
	];

let tempDataset = [
	{"Color" : 0.16052112,"Intensity" : 0.043206494,"Orientation" : 0.132315235},
	{"Color" : 0.16052112,"Intensity" : 0.043206494,"Orientation" : 0.132315235},
	{"Color" : 0.16282108,"Intensity" : 0.075270265,"Orientation" : 0.129309805},
	{"Color" : 0.16282108,"Intensity" : 0.075270265,"Orientation" : 0.129309805},
	{"Color" : 0.16282108,"Intensity" : 0.075270265,"Orientation" : 0.129309805},
	{"Color" : 0.16290806,"Intensity" : 0.07658491,"Orientation" : 0.126773377},
	{"Color" : 0.16180123,"Intensity" : 0.051869046,"Orientation" : 0.124476494},
	{"Color" : 0.16290806,"Intensity" : 0.07658491,"Orientation" : 0.126773377},
	{"Color" : 0.16290806,"Intensity" : 0.07658491,"Orientation" : 0.126773377},
	{"Color" : 0.16290806,"Intensity" : 0.07658491,"Orientation" : 0.126773377},
	{"Color" : 0.15678275,"Intensity" : 0.06431136,"Orientation" : 0.128535787},
	{"Color" : 0.15746929,"Intensity" : 0.05784232,"Orientation" : 0.129066753},
	{"Color" : 0.15746929,"Intensity" : 0.05784232,"Orientation" : 0.129066753},
	{"Color" : 0.15746929,"Intensity" : 0.05784232,"Orientation" : 0.129066753},
	{"Color" : 0.15129656,"Intensity" : 0.055002317,"Orientation" : 0.130483858},
	{"Color" : 0.15129656,"Intensity" : 0.055002317,"Orientation" : 0.130483858},
	{"Color" : 0.1522229,"Intensity" : 0.04385593,"Orientation" : 0.128830063},
	{"Color" : 0.1522229,"Intensity" : 0.04385593,"Orientation" : 0.128830063},
	{"Color" : 0.1522229,"Intensity" : 0.04385593,"Orientation" : 0.128830063},
	{"Color" : 0.15129656,"Intensity" : 0.055002317,"Orientation" : 0.130483858},
	{"Color" : 0.15129656,"Intensity" : 0.055002317,"Orientation" : 0.130483858},
	{"Color" : 0.15678275,"Intensity" : 0.06431136,"Orientation" : 0.128535787},
	{"Color" : 0.15678275,"Intensity" : 0.06431136,"Orientation" : 0.128535787},
	{"Color" : 0.15678275,"Intensity" : 0.06431136,"Orientation" : 0.128535787},
	{"Color" : 0.16180123,"Intensity" : 0.051869046,"Orientation" : 0.124476494},
	{"Color" : 0.16180123,"Intensity" : 0.051869046,"Orientation" : 0.124476494},
	{"Color" : 0.15571661,"Intensity" : 0.040180184,"Orientation" : 0.128067995},
	{"Color" : 0.16180123,"Intensity" : 0.051869046,"Orientation" : 0.124476494},
	{"Color" : 0.16290806,"Intensity" : 0.07658491,"Orientation" : 0.126773377},
	{"Color" : 0.15678275,"Intensity" : 0.06431136,"Orientation" : 0.128535787},
	{"Color" : 0.15746929,"Intensity" : 0.05784232,"Orientation" : 0.129066753},
	{"Color" : 0.15746929,"Intensity" : 0.05784232,"Orientation" : 0.129066753},
	{"Color" : 0.16052112,"Intensity" : 0.043206494,"Orientation" : 0.132315235},
	{"Color" : 0.16052112,"Intensity" : 0.043206494,"Orientation" : 0.132315235},
	{"Color" : 0.16553298,"Intensity" : 0.066077046,"Orientation" : 0.134898389},
	{"Color" : 0.16613753,"Intensity" : 0.039621525,"Orientation" : 0.135329796},
	{"Color" : 0.16613753,"Intensity" : 0.039621525,"Orientation" : 0.135329796},
	{"Color" : 0.16613753,"Intensity" : 0.039621525,"Orientation" : 0.135329796},
	{"Color" : 0.16553298,"Intensity" : 0.066077046,"Orientation" : 0.134898389},
	{"Color" : 0.16553298,"Intensity" : 0.066077046,"Orientation" : 0.134898389},
	{"Color" : 0.16282108,"Intensity" : 0.075270265,"Orientation" : 0.129309805},
	{"Color" : 0.15746929,"Intensity" : 0.05784232,"Orientation" : 0.129066753},
	{"Color" : 0.15746929,"Intensity" : 0.05784232,"Orientation" : 0.129066753},
	{"Color" : 0.15678275,"Intensity" : 0.06431136,"Orientation" : 0.128535787},
	{"Color" : 0.15678275,"Intensity" : 0.06431136,"Orientation" : 0.128535787},
	{"Color" : 0.15571661,"Intensity" : 0.040180184,"Orientation" : 0.128067995},
	{"Color" : 0.15571661,"Intensity" : 0.040180184,"Orientation" : 0.128067995},
	{"Color" : 0.15571661,"Intensity" : 0.040180184,"Orientation" : 0.128067995},
	{"Color" : 0.15571661,"Intensity" : 0.040180184,"Orientation" : 0.128067995},
	{"Color" : 0.15571661,"Intensity" : 0.040180184,"Orientation" : 0.128067995},
	{"Color" : 0.15025252,"Intensity" : 0.03500809,"Orientation" : 0.131473541},
	{"Color" : 0.16065125,"Intensity" : 0.038762543,"Orientation" : 0.122675567},
	{"Color" : 0.1639236,"Intensity" : 0.08823463,"Orientation" : 0.122989109},
	{"Color" : 0.14563946,"Intensity" : 0.08131657,"Orientation" : 0.097026506},
	{"Color" : 0.13930583,"Intensity" : 0.027225655,"Orientation" : 0.10219948},
	{"Color" : 0.16939095,"Intensity" : 0.051507078,"Orientation" : 0.144967664},
	{"Color" : 0.15462878,"Intensity" : 0.18776779,"Orientation" : 0.267500681},
	{"Color" : 0.16546786,"Intensity" : 0.3357673,"Orientation" : 0.336437442},
	{"Color" : 0.2077326,"Intensity" : 0.51447546,"Orientation" : 0.528138737},
	{"Color" : 0.22464232,"Intensity" : 0.19187912,"Orientation" : 0.373964667},
	{"Color" : 0.24406457,"Intensity" : 0.07603589,"Orientation" : 0.174658213},
	{"Color" : 0.1909773,"Intensity" : 0.30653575,"Orientation" : 0.23763982},
	{"Color" : 0.22845909,"Intensity" : 0.17031114,"Orientation" : 0.139181993},
	{"Color" : 0.2325291,"Intensity" : 0.14128101,"Orientation" : 0.047506112},
	{"Color" : 0.2312893,"Intensity" : 0.07477626,"Orientation" : 0.055758168},
	{"Color" : 0.22751166,"Intensity" : 0.098526575,"Orientation" : 0.107528413},
	{"Color" : 0.22615628,"Intensity" : 0.18045746,"Orientation" : 0.160558379},
	{"Color" : 0.23543021,"Intensity" : 0.24858856,"Orientation" : 0.227702103},
	{"Color" : 0.22448786,"Intensity" : 0.21403486,"Orientation" : 0.208138121},
	{"Color" : 0.21832664,"Intensity" : 0.20797352,"Orientation" : 0.198353907},
	{"Color" : 0.20587884,"Intensity" : 0.20894553,"Orientation" : 0.180176275},
	{"Color" : 0.19256005,"Intensity" : 0.22474267,"Orientation" : 0.173134566},
	{"Color" : 0.17262988,"Intensity" : 0.26370603,"Orientation" : 0.183109296},
	{"Color" : 0.17065287,"Intensity" : 0.27603704,"Orientation" : 0.189090259},
	{"Color" : 0.16776314,"Intensity" : 0.286195,"Orientation" : 0.16711278},
	{"Color" : 0.16776314,"Intensity" : 0.286195,"Orientation" : 0.16711278},
	{"Color" : 0.16634843,"Intensity" : 0.22458224,"Orientation" : 0.122568466},
	{"Color" : 0.16913813,"Intensity" : 0.24329439,"Orientation" : 0.151026341},
	{"Color" : 0.1720732,"Intensity" : 0.23068713,"Orientation" : 0.14322031},
	{"Color" : 0.18122327,"Intensity" : 0.20919533,"Orientation" : 0.129128839},
	{"Color" : 0.18122327,"Intensity" : 0.20919533,"Orientation" : 0.129128839},
	{"Color" : 0.18122327,"Intensity" : 0.20919533,"Orientation" : 0.129128839},
	{"Color" : 0.1958371,"Intensity" : 0.114393935,"Orientation" : 0.079120496},
	{"Color" : 0.19394998,"Intensity" : 0.1919716,"Orientation" : 0.129296888},
	{"Color" : 0.19394998,"Intensity" : 0.1919716,"Orientation" : 0.129296888},
	{"Color" : 0.19394998,"Intensity" : 0.1919716,"Orientation" : 0.129296888},
	{"Color" : 0.18122327,"Intensity" : 0.20919533,"Orientation" : 0.129128839},
	{"Color" : 0.1720732,"Intensity" : 0.23068713,"Orientation" : 0.14322031},
	{"Color" : 0.1720732,"Intensity" : 0.23068713,"Orientation" : 0.14322031},
	{"Color" : 0.1720732,"Intensity" : 0.23068713,"Orientation" : 0.14322031},
	{"Color" : 0.1770246,"Intensity" : 0.26939642,"Orientation" : 0.215767907},
	{"Color" : 0.2092587,"Intensity" : 0.26758668,"Orientation" : 0.161264649},
	{"Color" : 0.23403767,"Intensity" : 0.1255887,"Orientation" : 0.107564213},
	{"Color" : 0.22926106,"Intensity" : 0.3084171,"Orientation" : 0.361817642},
	{"Color" : 0.20749572,"Intensity" : 0.3531313,"Orientation" : 0.44865196},
	{"Color" : 0.1456736,"Intensity" : 0.41368622,"Orientation" : 0.552998171},
	{"Color" : 0.10460751,"Intensity" : 0.387622,"Orientation" : 0.574935445},
	{"Color" : 0.073378965,"Intensity" : 0.20639738,"Orientation" : 0.624034114},
	{"Color" : 0.07097964,"Intensity" : 0.20841646,"Orientation" : 0.603215324},
	{"Color" : 0.07529202,"Intensity" : 0.18649895,"Orientation" : 0.580518044},
	{"Color" : 0.08252928,"Intensity" : 0.2405681,"Orientation" : 0.571865236},
	{"Color" : 0.08252928,"Intensity" : 0.2405681,"Orientation" : 0.571865236},
	{"Color" : 0.08073819,"Intensity" : 0.235622,"Orientation" : 0.612912389},
	{"Color" : 0.07657708,"Intensity" : 0.2042624,"Orientation" : 0.646469521},
	{"Color" : 0.068917364,"Intensity" : 0.41838616,"Orientation" : 0.725734488},
	{"Color" : 0.06557594,"Intensity" : 0.49422058,"Orientation" : 0.753324619},
	{"Color" : 0.06846899,"Intensity" : 0.5159404,"Orientation" : 0.768379879},
	{"Color" : 0.06846899,"Intensity" : 0.5159404,"Orientation" : 0.768379879},
	{"Color" : 0.06846899,"Intensity" : 0.5159404,"Orientation" : 0.768379879},
	{"Color" : 0.070435196,"Intensity" : 0.5392858,"Orientation" : 0.775782386},
	{"Color" : 0.070435196,"Intensity" : 0.5392858,"Orientation" : 0.775782386},
	{"Color" : 0.068917364,"Intensity" : 0.41838616,"Orientation" : 0.725734488},
	{"Color" : 0.068917364,"Intensity" : 0.41838616,"Orientation" : 0.725734488},
	{"Color" : 0.06961268,"Intensity" : 0.45472962,"Orientation" : 0.738328619},
	{"Color" : 0.06961268,"Intensity" : 0.45472962,"Orientation" : 0.738328619},
	{"Color" : 0.06961268,"Intensity" : 0.45472962,"Orientation" : 0.738328619},
	{"Color" : 0.06846899,"Intensity" : 0.5159404,"Orientation" : 0.768379879},
	{"Color" : 0.068104625,"Intensity" : 0.6261708,"Orientation" : 0.851223021},
	{"Color" : 0.053764362,"Intensity" : 0.71150005,"Orientation" : 0.888410035},
	{"Color" : 0.059789266,"Intensity" : 0.61112446,"Orientation" : 0.793232487},
	{"Color" : 0.054802362,"Intensity" : 0.56712645,"Orientation" : 0.680864158},
	{"Color" : 0.051411103,"Intensity" : 0.5682251,"Orientation" : 0.62724255},
	{"Color" : 0.063966796,"Intensity" : 0.5127094,"Orientation" : 0.503170018},
	{"Color" : 0.07624698,"Intensity" : 0.37759665,"Orientation" : 0.301359169},
	{"Color" : 0.07794705,"Intensity" : 0.12248223,"Orientation" : 0.114713208},
	{"Color" : 0.08041997,"Intensity" : 0.10379384,"Orientation" : 0.099973424},
	{"Color" : 0.09119491,"Intensity" : 0.12370708,"Orientation" : 0.103673958},
	{"Color" : 0.09119491,"Intensity" : 0.12370708,"Orientation" : 0.103673958},
	{"Color" : 0.07794705,"Intensity" : 0.12248223,"Orientation" : 0.114713208},
	{"Color" : 0.06992833,"Intensity" : 0.11539672,"Orientation" : 0.102478492},
	{"Color" : 0.06369382,"Intensity" : 0.09590633,"Orientation" : 0.080661466},
	{"Color" : 0.06369382,"Intensity" : 0.09590633,"Orientation" : 0.080661466},
	{"Color" : 0.06046052,"Intensity" : 0.1640697,"Orientation" : 0.098887149},
	{"Color" : 0.06046052,"Intensity" : 0.1640697,"Orientation" : 0.098887149},
	{"Color" : 0.08094026,"Intensity" : 0.28212368,"Orientation" : 0.185793168},
	{"Color" : 0.05672029,"Intensity" : 0.4458254,"Orientation" : 0.328736162},
	{"Color" : 0.053624976,"Intensity" : 0.39667973,"Orientation" : 0.232178739},
	{"Color" : 0.05037461,"Intensity" : 0.43628597,"Orientation" : 0.265323226},
	{"Color" : 0.05037461,"Intensity" : 0.43628597,"Orientation" : 0.265323226},
	{"Color" : 0.05037461,"Intensity" : 0.43628597,"Orientation" : 0.265323226},
	{"Color" : 0.05672029,"Intensity" : 0.4458254,"Orientation" : 0.328736162},
	{"Color" : 0.05672029,"Intensity" : 0.4458254,"Orientation" : 0.328736162},
	{"Color" : 0.05672029,"Intensity" : 0.4458254,"Orientation" : 0.328736162},
	{"Color" : 0.06234395,"Intensity" : 0.4487122,"Orientation" : 0.379944286}
	];


let feature_color = [
	{"level": 0, "value": 37693},
	{"level": 1, "value": 89070},
	{"level": 2, "value": 157429},
	{"level": 3, "value": 77248},
	{"level": 4, "value": 39504},
	{"level": 5, "value": 27391},
	{"level": 6, "value": 19401},
	{"level": 7, "value": 18662},
	{"level": 8, "value": 10880},
	{"level": 9, "value": 2722}
	];

let feature_intensity = [
	{"level": 0, "value": 31044},
	{"level": 1, "value": 185173},
	{"level": 2, "value": 135627},
	{"level": 3, "value": 66438},
	{"level": 4, "value": 35454},
	{"level": 5, "value": 17375},
	{"level": 6, "value": 6284},
	{"level": 7, "value": 2051},
	{"level": 8, "value": 389},
	{"level": 9, "value": 165}
	];

let feature_orientation = [
	{"level": 0, "value": 23310}, 
	{"level": 1, "value": 109347}, 
	{"level": 2, "value": 107571}, 
	{"level": 3, "value": 82478}, 
	{"level": 4, "value": 68123}, 
	{"level": 5, "value": 49416}, 
	{"level": 6, "value": 21345}, 
	{"level": 7, "value": 9778}, 
	{"level": 8, "value": 6458}, 
	{"level": 9, "value": 2174}
	];

let feats_JSON = [];
feats_JSON.push(feature_color);
feats_JSON.push(feature_intensity);
feats_JSON.push(feature_orientation);

let recds_feat_JSON = [];
for(let i=0; i<NAME_FEATURES.length; i++){

	let feat_JSON = [];
	for(let j=0; j<10; j++){
		let _min = j*0.1;
		let _max = (j+1)*0.1;
		let _recs = countingRecords_idx(i, [_min, _max]);

		//console.log("f"+i+"_"+j+": "+_recs);
		let feat_obj = {
			"level": j,
			"value": _recs
		};

		feat_JSON.push(feat_obj);
	}
	recds_feat_JSON.push(feat_JSON);
}

console.log(recds_feat_JSON);





// update data panel
let iDataColumns = $('#data_columns');
//iDataColumns.append("<br>");
//let dlColumns = $(`
//	<selectgroup>
//       <option value="d_x">x</option>
//        <option value="d_y">y</option>
//    </selectgroup>
//`);

//iDataColumns.append(dlColumns);

let iMeasurement = $('#data_measurement');
iMeasurement.append("<br>");
let dlMeasurement = $(`
	<input type='radio' value='m_duration' checked/>
	<label>duration</label>
	<input type='radio' value='m_pupil_diameter' />
	<label>pupil diameter</label>
`);
iMeasurement.append(dlMeasurement);

let iEvent = $('#data_event');
iEvent.append("<br>");
let dlEvent = $(`
	<input type='checkbox' id="e_fix" name='e_fix' checked/>
	<label for="e_fix">fixation</label>
	<input type='checkbox' id="e_sac" name='e_sac' checked/>
	<label for="e_sac">saccade</label>
	<input type='checkbox' id="e_pso" name='e_pso' />
	<label for="e_pso">PSO</label>
	<br>
	<input type='checkbox' id="e_bli" name='e_bli' />
	<label for="e_bli">blink</label>
	<input type='checkbox' id="e_unk" name='e_unk' />
	<label for="e_unk">unknown</label>
`);
iEvent.append(dlEvent);

selectAOIgrid(10, 10);
for(let p=0; p<feats_JSON.length; p++){
	drawStimulusFeature(feats_JSON[p], p);
}


//drawStimulusFeature("http://127.0.0.1:8000/data_processing/U0121_1RTE_saliency_color.csv");
//drawStimulusFeature("http://127.0.0.1:8000/data_processing/U0121_1RTE_saliency_intensity.csv");
//drawStimulusFeature("http://127.0.0.1:8000/data_processing/U0121_1RTE_saliency_orientation.csv");





function drawGridHeat_fixDur(dataset){
	console.log("drawGridHeat_fixDur");

	let dim_1 = [];
	let dim_2 = [];
	let _value = [];
	for(let i=0; i<dataset.length; i++){
		let _v = [];

		for(let j=0; j<dataset[i].length; j++){
			dim_2.push("t"+(j+1).toString());
			_v.push(dataset[i][j].duration);
		}
		dim_1.push("o"+(i+1).toString());
		_value.push(_v);
	}

	// make index data (instead *.tsv file)
	let indexData = [];
	for(let i=0; i<dim_1.length; i++){
		if (i==0){
			indexData.push(["dim1", "dim2"]);
		}
		for(let j=0; j<dim_2.length; j++){
			indexData.push([i+1, j+1]);
		}
	}

	// make input data: index & value
	let inputdata = [];
	for(let i=1; i<indexData.length; i++){
		let _row = [indexData[i][0], indexData[i][1], _value[indexData[i][0]-1][i-1]];
		inputdata.push(_row);
	}
	//console.log(inputdata);

	//UI configuration
	const margin = { top: 50, right: 0, bottom: 100, left: 30 },
		width = 400 - margin.left - margin.right,
		height = 430 - margin.top - margin.bottom,
		gridSize = Math.floor(width / 10),
		legendElementWidth = gridSize*1,
		buckets = 9,
		colors = ["#ffffd9","#edf8b1","#c7e9b4","#7fcdbb","#41b6c4","#1d91c0","#225ea8","#253494","#081d58"]; // alternatively colorbrewer.YlGnBu[9]

	var svg = d3.select("body").append("svg")
		.attr("width", width + margin.left + margin.right)
		.attr("height", height + margin.top + margin.bottom)
		.append("g")
		.attr("transform", "translate(" + margin.left + "," + margin.top + ")");

	const yLabel = svg.selectAll(".yLabel")
		.data(dim_1)
		.enter().append("text")
			.text(function(d){return d;})
			.attr("x",0)
			.attr("y", function(d, i){return i*gridSize;})
		.style("text-anchor", "end")
		.attr("transform", "translate(-6,"+gridSize/1.5+")");

	const xLabel = svg.selectAll(".xLabel")
		.data(dim_2)
		.enter().append("text")
			.text(function(d){return d;})
			.attr("x", function(d, i){return i*gridSize;})
			.attr("y",0)
		.style("text-anchor", "middle")
		.attr("transform", "translate(" + gridSize / 2 + ", -6)");

	const heatmapChart = function(_data){
		//console.log(_data);
		const colorScale = d3.scaleQuantile()
            .domain([0, buckets - 1, d3.max(_data, function(d){return d[2];})])
            .range(colors);

        const cards = svg.selectAll(".card")
        	.data(_data, function(d){return d[0]+':'+d[1];});
    	cards.append("title");

    	cards.enter().append("rect")
    		.attr("x", function(d){return (d[1]-1)*gridSize;})
    		.attr("y", function(d){return (d[0]-1)*gridSize;})
    		.attr("rx", 4)
    		.attr("ry", 4)
    		.attr("width", gridSize)
    		.attr("height", gridSize)
    		.style("fill", colors[0])
		.merge(cards)
			.transition()
			.duration(1000)
			.style("fill", function(d){return colorScale(d[2]);});

		cards.select("title").text(function(d){return d[2];});
		cards.exit().remove();

		const legend = svg.selectAll(".legend")
			.data([0].concat(colorScale.quantiles()), function(d){return d;});

		const legend_g = legend.enter().append("g")
			.attr("class", "legend");

		legend_g.append("rect")
			.attr("x", function(d, i){return legendElementWidth*i;})
			.attr("y", height)
			.attr("width", legendElementWidth)
			.attr("height", gridSize/2)
			.style("fill", function(d, i){return colors[i];});

		legend_g.append("text")
			.attr("class", "mono")
			.text(function(d){return "≥ "+Math.round(d);})
			.attr("x", function(d, i){return legendElementWidth*i;})
			.attr("y", height + gridSize);

		legend.exit().remove();
	}
	heatmapChart(inputdata);
}


function selectAOIgrid(setRow, setCol){
	// UI setting: "ADD" & "CLEAR" button
	let AOIselector = $('#aoi_selector');
	AOIselector.append("<br>");

	let xpos = 0;
	let ypos = 0;
	let cellWidth = 30;
	let cellHeight = 30;
	let click = 0;
	let rowNum = 10;
	let colNum = 10;

  	let margin = {top: 10, right: 10, bottom: 10, left: 10};
	let SVG_width = DIV_WIDTH;
	let SVG_height = DIV_HEIGHT;

  	if(STIMULUS_WIDTH>STIMULUS_HEIGHT){
  		let _r = SVG_width/STIMULUS_WIDTH;
  		SVG_width = STIMULUS_WIDTH*_r;
  		SVG_height = STIMULUS_HEIGHT*_r;
  	}else{
  		let _r = SVG_height/STIMULUS_HEIGHT;
  		SVG_width = STIMULUS_WIDTH*_r;
  		SVG_height = STIMULUS_HEIGHT*_r;
  	}

	// set cell width & height 
	cellWidth = SVG_width/rowNum;
	cellHeight = SVG_height/colNum;

	for(let _r=0; _r<setRow; _r++){
		let _row = [];

		for(let _c=0; _c<setCol; _c++){
			let _col = {
				x: xpos,
				y: ypos,
				width: cellWidth,
				height: cellHeight,
				click: click,
				group: 0
			};

			_row.push(_col);
			xpos += cellWidth;
		}
		stiGrid.push(_row);

		xpos = 0;
		ypos += cellHeight;
	}

	d3.select("#aoi_selector").append("button")
		.text("ADD")
		.on("click", function(){
			SELECTED_AOI = -999;
			if(selectedAppendCell.length != 0){
				// push grid cell array in AOIarray
				let _add = [];
				for(let i=0; i<selectedAppendCell.length; i++){
					_add.push(selectedAppendCell[i])
					let _r = selectedAppendCell[i][0];
					let _c = selectedAppendCell[i][1];

					// set AOI group
					stiGrid[_r][_c].group = AOIarray.length+1;
					//stiGrid[_r][_c].click = -999;
					//console.log(stiGrid[_r][_c]);
				}
				AOIarray.push(_add);

				// clear selectedAppendCell
				selectedAppendCell = [];
			}

			let grid = d3.select("#aoi_selector").selectAll("svg");
			let row = grid.selectAll(".row")
				.data(stiGrid);

			let column = row.selectAll(".square")
				.data(function(d){return d;});

			column.exit().remove();
			column.enter().append("rect")
				.attr("x", 0)
				.attr("y", 0)
				.attr("opacity", STIMULUS_OPACITY)
				.attr("width", function(d){return d.width;})
				.attr("height", function(d){return d.height;})
				.style("fill", "red")
				.style("stroke", "#222");

			column.transition()
				.duration(500)
				.attr("opacity", STIMULUS_OPACITY)
				.attr("x", function(d){return d.x;})
				.attr("y", function(d){return d.y;})
				.style("fill", function(d, i){
					if(d.group == 0){
						return "#fff";
					}else{
						return AOIcolorBrewer_12class_set3[d.group];
					}
				});

			// counting fixation duration on each AOIs
			AOIduration = [];
			AOIduration = countingDurationInAOI(rawGazeData);

			// AOI list view
		    drawAOIlist(SVG_width, SVG_height);
		    
		    // make JSON array
		    let JSON_feats = getFeaturesInAOI_grouop(AOIarray, rawGazeData, saliencyFeatures, NAME_FEATURES);
		    //for(let i=0; i<JSON_feats.length; i++){
		    //	drawParallelCoordinatesChart(JSON_feats[i]);
		    //}
		    drawParallelCoordinatesChart_group(JSON_feats);

		});

	d3.select("#aoi_selector").append("button")
		.text("DELETE")
		.on("click", function(){
			if(selectedDeleteCell.length != 0){
				// push grid cell array in AOIarray
				let _dell = [];
				for(let i=0; i<selectedDeleteCell.length; i++){
					_dell.push(selectedDeleteCell[i])
					let _r = selectedDeleteCell[i][0];
					let _c = selectedDeleteCell[i][1];

					// set AOI group
					stiGrid[_r][_c].group = 0;
					stiGrid[_r][_c].click = 0;
					//console.log(stiGrid[_r][_c]);
				}
				for(let i=0; i<_dell.length; i++){
					let dellIdx = 0;
					for(let j=0; j<AOIarray[SELECTED_AOI-1].length; j++){
						if((AOIarray[SELECTED_AOI-1][j][0] == _dell[i][0]) && (AOIarray[SELECTED_AOI-1][j][1] == _dell[i][1])){
							dellIdx = j;
							break;
						}

					}
					AOIarray[SELECTED_AOI-1].splice(dellIdx,1);
				}

				// clear selectedDeleteCell
				selectedDeleteCell = [];
			}
			SELECTED_AOI = -999;

			let grid = d3.select("#aoi_selector").selectAll("svg");
			let row = grid.selectAll(".row")
				.data(stiGrid);

			let column = row.selectAll(".square")
				.data(function(d){return d;});

			column.exit().remove();
			column.enter().append("rect")
				.attr("x", 0)
				.attr("y", 0)
				.attr("opacity", STIMULUS_OPACITY)
				.attr("width", function(d){return d.width;})
				.attr("height", function(d){return d.height;})
				.style("fill", "red")
				.style("stroke", "#222");

			column.transition()
				.duration(500)
				.attr("opacity", STIMULUS_OPACITY)
				.attr("x", function(d){return d.x;})
				.attr("y", function(d){return d.y;})
				.style("fill", function(d, i){
					if(d.group == 0){
						return "#fff";
					}else{
						return AOIcolorBrewer_12class_set3[d.group];
					}
				});
		});

	d3.select("#aoi_selector").append("button")
		.text("GAZE DATA")
		.on("click", function(){
			FLAG_DRAW_GAZE_ON_STIMULUS++;
			
			let svg = d3.select("#aoi_selector").selectAll("svg");

			let circles = svg.selectAll("circle")
				.data(rawGazeData)
				.transition()
				.duration(500)
				.attr("r", function(d){
					if(FLAG_DRAW_GAZE_ON_STIMULUS%2 == 1){
						return 0;
					}else{
						return 2;
					}
				});			
		});

	d3.select("#aoi_selector").append("button")
		.text("CELL-OPACITY")
		.on("click", function(){
			STIMULUS_OPACITY += 0.25;

			let grid = d3.select("#aoi_selector").selectAll("svg");
			let row = grid.selectAll(".row")
				.data(stiGrid);

			let column = row.selectAll(".square")
				.data(function(d){return d;});

			column.transition()
				.duration(500)
				.attr("opacity", function(d){
					if(STIMULUS_OPACITY > 1){
						STIMULUS_OPACITY = 0;
					}
					return STIMULUS_OPACITY;
				});			
		});

	let grid = d3.select("#aoi_selector").append("svg")
		.attr("width", SVG_width)
		.attr("height", SVG_height);

	let stimulus = grid.append("image")
		.attr("xlink:href", "http://127.0.0.1:8000/static/stimulus/U0121_1RTE.jpg")
		.attr("width", SVG_width)
		.attr("height", SVG_height);

	let row = grid.selectAll(".row")
		.data(stiGrid)
		.enter().append("g")
		.attr("class", "row");

	let column = row.selectAll(".square")
		.data(function(d){return d;})
		.enter().append("rect")
		.attr("class", "square")
		.attr("opacity", STIMULUS_OPACITY)
		.attr("x", function(d){return d.x;})
		.attr("y", function(d){return d.y;})
		.attr("width", function(d){return d.width;})
		.attr("height", function(d){return d.height;})
		.style("fill", "#fff")
		.style("stroke", "#222")
		.on("click", function(d, i){
			d.click++;
			let gridArray = [Math.floor(d.y/d.height), i];
			//console.log(gridArray);
			if(d.group==0){
				if((d.click)%2==0){
					d3.select(this).style("fill", "#fff");
					selectedAppendCell.splice(selectedAppendCell.indexOf(gridArray),1);
				}
				if((d.click)%2==1){
					d3.select(this).style("fill", AOIcolorBrewer_12class_set3[0]);
					selectedAppendCell.push(gridArray);
				}
			}else{
				// d.group!=0
				if(SELECTED_AOI < 0){
					SELECTED_AOI = d.group;
				}

				if(SELECTED_AOI == d.group){
					if((d.click)%2==0){
						d3.select(this).style("fill", "black");
						selectedDeleteCell.push(gridArray);
					}
					if((d.click)%2==1){
						d3.select(this).style("fill", AOIcolorBrewer_12class_set3[d.group+1]);
						selectedDeleteCell.splice(selectedDeleteCell.indexOf(gridArray),1);
					}
				}
			}
			//console.log(selectedAppendCell);
		});

	let gazePoint = grid.selectAll(".gp")
		.data(rawGazeData)
		.enter().append("circle")
		.attr("cx", function(d){return convertGazeToSVG(d)[0];})
		.attr("cy", function(d){return convertGazeToSVG(d)[1];})
		.attr("r", 2)
		//.attr("opacity", function(d, i){return (1/rawGazeData.length)*i;})
		.style("fill","red")
		.style("stroke", "black");

	function drawAOIlist(SVG_w, SVG_h){
		// AOI list view
	    let AoIboxWidth = 40;
	    let AoIboxHeight = 40;
	    let gap = 10;
	    let textHeight = 20;

	    if(AOIarray.length > 7){
	    	AoIboxWidth = (SVG_w/AOIarray.length)-gap;
	    }

	    d3.select("#aoi_list").selectAll("*").remove();

	    let svg = d3.select("#aoi_list").append("svg")
	    	.attr("width", SVG_w)
	    	.attr("height", SVG_h);

    	let aois = svg.selectAll("rect")
    		.data(AOIarray);
		aois.exit().remove();

		aois.enter().append("rect")
			.attr("x", function(d, i){return gap/2+(AoIboxWidth+gap)*i})
			.attr("y", gap+textHeight)
			.attr("width", AoIboxWidth)
			.attr("height", AoIboxHeight)
			//.attr("opacity", 0.5)
			.attr("fill", function(d, i){return AOIcolorBrewer_12class_set3[i+1]})
			.attr("stroke", "black");

		aois.enter().append("text")
			.text(function(d, i){
				return "AOI_"+i.toString();
			})
			.attr("x", function(d, i){return gap+(AoIboxWidth+gap)*i})
			.attr("y", (gap+textHeight)/2)
			.attr("font-size", "10px")
			.attr("fill", "black");

		// set max duration
		let maxDuration = 0;
		for(let i=0; i<AOIduration.length; i++){
			if(AOIduration[i] > maxDuration){
				maxDuration = AOIduration[i];
			}
		}

		let setMaxHeight = 100;
		let heightRatio = 1;
		if(maxDuration != 0 ){
			if(maxDuration > setMaxHeight){
				heightRatio = setMaxHeight/maxDuration;
			}else{
				heightRatio = 1
			}
		}else{
			heightRatio = 0;
		}

		let durationBar = svg.selectAll("bar")
			.data(AOIduration);
		durationBar.exit().remove();
		durationBar.enter().append("rect")
			.attr("x", function(d, i){return AoIboxWidth/3+gap/2+(AoIboxWidth+gap)*i})
			.attr("y", gap+textHeight+AoIboxHeight)
			.attr("width", AoIboxWidth/3)
			.attr("height", function(d){return d*heightRatio})
			.attr("fill", "black");

		durationBar.enter().append("text")
			.text(function(d){return d;})
			.attr("x", function(d, i){return AoIboxWidth/3+gap/2+(AoIboxWidth+gap)*i})
			.attr("y", function(d){return d*heightRatio+gap+textHeight+AoIboxHeight+10})
			.attr("font-size", "10px")
			.attr("fill", "black");
	}

}

function drawStimulusFeature(_featsJSON, _idx){
	// set the dimensions and margins of the graph
	let margin = {top: 20, right: 20, bottom: 30, left: 50},
	    width = 350 - margin.left - margin.right,
	    height = 150 - margin.top - margin.bottom;

	// set the ranges
	let x = d3.scaleBand()
	          .range([0, width])
	          .padding(0.1);
	let y = d3.scaleLinear()
	          .range([height, 0]);

  	let brushedArea = [0, 0];

	// append the svg object to the body of the page
	// append a 'group' element to 'svg'
	// moves the 'group' element to the top left margin
	let svg = d3.select("#feature_overview").append("svg")
		.attr("class", "feature f"+_idx)
	    .attr("width", width + margin.left + margin.right)
	    .attr("height", height + margin.top + margin.bottom)
	  .append("g")
	    .attr("transform",
	          "translate(" + margin.left + "," + margin.top + ")");
	    /*
    // set the ranges
	let gx = d3.scaleBand()
	          .range([0, width])
	          .padding(0.1);
	let gy = d3.scaleLinear()
	          .range([height, 0]);

	let gsvg = d3.select("#feature_overview").append("svg")
		.attr("class", "gaze f"+_idx)
	    .attr("width", width + margin.left + margin.right)
	    .attr("height", height + margin.top + margin.bottom)
	  .append("g")
	    .attr("transform",
	          "translate(" + margin.left + "," + margin.top + ")");
*/


    // add brush
    let brush = d3.brushX()
    	.extent([[0,0], [width, height]])
    	.on("end", brushed);

	svg.append("g")
		.attr("class","brush "+_idx)
		.call(brush);
	
	d3.select("#feature_overview").selectAll("."+"f"+_idx)
		.append("text")
		.attr("class", "f"+_idx+"v0")
		.attr("x", 0)
		.attr("y", 0);

	d3.select("#feature_overview").selectAll("."+"f"+_idx)		
		.append("text")
		.attr("class", "f"+_idx+"v1")
		.attr("x", 0)
		.attr("y", 0);
	
	d3.select("#feature_overview").selectAll("."+"f"+_idx)		
		.append("text")
		.attr("class", "f"+_idx+"vc")
		.attr("x", 0)
		.attr("y", 0);

	function cvtArea(_arr, _w){
		let _cvt = [0, 0];
		_cvt[0] = _arr[0]/_w;
		_cvt[1] = _arr[1]/_w;
		return _cvt;
	}

	function reverseCvt(_v, _w){
		return _v*_w;
	}

    function brushed(){
    	if(!d3.event.sourceEvent) return;
    	if(!d3.event.selection) return;
    	//console.log(d3.event.selection);
    	let areaArray = d3.event.selection;
    	let convertedArea = cvtArea(areaArray, width);
    	brushedArea = convertedArea;
    	let svg = d3.select("#feature_overview").selectAll("svg");

    	let feat = svg.selectAll("."+"feature f"+_idx);
    	let gText = svg.selectAll("text");

    	for(let i=0; i<2; i++){
    		svg.selectAll(".f"+_idx+"v"+i)
	    		.transition().delay(50)
	    		.attr("x", reverseCvt(brushedArea[i], width)+30)
	    		.attr("y", 15)
	    		.style("font-size", "13")
	    		.text(brushedArea[i].toFixed(3));
    	}

    	let _records = countingRecords(NAME_FEATURES[_idx], brushedArea);

    	svg.selectAll(".f"+_idx+"vc")
    		.transition().delay(50)
    		.attr("x", reverseCvt(brushedArea[1], width)+30)
    		.attr("y", 50)
    		.style("font-size", "10")
    		.text("records: "+_records);
    }

    /*
    // draw feature overview on stimulus
	let xpos = 0;
	let ypos = 0;
	let cellWidth = 1;
	let cellHeight = 1;
	let rowNum = 1;
	let colNum = 1;

	let SVG_width = 100;
	let SVG_height = 100;

  	if(STIMULUS_WIDTH>STIMULUS_HEIGHT){
  		let _r = SVG_width/STIMULUS_WIDTH;
  		SVG_width = STIMULUS_WIDTH*_r;
  		SVG_height = STIMULUS_HEIGHT*_r;
  	}else{
  		let _r = SVG_height/STIMULUS_HEIGHT;
  		SVG_width = STIMULUS_WIDTH*_r;
  		SVG_height = STIMULUS_HEIGHT*_r;
  	}

	// set cell width & height 
	cellWidth = SVG_width/rowNum;
	cellHeight = SVG_height/colNum;

	for(let _r=0; _r<rowNum; _r++){
		let _row = [];

		for(let _c=0; _c<colNum; _c++){
			let _col = {
				x: xpos,
				y: ypos,
				width: cellWidth,
				height: cellHeight,
				group: 0
			};

			_row.push(_col);
			xpos += cellWidth;
		}
		stiGrid.push(_row);

		xpos = 0;
		ypos += cellHeight;
	}

	let grid = d3.select("#feature_overview").append("svg")
		.attr("width", SVG_width)
		.attr("height", SVG_height);

	let stimulus = grid.append("image")
		.attr("xlink:href", "http://127.0.0.1:8000/static/stimulus/U0121_1RTE.jpg")
		.attr("width", SVG_width)
		.attr("height", SVG_height);

	let row = grid.selectAll(".row")
		.data(stiGrid)
		.enter().append("g")
		.attr("class", "row");

	let column = row.selectAll(".square")
		.data(function(d){return d;})
		.enter().append("rect")
		.attr("class", "square")
		.attr("opacity", STIMULUS_OPACITY)
		.attr("x", function(d){return d.x;})
		.attr("y", function(d){return d.y;})
		.attr("width", function(d){return d.width;})
		.attr("height", function(d){return d.height;})
		.style("fill", "#fff")
		.style("stroke", "#222");
	*/

	// get the data
	let data = _featsJSON;

	// Scale the range of the data in the domains
	x.domain(data.map(function(d) { return d.level; }));
	y.domain([0, d3.max(data, function(d) { return d.value; })]);

	// append the rectangles for the bar chart
	svg.selectAll(".bar")
	  .data(data)
	.enter().append("rect")
	  .attr("class", "bar")
	  .attr("x", function(d) { return x(d.level); })
	  .attr("width", x.bandwidth())
	  .attr("y", function(d) { return y(d.value); })
	  .attr("height", function(d) { return height - y(d.value); });

  	svg.append("text")
  		.attr("x", 0-(margin.left/2))
  		.attr("y", 0-(margin.top/3))
  		.attr("text-anchor", "middle")
  		.style("font-size", "14px")
  		.text("f_"+featureCounting);
	//featureCounting++;
	

	// add the x Axis
	svg.append("g")
	  .attr("transform", "translate(0," + height + ")")
	  .call(d3.axisBottom(x));

	// add the y Axis
	svg.append("g")
	  .call(d3.axisLeft(y));

	/*	
	let gData = recds_feat_JSON;

	gx.domain(gData.map(function(d){return d.level;}));
	gy.domain([0, d3.max(gData, function(d){return d.value;})]);

	gsvg.selectAll(".bar")
	  .data(gData)
	.enter().append("rect")
	  .attr("class", "bar")
	  .attr("x", function(d) {console.log(gx(d.level)); return gx(d.level); })
	  .attr("width", x.bandwidth())
	  .attr("y", function(d) { return gy(d.value); })
	  .attr("height", function(d) { return height - gy(d.value); });

  	gsvg.append("text")
  		.attr("x", 0-(margin.left/2))
  		.attr("y", 0-(margin.top/3))
  		.attr("text-anchor", "middle")
  		.style("font-size", "14px")
  		.text("g_"+featureCounting);

	// add the x Axis
	gsvg.append("g")
	  .attr("transform", "translate(0," + height + ")")
	  .call(d3.axisBottom(gx));

	// add the y Axis
	gsvg.append("g")
	  .call(d3.axisLeft(gy));
*/

  	featureCounting++;

}

function drawStimulusFeature_URL(_dataURL){
	// set the dimensions and margins of the graph
	let margin = {top: 20, right: 20, bottom: 30, left: 50},
	    width = 350 - margin.left - margin.right,
	    height = 150 - margin.top - margin.bottom;

	// set the ranges
	let x = d3.scaleBand()
	          .range([0, width])
	          .padding(0.1);
	let y = d3.scaleLinear()
	          .range([height, 0]);

  	let brushedArea = [0, 0];

	// append the svg object to the body of the page
	// append a 'group' element to 'svg'
	// moves the 'group' element to the top left margin
	var svg = d3.select("#feature_overview").append("svg")
	    .attr("width", width + margin.left + margin.right)
	    .attr("height", height + margin.top + margin.bottom)
	  .append("g")
	    .attr("transform",
	          "translate(" + margin.left + "," + margin.top + ")");

    // add brush
    let brush = d3.brushX()
    	.extent([[0,0], [width, height]])
    	.on("end", brushed);

	svg.append("g")
		.attr("class","brush")
		.call(brush);
		
	svg.append("text")
		.attr("class", " v0")
		.attr("x", 0)
		.attr("y", 0);

	svg.append("text")
		.attr("class", " v1")
		.attr("x", 0)
		.attr("y", 0);


	function cvtArea(_arr, _w){
		let _cvt = [0, 0];
		_cvt[0] = _arr[0]/_w;
		_cvt[1] = _arr[1]/_w;
		return _cvt;
	}

	function reverseCvt(_v, _w){
		return _v*_w;
	}

    function brushed(){
    	if(!d3.event.sourceEvent) return;
    	if(!d3.event.selection) return;
    	//console.log(d3.event.selection);
    	let areaArray = d3.event.selection;
    	let convertedArea = cvtArea(areaArray, width);
    	console.log(convertedArea);
    	brushedArea = convertedArea;

    	let svg = d3.select("#feature_overview").selectAll("svg");

    	svg.selectAll(".v0")
    		.transition().delay(100)
    		.attr("x", reverseCvt(brushedArea[0], width)-15)
    		.attr("y", 0)
    		.text(brushedArea[0].toFixed(3));

		svg.selectAll(".v1")
    		.transition().delay(100)
    		.attr("x", reverseCvt(brushedArea[1], width)-15)
    		.attr("y", 0)
    		.text(brushedArea[1].toFixed(3));
    }

	// get the data
	d3.csv(_dataURL, function(error, data) {
		if (error) throw error;
		//console.log(data);

		// format the data
		data.forEach(function(d) {
			d.value = +d.value;
		});

		// Scale the range of the data in the domains
		x.domain(data.map(function(d) { return d.level; }));
		y.domain([0, d3.max(data, function(d) { return d.value; })]);

		// append the rectangles for the bar chart
		svg.selectAll(".bar")
		  .data(data)
		.enter().append("rect")
		  .attr("class", "bar")
		  .attr("x", function(d) {return x(d.level);})
		  .attr("width", x.bandwidth())
		  .attr("y", function(d) {return y(d.value);})
		  .attr("height", function(d) {return height - y(d.value);});

	  	svg.append("text")
	  		.attr("x", 0-(margin.left/3))
	  		.attr("y", 0-(margin.top/3))
	  		.attr("text-anchor", "middle")
	  		.style("font-size", "14px")
	  		.text("feature_"+featureCounting);
		featureCounting++;

		// add the x Axis
		svg.append("g")
		  .attr("transform", "translate(0," + height + ")")
		  .call(d3.axisBottom(x));

		// add the y Axis
		svg.append("g")
		  .call(d3.axisLeft(y));

	});
}

function convertGazeToIdx(_g){
	let _idx = [];

	let SVG_width = DIV_WIDTH;
	let SVG_height = DIV_HEIGHT;
	let cellWidth = 30;
	let cellHeight = 30;
	let rowNum = 10;
	let colNum = 10;

	let _ratio = 1;
  	if(STIMULUS_WIDTH>STIMULUS_HEIGHT){
  		_ratio = DIV_WIDTH/STIMULUS_WIDTH;
  		SVG_width = STIMULUS_WIDTH*_ratio;
  		SVG_height = STIMULUS_HEIGHT*_ratio;
  	}else{
  		_ratio = DIV_HEIGHT/STIMULUS_HEIGHT;
  		SVG_width = STIMULUS_WIDTH*_ratio;
  		SVG_height = STIMULUS_HEIGHT*_ratio;
  	}

	// set cell width & height 
	cellWidth = SVG_width/rowNum;
	cellHeight = SVG_height/colNum;

	let _x = +_g[0]*_ratio;
	let _y = +_g[1]*_ratio;

	//console.log("x: "+_g[0]+" -> "+_x);
	//console.log("y: "+_g[1]+" -> "+_y);

	_x /= cellWidth;
	_y /= cellHeight;

	_idx[0] = Math.trunc(Math.floor(_y));
	_idx[1] = Math.trunc(Math.floor(_x));

	//console.log(_idx[0]+", "+_idx[1]);

	return _idx;
}

function convertGazeToSVG(_g){
	let _cg = [];

	let SVG_width = DIV_WIDTH;
	let SVG_height = DIV_HEIGHT;
	let cellWidth = 30;
	let cellHeight = 30;
	let rowNum = 10;
	let colNum = 10;

	let _ratio = 1;
  	if(STIMULUS_WIDTH>STIMULUS_HEIGHT){
  		_ratio = DIV_WIDTH/STIMULUS_WIDTH;
  		SVG_width = STIMULUS_WIDTH*_ratio;
  		SVG_height = STIMULUS_HEIGHT*_ratio;
  	}else{
  		_ratio = DIV_HEIGHT/STIMULUS_HEIGHT;
  		SVG_width = STIMULUS_WIDTH*_ratio;
  		SVG_height = STIMULUS_HEIGHT*_ratio;
  	}

	// set cell width & height 
	cellWidth = SVG_width/rowNum;
	cellHeight = SVG_height/colNum;

	let _x = +_g[0]*_ratio;
	let _y = +_g[1]*_ratio;

	_cg[0] = Math.floor(_x);
	_cg[1] = Math.floor(_y);

	return _cg;
}

function countingDurationInAOI(_g){
	let _count = [];

	for(let i=0; i<AOIarray.length; i++){
		_count.push(0);
	}

	for(let i=0; i<rawGazeData.length; i++){
		let _idx = convertGazeToIdx(rawGazeData[i]);

		for(let j=0; j<AOIarray.length; j++){
			for(let k=0; k<AOIarray[j].length; k++){
				let _aoiCell = AOIarray[j][k];
				if((_idx[0] == _aoiCell[0]) && (_idx[1] == _aoiCell[1])){
					_count[j]++;
					break;
				}
			}
		}
	}
	//console.log(_count);
	return _count;
}

function getFullAOITransition(){
	let _transition = [];

	for(let i=0; i<rawGazeData.length; i++){
		let gIdx = convertGazeToIdx(rawGazeData[i]);

		let flag_break = 0;
		for(let j=0; j<AOIarray.length; j++){
			if(flag_break == 1){
				break;
			}

			for(let k=0; k<AOIarray[j].length; k++){
				let _cell = AOIarray[j][k];
				if((_cell[0]==gIdx[0])&&(_cell[1]==gIdx[1])){
					_transition.push(j);
					flag_break = 1;
					break;
				}
			}
		}
	}

	return _transition;
}

function getAOITransition(){
	let _transition = [];

	for(let i=0; i<rawGazeData.length; i++){
		let gIdx = convertGazeToIdx(rawGazeData[i]);

		let flag_break = 0;
		for(let j=0; j<AOIarray.length; j++){
			if(flag_break == 1){
				break;
			}

			for(let k=0; k<AOIarray[j].length; k++){
				let _cell = AOIarray[j][k];
				if((_cell[0]==gIdx[0])&&(_cell[1]==gIdx[1])){
					_transition.push(j);
					flag_break = 1;
					break;
				}
			}
		}
	}

	let prevAOI = 0;
	for(let i=0; i<_transition.length; i++){
		let curAOI = _transition[i];
		if(i==0){
			prevAOI = curAOI;
			continue;
		}

		if(prevAOI == curAOI){
			_transition.splice(i,1);
			i--;
		}
		prevAOI = curAOI;
	}

	return _transition;
}


//drawParallelCoordinatesChart(tempDataset);

function drawParallelCoordinatesChart(_dataset){
	// set the dimensions and margins of the graph
	let margin = {top: 30, right: 10, bottom: 10, left: 0};
	let SVG_width = DIV_WIDTH;
	let SVG_height = DIV_HEIGHT;

	d3.select("#attention_explorer").selectAll("svg").remove();

	// append the svg object to the body of the page
	let svg = d3.select("#attention_explorer").append("svg")
		.attr("width", SVG_width + margin.left + margin.right)
		.attr("height", SVG_height + margin.top + margin.bottom)
		.append("g")
		.attr("transform", "translate(" + margin.left + "," + margin.top + ")");
	
	let dimensions = Object.keys(_dataset[0]);
	// For each dimension, I build a linear scale. I store all in a y object
	let y = {};
	for (i in dimensions){
		name = dimensions[i];
		let _range = [0, 1];
		if(name=="Color"){
			_range = [0, 1];
		}else if(name=="Intensity"){
			_range = [0, 1];
		}else if(name=="Orientation"){
			_range = [0, 1];
		}else{
			_range = [0, 1];
		}

		y[name] = d3.scaleLinear()
			.domain(_range)
			.range([SVG_height, 0]);
	}

	// Build the X scale -> it find the best position for each Y axis
	let x = d3.scalePoint()
		.range([0, SVG_width])
		.padding(1)
		.domain(dimensions);

	// The path function take a row of the csv as input, and return x and y coordinates of the line to draw for this raw.
	function path(d) {
		return d3.line()(dimensions.map(function(p){return [x(p), y[p](d[p])];}));
	}

	// Draw the lines
	svg.selectAll("myPath")
		.data(_dataset)
		.enter().append("path")
		.attr("d",  path)
		.style("fill", "none")
		.style("stroke", "#69b3a2")
		.style("opacity", 0.5);

	// Draw the axis:
	svg.selectAll("myAxis")
		// For each dimension of the dataset I add a 'g' element:
		.data(dimensions).enter()
		.append("g")
		// I translate this element to its right position on the x axis
		.attr("transform", function(d){return "translate(" + x(d) + ")";})
		// And I build the axis with the call function
		.each(function(d){d3.select(this).call(d3.axisLeft().scale(y[d]));})
		// Add axis title
		.append("text")
		.style("text-anchor", "middle")
		.attr("y", -9)
		.text(function(d){return d;})
		.style("fill", "black");
}

function drawParallelCoordinatesChart_group(_dataset){
	// set the dimensions and margins of the graph
	let margin = {top: 30, right: 10, bottom: 10, left: 0};
	let SVG_width = DIV_WIDTH;
	let SVG_height = DIV_HEIGHT;

	d3.select("#attention_explorer").selectAll("svg").remove();

	// append the svg object to the body of the page
	let svg = d3.select("#attention_explorer").append("svg")
		.attr("width", SVG_width + margin.left + margin.right)
		.attr("height", SVG_height + margin.top + margin.bottom)
		.append("g")
		.attr("transform", "translate(" + margin.left + "," + margin.top + ")");
	
	let dimensions = Object.keys(_dataset[0]);

	// For each dimension, I build a linear scale. I store all in a y object
	let y = {};
	for (i in dimensions){
		name = dimensions[i];
		let _range = [0, 1];
		if(name=="Color"){
			_range = [0, 1];
		}else if(name=="Intensity"){
			_range = [0, 1];
		}else if(name=="Orientation"){
			_range = [0, 1];
		}else{
			_range = [0, AOIarray.length];
		}

		y[name] = d3.scaleLinear()
			.domain(_range)
			.range([SVG_height, 0]);
	}

	// Build the X scale -> it find the best position for each Y axis
	let x = d3.scalePoint()
		.range([0, SVG_width])
		.padding(1)
		.domain(dimensions);

	// The path function take a row of the csv as input, and return x and y coordinates of the line to draw for this raw.
	function path(d) {
		return d3.line()(dimensions.map(function(p){return [x(p), y[p](d[p])];}));
	}

	let highlight = function(d){

		d3.selectAll(".line")
			.transition().duration(200)
			.style("stroke", "lightgrey")
			.style("opacity", "0.2");


		//d3.selectAll(".line"+d.group)
		d3.selectAll("."+"g"+d.group)
			.transition().duration(200)
			.style("stroke", function(d){
				return AOIcolorBrewer_12class_set3[d.group+1];
			})
			.style("opacity", "0.5");
	}

	let doNotHighlight = function(d){
		d3.selectAll(".line")
			.transition().duration(200).delay(1000)
			.style("stroke", function(d){
				return AOIcolorBrewer_12class_set3[d.group+1];
			})
			.style("opacity", "0.5");
	}

	// Draw the lines
	svg.selectAll("myPath")
		.data(_dataset)
		.enter().append("path")
		.attr("class", function (d) { return "line " + "g"+d.group } )
		.attr("d",  path)
		.style("fill", "none")
		.style("stroke", function(d){
			return AOIcolorBrewer_12class_set3[d.group+1];
		})
		.style("opacity", 0.5)
		.on("mouseover", highlight)
      	.on("mouseleave", doNotHighlight);

	// Draw the axis:
	svg.selectAll("myAxis")
		// For each dimension of the dataset I add a 'g' element:
		.data(dimensions).enter()
		.append("g")
		// I translate this element to its right position on the x axis
		.attr("transform", function(d){return "translate(" + x(d) + ")";})
		// And I build the axis with the call function
		.each(function(d){d3.select(this).call(d3.axisLeft().scale(y[d]));})
		// Add axis title
		.append("text")
		.style("text-anchor", "middle")
		.attr("y", -9)
		.text(function(d){return d;})
		.style("fill", "black");
}

function getFeaturesInAOI(_AOIarray, _rawGaze, _feats ,_featColName){
	let _features = [];

	//console.log(_AOIarray.length);
	let AOI_pts = [];
	let AOI_feats = [];
	for(let i=0; i<_AOIarray.length; i++){
		let _aoi = _AOIarray[i];
		//console.log(_aoi);
		
		let _gPtsAoi = [];
		let _fsAoi = [];
		for(let j=0; j<_aoi.length; j++){
			let _ir = _aoi[j][0];
			let _ic = _aoi[j][1];
	
			//console.log("_ir: "+_ir+", _ic:"+_ic);
			for(let k=0; k<_rawGaze.length; k++){
				let _y_r = convertGazeToIdx(_rawGaze[k])[0];
				let _x_c = convertGazeToIdx(_rawGaze[k])[1];
				//console.log("+++_y_r: "+_y_r+", _x_c:"+_x_c);
				if(_ir == _y_r && _ic == _x_c){
					//console.log("find --- " + k);
					_gPtsAoi.push([_rawGaze[k][0], _rawGaze[k][1]]);
					_fsAoi.push(_feats[k]);
				}
			}
		}
		AOI_pts.push(_gPtsAoi);
		AOI_feats.push(_fsAoi);
	
	}
	//console.log("AOI_pts");
	//console.log(AOI_pts);
	//console.log("AOI_feats");
	//console.log(AOI_feats);

	// make JSON format
	let _JSON_string_arr = [];
	for(let i=0; i<AOI_feats.length; i++){
		let _json = "[";
		for(let j=0; j<AOI_feats[i].length; j++){
			let _obj = "{";
			for(let k=0; k<_featColName.length; k++){
				_obj += '"';
				_obj += _featColName[k];
				_obj += '"';
				_obj += ': ';
				_obj += AOI_feats[i][j][k];
				if(k != _featColName.length-1){
					_obj += ', ';
				}else{
					_obj += '}';
				}
			}

			_json += _obj;
			if(j != AOI_feats[i].length-1){
				_json += ", ";
			}else{
				_json += "]";
			}
		}
		
		_JSON_string_arr.push(_json);
	}

	let JSON_arr = [];
	for(let i=0; i<_JSON_string_arr.length; i++){
		let _obj_parse = JSON.parse(_JSON_string_arr[i]);
		JSON_arr.push(_obj_parse);
	}
	

	//console.log(JSON_arr);
	return JSON_arr;
}

function getFeaturesInAOI_grouop(_AOIarray, _rawGaze, _feats ,_featColName){
	let _features = [];

	//console.log(_AOIarray.length);
	let AOI_pts = [];
	let AOI_feats = [];
	for(let i=0; i<_AOIarray.length; i++){
		let _aoi = _AOIarray[i];
		//console.log(_aoi);
		
		let _gPtsAoi = [];
		let _fsAoi = [];
		for(let j=0; j<_aoi.length; j++){
			let _ir = _aoi[j][0];
			let _ic = _aoi[j][1];
	
			//console.log("_ir: "+_ir+", _ic:"+_ic);
			for(let k=0; k<_rawGaze.length; k++){
				let _y_r = convertGazeToIdx(_rawGaze[k])[0];
				let _x_c = convertGazeToIdx(_rawGaze[k])[1];
				//console.log("+++_y_r: "+_y_r+", _x_c:"+_x_c);
				if(_ir == _y_r && _ic == _x_c){
					//console.log("find --- " + k);
					_gPtsAoi.push([_rawGaze[k][0], _rawGaze[k][1]]);
					_fsAoi.push(_feats[k]);
				}
			}
		}
		AOI_pts.push(_gPtsAoi);
		AOI_feats.push(_fsAoi);
	
	}
	//console.log("AOI_pts");
	//console.log(AOI_pts);
	//console.log("AOI_feats");
	//console.log(AOI_feats);

	// make JSON format
	let _json = "[";
	for(let i=0; i<AOI_feats.length; i++){
		for(let j=0; j<AOI_feats[i].length; j++){
			let _obj = "{";
			for(let k=0; k<_featColName.length; k++){
				_obj += '"';
				_obj += _featColName[k];
				_obj += '"';
				_obj += ': ';
				_obj += AOI_feats[i][j][k];
				if(k != _featColName.length-1){
					_obj += ', ';
				}else{
					_obj += ', ';
					_obj += '"group": ';
					_obj += i;
					_obj += '}';
				}
			}

			_json += _obj;
			if(j != AOI_feats[i].length-1){
				_json += ", ";
			}
		}

		if(i == AOI_feats.length-1){
			_json += "]";
		}else{
			_json += ", ";
		}
		
	}
	//console.log(_json);
	let _JSON = JSON.parse(_json);

	//console.log(_JSON);
	return _JSON;
}

function countingRecords(_featName, _featRange){
	let _count = 0;

	let feat_idx = 0;
	for(let i=0; i<NAME_FEATURES.length; i++){
		if(NAME_FEATURES[i] == _featName){
			feat_idx = i;
		}
	}

	let s_range = _featRange[0];
	let e_range = _featRange[1];
	for(let i=0; i<saliencyFeatures.length; i++){
		if(saliencyFeatures[i][feat_idx] > s_range && saliencyFeatures[i][feat_idx] <= e_range){
			_count++;
		}
	}

	return _count;
}

function countingRecords_idx(_idx, _featRange){
	let _count = 0;
	
	let s_range = _featRange[0];
	let e_range = _featRange[1];
	for(let i=0; i<saliencyFeatures.length; i++){
		if(saliencyFeatures[i][_idx] > s_range && saliencyFeatures[i][_idx] <= e_range){
			_count++;
		}
	}

	return _count;
}