function clearCanvase(_c){
	var _ctx = _c.getContext("2d");
	_ctx.clearRect(0,0,_c.width, _c.height);
}

function drawImage(_c){
	clearCanvase(_c);
	var _ctx = _c.getContext("2d");

	var _img = new Image();
  	_img.onload = function(){
    if(_img.width>_img.height)
    {
      var temp = w/_img.width;
      _img.width*=temp;
      _img.height*=temp;
    }
    else
    {
     var temp = h/_img.height;
     _img.width*=temp;
     _img.height*=temp;
    }
      
    _c.width=_img.width;
    _c.height=_img.height;
    //_ctx.globalAlpha = 0.5;
    _ctx.drawImage(_img, 0, 0, _c.width, _c.height);
    //_ctx.globalAlpha = 1;

    canvas_draw_gazePoints(_c, rawEyeData.gaze);
  };
  _img.src = "http://127.0.0.1:8000/static/data/stimulus/U0121_1RTE.jpg";	
}