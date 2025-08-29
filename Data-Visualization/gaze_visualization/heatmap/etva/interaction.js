/*
$('#drawbutton').click(function(){
  $('#my_dataviz').empty();
  //draw_x_chart(rawEyeData.gaze);
  //draw_y_chart(rawEyeData.gaze);
  //draw_gaze_points(rawEyeData.gaze);
  //draw_scanpath(rawEyeData.gaze);

  draw_dx_chart(rawEyeData.gaze);
  draw_dy_chart(rawEyeData.gaze);
  //draw_heatmap(rawEyeData.gaze);
  draw_saccade_length_chart(rawEyeData.gaze);
  draw_distance_from_center_chart(rawEyeData.gaze);

  //draw_parallel_coordinates();
});

function buttonClickFunc(_idx){
  var btnIdName = ["btn_f_color", "btn_f_intensity", "btn_f_orientation", "btn_f_hog"];

  if(feat_flag[_idx] == true){
    feat_flag[_idx] = false;
    document.getElementById(btnIdName[_idx]).style.backgroundColor = "#fbb4ae";
  }else{
    feat_flag[_idx] = true;
    document.getElementById(btnIdName[_idx]).style.backgroundColor = "#4CAF50";
  }
  drawImage(c)
  canvas_draw_feature_areas(c);
}
*/

$('#btn_f_color').click(function(){
  $('#AUC_field').empty();
  $.ajax({
      type: "GET",
      url: "http://127.0.0.1:8000/static/data/features/U0121_1RTE_color_re.csv",
      aync: false,
      success: function (csvd) {
        csv_as_array = csvd;
      },
      dataType:"text",
      complete:function(){
        df_color = [];
        var _rows = csv_as_array.split("\n");
        for(var i=1; i<_rows.length-1; i++){
          var _row = _rows[i].split(",");
          var _v = {
            "Row": +_row[0],
            "Col": +_row[1],
            "Val": +_row[2]
          };
          df_color.push(_v);

        }
      }
  });

  changeBTNcolor(0);
  drawImage(c)
});

$('#btn_f_intensity').click(function(){
  $('#AUC_field').empty();
  $.ajax({
      type: "GET",
      url: "http://127.0.0.1:8000/static/data/features/U0121_1RTE_intensity_re.csv",
      aync: false,
      success: function (csvd) {
        csv_as_array = csvd;
      },
      dataType:"text",
      complete:function(){
        df_intensity = [];
        var _rows = csv_as_array.split("\n");
        for(var i=1; i<_rows.length-1; i++){
          var _row = _rows[i].split(",");
          var _v = {
            "Row": +_row[0],
            "Col": +_row[1],
            "Val": +_row[2]
          };
          df_intensity.push(_v);

        }
      }
  });

  changeBTNcolor(1);
  drawImage(c)
});


$('#btn_f_orientation').click(function(){
  $('#AUC_field').empty();
  $.ajax({
      type: "GET",
      url: "http://127.0.0.1:8000/static/data/features/U0121_1RTE_orientation_re.csv",
      aync: false,
      success: function (csvd) {
        csv_as_array = csvd;
      },
      dataType:"text",
      complete:function(){
        df_orientation = [];
        var _rows = csv_as_array.split("\n");
        for(var i=1; i<_rows.length-1; i++){
          var _row = _rows[i].split(",");
          var _v = {
            "Row": +_row[0],
            "Col": +_row[1],
            "Val": +_row[2]
          };
          df_orientation.push(_v);

        }
      }
  });

  changeBTNcolor(2);
  drawImage(c)
});

$('#btn_f_hog').click(function(){
  $('#AUC_field').empty();
  $.ajax({
      type: "GET",
      url: "http://127.0.0.1:8000/static/data/features/U0121_1RTE_hog_re.csv",
      aync: false,
      success: function (csvd) {
        csv_as_array = csvd;
      },
      dataType:"text",
      complete:function(){
        df_hog = [];
        var _rows = csv_as_array.split("\n");
        for(var i=1; i<_rows.length-1; i++){
          var _row = _rows[i].split(",");
          var _v = {
            "Row": +_row[0],
            "Col": +_row[1],
            "Val": +_row[2]
          };
          df_hog.push(_v);

        }
      }
  });
  changeBTNcolor(3);
  drawImage(c)
});

function changeBTNcolor(_idx){
  var btnIdName = ["btn_f_color", "btn_f_intensity", "btn_f_orientation", "btn_f_hog"];

  if(feat_flag[_idx] == true){
    feat_flag[_idx] = false;
    document.getElementById(btnIdName[_idx]).style.backgroundColor = "#fbb4ae";
  }else{
    feat_flag[_idx] = true;
    document.getElementById(btnIdName[_idx]).style.backgroundColor = "#4CAF50";
  }
}