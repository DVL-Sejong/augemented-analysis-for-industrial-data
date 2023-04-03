import cv2

def mouse_call_back(event, x, y, flags, param):  
    if event == cv2.EVENT_LBUTTONDOWN:
        for i in range(0, len(contours)):         
            r = cv2.pointPolygonTest(contours[i], (x, y), False)
            # print(r)
            if r > 0:
                print("Selected contour ", i)


dots            = cv2.imread('c0+indoor_001.jpg', cv2.IMREAD_COLOR)
dots_cpy        = cv2.cvtColor(dots, cv2.COLOR_BGR2GRAY)
(threshold, bw) = cv2.threshold(dots_cpy, 127, 255, cv2.THRESH_BINARY)
contours, hier  = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
contours        = contours[0:-1] # takes out contour bounding the whole image

cv2.namedWindow("res")
cv2.drawContours(dots, contours, -1, (0, 255, 0), 3)
for idx, c in enumerate(contours):  # numbers the contours
    x = int(sum(c[:,0,0]) / len(c))
    y = int(sum(c[:,0,1]) / len(c))
    # cv2.putText(dots, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
cv2.imshow("res", dots)
cv2.setMouseCallback('res', mouse_call_back)

cv2.waitKey()

# import cv2
 
# # Read image for contour detection
# input_image = cv2.imread("c0+indoor_001.jpg")
 
# # Make a copy to draw bounding box
# input_image_cpy = input_image.copy()
 
# # Convert input image to grayscale
# gray_img = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
 
# threshold_value = gray_img[216, 402]
# print(threshold_value)
 
# # Convert the grayscale image to binary (image binarization <a href="https://thinkinfi.com/basic-python-opencv-tutorial-function/" data-internallinksmanager029f6b8e52c="14" title="OpenCV" target="_blank" rel="noopener">opencv</a> <a href="https://thinkinfi.com/learn-python/" data-internallinksmanager029f6b8e52c="13" title="Best way to learn Python" target="_blank" rel="noopener">python</a>)
# ret, binary_img = cv2.threshold(gray_img, threshold_value, 255, cv2.THRESH_BINARY)
 
# # Invert image
# inverted_binary_img = ~ binary_img
 
# # Detect contours
# # hierarchy variable contains information about the relationship between each contours
# contours_list, hierarchy = cv2.findContours(inverted_binary_img,
#                                        cv2.RETR_TREE,
#                                        cv2.CHAIN_APPROX_SIMPLE) # Find contours
 
# # for each detected contours
# for contour_num in range(len(contours_list)):
 
#     # Draw detected contour with shape name
#     contour1 = cv2.drawContours(input_image_cpy, contours_list, contour_num, (255, 0, 255), 3)
 
#     # Find number of points of detected contour
#     end_points = cv2.approxPolyDP(contours_list[contour_num], 0.01 * cv2.arcLength(contours_list[contour_num], True), True)
 
#     # Make sure contour area is large enough (Rejecting unwanted contours)
#     if (cv2.contourArea(contours_list[contour_num])) > 10000:
 
#         # Find first point of each shape
#         point_x = end_points[0][0][0]
#         point_y = end_points[0][0][1]
 
#         # Writing shape name at center of each shape in black color (0, 0, 0)
#         text_color_black = (0, 0, 0)
 
#         # If a contour have three end points, then shape should be a Triangle
#         if len(end_points) == 3:
#             cv2.putText(input_image_cpy, 'Triangle', (point_x, point_y),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color_black, 2)
 
#         # If a contour have four end points, then shape should be a Rectangle or Square
#         elif len(end_points) == 4:
#             cv2.putText(input_image_cpy, 'Rectangle', (point_x, point_y),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color_black, 2)
 
#         # If a contour have five end points, then shape should be a Pentagon
#         elif len(end_points) == 5:
#             cv2.putText(input_image_cpy, 'Pentagon', (point_x, point_y),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color_black, 2)
 
#         # If a contour have ten end points, then shape should be a Star
#         elif len(end_points) == 10:
#             cv2.putText(input_image_cpy, 'Star', (point_x, point_y),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color_black, 2)
 
#         # If a contour have more than ten end points, then shape should be a Star
#         else:
#             cv2.putText(input_image_cpy, 'circle', (point_x, point_y),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color_black, 2)
 
#     cv2.imshow('First detected contour', contour1)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()