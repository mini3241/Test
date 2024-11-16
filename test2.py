import cv2
import face_recognition

# 加载已知的人脸图像并编码
known_image = face_recognition.load_image_file("known_person.jpg")
known_face_encoding = face_recognition.face_encodings(known_image)[0]

# 初始化已知人脸的编码和名称
known_face_encodings = [known_face_encoding]
known_face_names = ["Known Person"]

# 加载需要识别的人脸图像
unknown_image = face_recognition.load_image_file("unknown_person.jpg")

# 查找未知图像中的所有人脸和面部特征编码
face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

# 将结果转换为OpenCV格式
unknown_image = cv2.cvtColor(unknown_image, cv2.COLOR_RGB2BGR)

# 遍历每一个在未知图像中检测到的人脸
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Unknown"

    # 如果在已知的人脸编码中找到了匹配
    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]

    # 在图像中绘制框和标签
    cv2.rectangle(unknown_image, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.putText(unknown_image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

# 显示结果图像
cv2.imshow("Face Recognition", unknown_image)
cv2.waitKey(0)
cv2.destroyAllWindows()