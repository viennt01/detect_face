### Cách sử dụng

Vào file dataset: tạo tên mình rồi cập nhật ảnh bằng cách chạy file dataset với lệnh:

Build dataset: python build_dataset.py --output dataset/vien
Build dataset gender: python build_dataset.py --output datasetGender/Woman


có data xong chạy 2 lệnh này để train
encode: python encode_faces.py --dataset dataset --encodings encodings.pickle --detection_method hog
encode gender: python encode_faces_gender.py --dataset dataSetGender --encodings encodings_gender.pickle --detection_method hog

Cách thực hiện:
img: python recognize_faces_image.py --encodings encodings.pickle --encodings_gender encodings_gender.pickle  --image test_images/many_people.png   
Video: python recognize_faces_video.py --encodings encodings.pickle --encodings_gender encodings_gender.pickle 
