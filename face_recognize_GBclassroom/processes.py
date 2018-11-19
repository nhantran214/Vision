import os
import time
import face_recognition
import cv2
import pickle
import pandas as pd
import shutil

'''
Faces = {
    'AiVanh':{'job':'teacher STEM','gender':'female','age':[18, 24]},
    'BaoVan':{'job':'teacher STEM','gender':'female','age':[18, 24]},
    'DangHuy':{'job':'Engineer','gender':'male','age':[18, 24]},
    'HongLinh':{'job':'Sale Marketer','gender':'female','age':[18, 24]},
    'HongThi':{'job':'teacher STEM','gender':'female','age':[25, 32]},
    'KimNgan':{'job':'teacher STEM','gender':'female','age':[25, 32]},
    'MongHuyen':{'job':'teacher STEM','gender':'female','age':[25, 32]},
    'ThanhGiang':{'job':'Manager','gender':'female','age':[25, 32]},
    'ThuThuy':{'job':'teacher STEM','gender':'female','age':[25, 32]},
    'TrangQuynh':{'job':'Consultant Lead','gender':'female','age':[18, 24]},
    'VanNhan':{'job':'Engineer','gender':'male','age':[18, 24]},
    'XuanSon':{'job':'teacher STEM','gender':'male','age':[25, 32]},
    'YenNhi':{'job':'teacher STEM','gender':'female','age':[25, 32]}
}
'''

def update_member_to_csv(nickname, name, age, gender, job):
    info = pd.read_csv('FaceDB.csv')
    df2 = pd.DataFrame([[nickname, name, job, gender, age]], columns=['nickname','name','job','gender','age'])
    info = info.append(df2)
    f = open('FaceDB.csv', 'wb')
    f.close()
    info.to_csv('FaceDB.csv',index=False, index_label='nickname', columns=['nickname','name','job','gender','age'])
    return info

def load_face_db(path='FaceDb'):
    known_face_encodings = []
    label_names = []
    names = os.listdir(path)
    for name in names:
        name_path = os.path.join(path, name)
        im_names = os.listdir(name_path)
        for im_name in im_names:
            if '.txt' in im_name:
                continue
            im_path = os.path.join(name_path, im_name)
            # im = face_recognition.load_image_file(im_path)
            im = cv2.imread(im_path)
            face_encodings = face_recognition.face_encodings(im)
            if len(face_encodings) < 1:
                print (im_name, ' loading fail')
                continue
            known_face_encodings.append(face_encodings[0])
            label_names.append(name)
            print (im_name, im.shape, 'num faces:', len(face_encodings), ' success')
    known_face_encodings_f = open('known_face_encodings.pkl', 'wb')
    label_names_f = open('label_names.pkl', 'wb')
    pickle.dump(known_face_encodings, known_face_encodings_f)
    pickle.dump(label_names, label_names_f)
    known_face_encodings_f.close()
    label_names_f.close()
    return known_face_encodings, label_names

# known_face_encodings, known_face_names = load_face_db()
known_face_encodings_f = open('known_face_encodings.pkl', 'rb')
label_names_f = open('label_names.pkl', 'rb')
known_face_encodings = pickle.load(known_face_encodings_f)
known_face_names = pickle.load(label_names_f)
known_face_encodings_f.close()
label_names_f.close()


def recognize(im_array):
    info = pd.read_csv('FaceDB.csv')
    face_locations = face_recognition.face_locations(im_array)
    face_encodings = face_recognition.face_encodings(im_array, face_locations)
    face_names = []
    infos_of_face = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            face_names.append(name)
            info_of_face = info.query('nickname == @name')
            infos_of_face.append(info_of_face.to_dict(orient='records')[0])
        else:
            face_names = None
    return infos_of_face

def train_one_member(nickname):
    nicknameFolderPath = os.path.join('FaceDb', nickname)
    imageNames = os.listdir(nicknameFolderPath)
    for im_name in imageNames:
        im_path = os.path.join(nicknameFolderPath, im_name)
        # im = face_recognition.load_image_file(im_path)
        im = cv2.imread(im_path)
        face_encodings = face_recognition.face_encodings(im)
        if len(face_encodings) < 1:
            print(im_name, ' loading fail')
            continue
        known_face_encodings.append(face_encodings[0])
        known_face_names.append(nickname)
        #
        known_face_encodings_f = open('known_face_encodings.pkl', 'wb')
        label_names_f = open('label_names.pkl', 'wb')
        pickle.dump(known_face_encodings, known_face_encodings_f)
        pickle.dump(known_face_names, label_names_f)
        known_face_encodings_f.close()
        label_names_f.close()

def addnewmember(nickname):
    members = os.listdir('FaceDb')
    if nickname not in members: os.mkdir(os.path.join('FaceDb', nickname))
    imageNames = os.listdir('tempFacePhotos')
    nicknamePhotos = []
    for imageName in imageNames:
        if nickname in imageName:
            tempImagePath = os.path.join('tempFacePhotos', imageName)
            destImagePath = os.path.join('FaceDb', nickname)
            try:
                shutil.move(tempImagePath, destImagePath)
                train_one_member(nickname)
            except shutil.Error as e:
                print (e)
                pass



if __name__ == "__main__":
    load_face_db()
