import cv2
from tqdm import tqdm
from natsort import natsorted
import os
import numpy as np

def load_images(input_path, input_type , interval=1):
    images = []
    filenames = None
    if input_type == 'directory':
        filenames = natsorted(os.listdir(input_path))
        for filename in filenames:
            img = cv2.imread(os.path.join(input_path, filename))
            if img is not None:
                images.append(img)
    elif input_type == 'video':
        cap = cv2.VideoCapture(input_path)
        counter = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if counter % interval == 0:
                images.append(frame)
            counter +=1
        cap.release()
    return images, filenames

def crop(images, roi_selection, params):
    cropped_images = []
    if roi_selection == True:
        roi = cv2.selectROI('ROI Selection', images[0])
        x, y, w, h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
        print("ROI Selected as x: ",x, " y: ",y, " w: ", w, " h: ",h)
        cv2.destroyWindow('ROI Selection')
    else:
        x, y, w, h = params[0], params[1], params[2], params[3]
    for img in tqdm(images):
        cropped_img = img[y:y+h, x:x+w]
        cropped_images.append(cropped_img)
    return cropped_images

def undistort(images, K, D):
    
    undistorted_images = []
    for img in tqdm(images):
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, D, (w,h), 1, (w,h))
        mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, newcameramtx, (w,h), 5)
        undistorted_img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
        #x, y, w, h = roi
        #undistorted_img = undistorted_img[y:y+h, x:x+w]
        #newcameramtx[0, 2] -= x
        #newcameramtx[1, 2] -= y
        undistorted_images.append(undistorted_img)
    newcameramtx[0,0] = newcameramtx[0,0]*1.33
    newcameramtx[1,1] = newcameramtx[1,1]*1.33
    print('New camera matrix: ', newcameramtx)
    return undistorted_images, newcameramtx

def save_images(images, output_path, filenames = None):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for i, img in enumerate(images):
        if filenames== None:
            cv2.imwrite(os.path.join(output_path, str(i).zfill(6)+'.jpg'), img)
        else:
            cv2.imwrite(os.path.join(output_path, filenames[i]), img)

if __name__ == '__main__':
    imgs, filenames = load_images('/home/ahmedabbas/data/bladder/cysto3.mp4', 'video', 1) #path, type: directory or video, sampling interval

    crop_params = [574, 145, 695, 790] #x, y, w, h
    K = np.array([[942.12754909237, 0, 364.731852351796],[0, 942.12754909237, 431.4927477284],[0, 0, 1]])
    D = np.array([-0.120762875460855, 0.356409561851108, 0.0180757324408566, 0.00228047768763547, 0])

    cropped_images = crop(imgs, False, crop_params) #images, roi_selection: select crop area manually from first image, parameters
    undistorted_images, newcameramtx = undistort(cropped_images, K, D) #images, K,D , returns undistorted images and new estimated camera parameters
    #New camera matrix is multiplied with 1.33, directly ready to use
    save_images(undistorted_images, 'Cysto3_Output/frames/')
    np.save('Cysto3_Output/NewIntrinsics.npy', newcameramtx)
    cameraMatrixDS = np.array([[newcameramtx[0, 0], newcameramtx[1, 1], newcameramtx[0, 2], newcameramtx[1, 2]]])
    np.savetxt('Cysto3_Output/calibDS.txt', cameraMatrixDS)


